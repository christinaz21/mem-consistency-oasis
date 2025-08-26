"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from train_oasis.model.rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from train_oasis.model.attention import SpatialAxialAttention
from timm.models.vision_transformer import Mlp
from .blocks import (
    PatchEmbed, 
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        attn_drop: float = 0.0,
        lstm_dropout: float = 0.0,
        lstm_layer_num: int = 1,
        inner_window_size: int = 10,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

        self.lstm_layer_num = lstm_layer_num
        self.inner_window_size = inner_window_size
        self.lstm = nn.LSTM(
            input_size=self.inner_dim,
            hidden_size=self.inner_dim,
            num_layers=lstm_layer_num,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        assert T % self.inner_window_size == 0, f"Expect training frames {T} to divide inner window size {self.inner_window_size}"
        block_num = T // self.inner_window_size
        outputs = []
        h0 = torch.zeros(self.lstm_layer_num, B * H * W, self.inner_dim, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(self.lstm_layer_num, B * H * W, self.inner_dim, device=x.device, dtype=x.dtype)
        
        for block in range(block_num):
            x_block = x[:, block * self.inner_window_size:(block + 1) * self.inner_window_size, :, :, :]  # (B, inner_window_size, H, W, D)
            q, k, v = self.to_qkv(x_block).chunk(3, dim=-1)

            q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
            k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
            v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

            q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
            k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

            q, k, v = map(lambda t: t.contiguous(), (q, k, v))
            x_block = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

            # LSTM
            x_block = rearrange(x_block, "BHW h T d -> BHW T (h d)")

            out, (h0, c0) = self.lstm(x_block, (h0, c0))  # out: tensor of shape (BHW, T, D)
            x_block = x_block + out  # residual connection

            x_block = rearrange(x_block, "(B H W) T D -> B T H W D", B=B, H=H, W=W)
            x_block = x_block.type_as(q)
            outputs.append(x_block)

        x = torch.cat(outputs, dim=1)  # (B, T, H, W, D)
        # linear proj
        x = self.to_out(x)
        return x

    def inference(self, x: torch.Tensor, output_buffer, state_buffer, update = False):
        """
        x: (B, T, H, W, D), the last frame will be denoised
        output_buffer: (BHW, T, D)
        state_buffer: ((num_layers, BHW, D), (num_layers, BHW, D)) h, c
        """
        B, T, H, W, D = x.shape
        BHW = B * H * W

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        # LSTM
        x = rearrange(x, "BHW h T d -> BHW T (h d)")

        last_frame = x[:, -1:, :]  # (BHW, 1, D)
        out, (h_new, c_new) = self.lstm(last_frame, state_buffer)  # out: tensor of shape (BHW, 1, D)
        if output_buffer is not None:
            if output_buffer.shape[1] == T:
                output_buffer = output_buffer[:, 1:, :]  # remove the first frame
            out = torch.cat([output_buffer, out], dim=1)  # append the new frame (BHW, T, D)
            
        if update:
            output_buffer = out  # update the output buffer
            state_buffer = (h_new, c_new)  # update the state buffer
        x = x + out  # residual connection

        x = rearrange(x, "(B H W) T D -> B T H W D", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x, output_buffer, state_buffer

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        lstm_dropout: float = 0.0,
        lstm_layer_num: int = 1,
        inner_window_size: int = 10,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
            lstm_dropout=lstm_dropout,
            lstm_layer_num=lstm_layer_num,
            inner_window_size=inner_window_size,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x

    def inference(self, x, c, output_buffer, state_buffer, update):
        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        t_attn_output, output_buffer, state_buffer = self.t_attn.inference(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa), output_buffer, state_buffer, update)
        x = x + gate(t_attn_output, t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x, output_buffer, state_buffer

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
        lstm_dropout=0.0,
        lstm_layer_num=1,
        inner_window_size=10,
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    lstm_dropout=lstm_dropout,
                    lstm_layer_num=lstm_layer_num,
                    inner_window_size=inner_window_size,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if torch.is_tensor(external_cond):
            c += self.external_cond(external_cond)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (N, T, H, W, D)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x

    def inference(self, x, t, external_cond=None, output_buffer_list=None, state_buffer_list=None, update=False):
        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", t=T)
        if torch.is_tensor(external_cond):
            c += self.external_cond(external_cond)
        new_output_buffer_list = []
        new_state_buffer_list = []
        if output_buffer_list is None:
            output_buffer_list = [None] * len(self.blocks)
        if state_buffer_list is None:
            state_buffer_list = [None] * len(self.blocks)
        for block, output_buffer, state_buffer in zip(self.blocks, output_buffer_list, state_buffer_list):
            if self.gradient_checkpointing and self.training:
                x, output_buffer, state_buffer = checkpoint(block.inference, x, c, output_buffer, state_buffer, update, use_reentrant=False)
            else:
                x, output_buffer, state_buffer = block.inference(x, c, output_buffer, state_buffer, update)  # (N, T, H, W, D)
            new_output_buffer_list.append(output_buffer)
            new_state_buffer_list.append(state_buffer)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x, new_output_buffer_list, new_state_buffer_list