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
from train_oasis.model.attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from .blocks import (
    PatchEmbed, 
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint
from train_oasis.model.mamba_dit import Mamba2
from train_oasis.model.ttt import TTTLinearSimple, TTTConfig
from tqdm import tqdm
import torch.distributed as dist

class RNNBlock(nn.Module):
    def __init__(self, inner_dim, rnn_config):
        super().__init__()
        self.rnn_config = rnn_config
        self.inner_dim = inner_dim
        self.combine_action_dim = getattr(rnn_config, "combine_action_dim", 0)
        inner_dim += self.combine_action_dim
        if self.combine_action_dim == 0 or rnn_config.rnn_type == "LSTM":
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(inner_dim, self.inner_dim)
        if rnn_config.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=inner_dim,
                hidden_size=rnn_config.hidden_size,
                num_layers=rnn_config.num_layers,
                batch_first=True,
                dropout=rnn_config.dropout,
                proj_size=self.inner_dim if self.inner_dim < rnn_config.hidden_size else 0,
            )
        elif rnn_config.rnn_type == "Mamba2":
            self.rnn = Mamba2(
                d_model=inner_dim,
                d_state=rnn_config.mamba_d_state,
                d_conv=rnn_config.mamba_d_conv,
                expand=rnn_config.mamba_expand,
                use_mem_eff_path=False
            )
        elif rnn_config.rnn_type == "TTT":
            self.rnn = TTTLinearSimple(TTTConfig(
                hidden_size=inner_dim,
                num_attention_heads=rnn_config.num_attention_heads,
                mini_batch_size=rnn_config.mini_batch_size,
            ))
        else:
            raise ValueError(f"Unknown rnn type: {rnn_config.rnn_type}")

    def forward(self, x):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B H W) T D")  # (BHW, T, D)
        if self.rnn_config.rnn_type == "LSTM":
            out, _ = self.rnn(x)  # out: tensor of shape (BHW, T, hidden_dim)
        elif self.rnn_config.rnn_type == "Mamba2":
            out = self.rnn(x)  # out: tensor of shape (BHW, T, D)
        elif self.rnn_config.rnn_type == "TTT":
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)  # (BHW, T)
            out, _ = self.rnn(x, position_ids)
        else:
            raise ValueError(f"Unknown rnn type: {self.rnn_config.rnn_type}")
        out = self.output_proj(out)  # (BHW, T, D)
        out = rearrange(out, "(B H W) T D -> B T H W D", B=B, H=H, W=W)  # (B, T, H, W, D)
        return out

    def inference(self, x, hidden_state, start_ids):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B H W) T D")  # (BHW, T, D)
        if self.rnn_config.rnn_type == "LSTM":
            out, new_hidden_state = self.rnn(x, hidden_state)  # out: tensor of shape (BHW, T, hidden_dim)
        elif self.rnn_config.rnn_type == "Mamba2":
            if hidden_state is None:
                conv_state, ssm_state = self.rnn.allocate_inference_cache(x.shape[0], x.shape[1], x.dtype)
            else:
                conv_state, ssm_state = hidden_state
            for step in range(x.shape[1]):
                out, conv_state, ssm_state = self.rnn.step(x[:, step:step+1, :], conv_state, ssm_state)  # out: tensor of shape (BHW, 1, D)
                if step == 0:
                    outputs = out
                else:
                    outputs = torch.cat([outputs, out], dim=1)
            out = outputs
            new_hidden_state = (conv_state, ssm_state)
        elif self.rnn_config.rnn_type == "TTT":
            position_ids = torch.arange(x.shape[1], device=x.device)
            position_ids += start_ids
            position_ids = position_ids.unsqueeze(0).expand(x.shape[0], -1)  # (BHW, T)
            out, new_hidden_state = self.rnn(x, position_ids, hidden_state)
        else:
            raise ValueError(f"Unknown rnn type: {self.rnn_config.rnn_type}")
        out = self.output_proj(out)  # (BHW, T, D)
        out = rearrange(out, "(B H W) T D -> B T H W D", B=B, H=H, W=W)  # (B, T, H, W, D)
        return out, new_hidden_state

class SpatioTemporalRNNDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        max_frames=32,
        rnn_config=None,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.max_frames = max_frames
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
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.combine_action_dim = getattr(rnn_config, "combine_action_dim", 0)
        if self.combine_action_dim > 0:
            self.combine_action_proj = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, self.combine_action_dim))
        else:
            self.combine_action_proj = None
        self.r_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.rnn = RNNBlock(hidden_size, rnn_config)
        self.r_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        all_outputs = []
        for start_idx in range(0, T, self.max_frames):
            end_idx = min(start_idx + self.max_frames, T)
            x_chunk = x[:, start_idx:end_idx, :, :, :]  # (B, chunk_size, H, W, D)
            c_chunk = c[:, start_idx:end_idx, :]  # (B, chunk_size, D)
            s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c_chunk).chunk(6, dim=-1)
            x_chunk = x_chunk + gate(self.s_attn(modulate(self.s_norm1(x_chunk), s_shift_msa, s_scale_msa)), s_gate_msa)
            x_chunk = x_chunk + gate(self.s_mlp(modulate(self.s_norm2(x_chunk), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

            # temporal block
            t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c_chunk).chunk(6, dim=-1)
            x_chunk = x_chunk + gate(self.t_attn(modulate(self.t_norm1(x_chunk), t_shift_msa, t_scale_msa)), t_gate_msa)
            x_chunk = x_chunk + gate(self.t_mlp(modulate(self.t_norm2(x_chunk), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

            all_outputs.append(x_chunk)
        x = torch.cat(all_outputs, dim=1)  # (B, T, H, W, D)

        # rnn block
        r_shift, r_scale, r_gate = self.r_adaLN_modulation(c).chunk(3, dim=-1)
        residual = x
        x = modulate(self.r_norm(x), r_shift, r_scale)
        if self.combine_action_proj is not None:
            action_feat = self.combine_action_proj(c)  # (B, T, combine_action_dim)
            action_feat = action_feat.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1)  # (B, T, H, W, combine_action_dim)
            x = torch.cat([x, action_feat], dim=-1)  # (B, T, H, W, D + combine_action_dim)
        x = self.rnn(x)
        x = residual + gate(x, r_gate)

        return x

    def inference(self, x, c, hidden_state, start_ids):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        # rnn block
        r_shift, r_scale, r_gate = self.r_adaLN_modulation(c).chunk(3, dim=-1)
        residual = x
        x = modulate(self.r_norm(x), r_shift, r_scale)
        if self.combine_action_proj is not None:
            action_feat = self.combine_action_proj(c)  # (B, T, combine_action_dim)
            action_feat = action_feat.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1)  # (B, T, H, W, combine_action_dim)
            x = torch.cat([x, action_feat], dim=-1)  # (B, T, H, W, D + combine_action_dim)
        x, new_hidden_state = self.rnn.inference(x, hidden_state, start_ids)
        x = residual + gate(x, r_gate)

        return x, new_hidden_state

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
        rnn_config=None,
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
                SpatioTemporalRNNDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    max_frames=max_frames,
                    rnn_config=rnn_config,
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
            nn.init.constant_(block.r_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.r_adaLN_modulation[-1].bias, 0)

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

    def inference(self, x, t, external_cond=None, hidden_states=None, start_ids=None):
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
        new_hidden_states = []
        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)
        for block, hidden_state in zip(self.blocks, hidden_states):
            if self.gradient_checkpointing and self.training:
                x, hidden_state = checkpoint(block.inference, x, c, hidden_state, start_ids, use_reentrant=False)
            else:
                x, hidden_state = block.inference(x, c, hidden_state, start_ids)  # (N, T, H, W, D)
            new_hidden_states.append(hidden_state)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x, new_hidden_states