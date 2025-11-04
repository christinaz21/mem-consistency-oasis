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
        self.one_mem = getattr(rnn_config, "one_mem", False)
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

    def forward(self, x, hidden_state=None):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B H W) T D")  # (BHW, T, D)
        use_one_mem = self.one_mem and (T > 1)
        if self.rnn_config.rnn_type == "LSTM":
            if use_one_mem:
                assert hidden_state is not None, "hidden_state should not be None when one_mem is True"
                h_0, c_0 = hidden_state  # each of shape (num_layers, BHW_total, hidden_size)
                h_0 = h_0.repeat(1, T, 1)  # (num_layers, BHW_total * T, hidden_size)
                c_0 = c_0.repeat(1, T, 1)  # (num_layers, BHW_total * T, hidden_size)
                x = rearrange(x, "BHW T D -> (BHW T) 1 D")  # (BHW_total * T, 1, D)
                out, _ = self.rnn(x, (h_0, c_0))  # out: tensor of shape (BHW * T, 1, hidden_dim)
                out = rearrange(out, "(BHW T) 1 D -> BHW T D", BHW=B*H*W, T=T)  # (BHW, T, D)
                new_hidden_state = None
            else:
                out, new_hidden_state = self.rnn(x, hidden_state)  # out: tensor of shape (BHW, T, hidden_dim)
        elif self.rnn_config.rnn_type == "Mamba2":
            if use_one_mem:
                conv_state, ssm_state = hidden_state
                for step in range(x.shape[1]):
                    out, _, _ = self.rnn.step(x[:, step:step+1, :], conv_state, ssm_state)  # out: tensor of shape (BHW, 1, D)
                    if step == 0:
                        outputs = out
                    else:
                        outputs = torch.cat([outputs, out], dim=1)
                out = outputs
                new_hidden_state = None
            else:
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
            if use_one_mem:
                raise NotImplementedError("one_mem is not implemented for TTT")
            else:
                position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)  # (BHW, T)
                out, new_hidden_state = self.rnn(x, position_ids, hidden_state)
        else:
            raise ValueError(f"Unknown rnn type: {self.rnn_config.rnn_type}")
        out = self.output_proj(out)  # (BHW, T, D)
        out = rearrange(out, "(B H W) T D -> B T H W D", B=B, H=H, W=W)  # (B, T, H, W, D)
        return out, new_hidden_state

    def window_size_1_forward(self, x, hidden_state=None, target_hidden_states=None):
        B, T, H, W, D = x.shape
        x = rearrange(x, "B T H W D -> (B H W) T D")  # (BHW, T, D)
        if self.rnn_config.rnn_type == "LSTM":
            all_hidden_states = []
            last_target_idx = 0
            for target_idx in target_hidden_states:
                # if dist.get_rank() == 0:
                #     print(f"Processing target index: {target_idx}, last target index: {last_target_idx}")
                out, hidden_state = self.rnn(x[:, last_target_idx:target_idx, :], hidden_state)
                all_hidden_states.append(hidden_state)
                if last_target_idx == 0:
                    outputs = out
                else:
                    outputs = torch.cat([outputs, out], dim=1)
                last_target_idx = target_idx
            out = outputs
        elif self.rnn_config.rnn_type == "Mamba2":
            if hidden_state is None:
                conv_state, ssm_state = self.rnn.allocate_inference_cache(x.shape[0], x.shape[1], x.dtype)
            else:
                conv_state, ssm_state = hidden_state
            all_hidden_states = []
            for step in range(x.shape[1]):
                out, conv_state, ssm_state = self.rnn.step(x[:, step:step+1, :], conv_state, ssm_state)  # out: tensor of shape (BHW, 1, D)
                if (step + 1) in target_hidden_states:
                    all_hidden_states.append((conv_state, ssm_state))
                if step == 0:
                    outputs = out
                else:
                    outputs = torch.cat([outputs, out], dim=1)
            out = outputs
        elif self.rnn_config.rnn_type == "TTT":
            all_hidden_states = []
            last_target_idx = 0
            position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)  # (BHW, T)
            for target_idx in target_hidden_states:
                # if dist.get_rank() == 0:
                #     print(f"Processing target index: {target_idx}, last target index: {last_target_idx}")
                position_ids_chunk = position_ids[:, last_target_idx:target_idx]
                out, hidden_state = self.rnn(x[:, last_target_idx:target_idx, :], position_ids_chunk, hidden_state)
                all_hidden_states.append(hidden_state)
                if last_target_idx == 0:
                    outputs = out
                else:
                    outputs = torch.cat([outputs, out], dim=1)
                last_target_idx = target_idx
            out = outputs
        else:
            raise ValueError(f"Unknown rnn type: {self.rnn_config.rnn_type}")
        out = self.output_proj(out)  # (BHW, T, D)
        out = rearrange(out, "(B H W) T D -> B T H W D", B=B, H=H, W=W)  # (B, T, H, W, D)
        return out, all_hidden_states

class SpatioTemporalRNNDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        rnn_config=None,
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

    def forward(self, x, c, hidden_state=None):
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
        x, hidden_state = self.rnn(x, hidden_state)
        x = residual + gate(x, r_gate)

        return x, hidden_state

    def window_size_1_forward(self, x, c, hidden_state=None, mini_batch_size=256, target_hidden_states=None):
        B, T, H, W, D = x.shape

        all_outputs = []
        for t_start in range(0, T, mini_batch_size):
            x_chunk = x[:, t_start:t_start+mini_batch_size, :, :, :]  # (B, chunk_size, H, W, D)
            c_chunk = c[:, t_start:t_start+mini_batch_size, :]  # (B, chunk_size, D)
            now_T = c_chunk.shape[1]
            # spatial block
            s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c_chunk).chunk(6, dim=-1)
            x_chunk = x_chunk + gate(self.s_attn(modulate(self.s_norm1(x_chunk), s_shift_msa, s_scale_msa)), s_gate_msa)
            x_chunk = x_chunk + gate(self.s_mlp(modulate(self.s_norm2(x_chunk), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

            # temporal block
            t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c_chunk).chunk(6, dim=-1)
            residual = x_chunk
            x_chunk = modulate(self.t_norm1(x_chunk), t_shift_msa, t_scale_msa)
            x_chunk = rearrange(x_chunk, "B T H W D -> (B T) H W D").unsqueeze(1)  # (B*T, 1, H, W, D)
            x_chunk = self.t_attn(x_chunk)  # (B*T, 1, H, W, D)
            x_chunk = rearrange(x_chunk.squeeze(1), "(B T) H W D -> B T H W D", B=B, T=now_T)  # (B, T, H, W, D)
            x_chunk = residual + gate(x_chunk, t_gate_msa)
            x_chunk = x_chunk + gate(self.t_mlp(modulate(self.t_norm2(x_chunk), t_shift_mlp, t_scale_mlp)), t_gate_mlp)
            all_outputs.append(x_chunk)
        x = torch.cat(all_outputs, dim=1)  # (B, T, H, W, D)
        # # spatial block
        # s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        # x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        # x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # # temporal block
        # t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        # residual = x
        # x = modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)
        # x = rearrange(x, "B T H W D -> (B T) H W D").unsqueeze(1)  # (B*T, 1, H, W, D)
        # x = self.t_attn(x)  # (B*T, 1, H, W, D)
        # x = rearrange(x.squeeze(1), "(B T) H W D -> B T H W D", B=B, T=T)  # (B, T, H, W, D)
        # x = residual + gate(x, t_gate_msa)
        # x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        # rnn block
        r_shift, r_scale, r_gate = self.r_adaLN_modulation(c).chunk(3, dim=-1)
        residual = x
        x = modulate(self.r_norm(x), r_shift, r_scale)
        if self.combine_action_proj is not None:
            action_feat = self.combine_action_proj(c)  # (B, T, combine_action_dim)
            action_feat = action_feat.unsqueeze(2).unsqueeze(2).expand(-1, -1, H, W, -1)  # (B, T, H, W, combine_action_dim)
            x = torch.cat([x, action_feat], dim=-1)  # (B, T, H, W, D + combine_action_dim)
        x, hidden_state = self.rnn.window_size_1_forward(x, hidden_state, target_hidden_states=target_hidden_states)
        x = residual + gate(x, r_gate)

        return x, hidden_state

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

    def forward(self, x, t, external_cond=None, hidden_states=None):
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
        hidden_states = [None] * len(self.blocks) if hidden_states is None else hidden_states
        new_hidden_states = []
        for block, hidden_state in zip(self.blocks, hidden_states):
            if self.gradient_checkpointing and self.training:
                x, new_hidden_state = checkpoint(block, x, c, hidden_state, use_reentrant=False)
            else:
                x, new_hidden_state = block(x, c, hidden_state=hidden_state)  # (N, T, H, W, D)
            new_hidden_states.append(new_hidden_state)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x, new_hidden_states

    def window_size_1_forward(self, x, t, external_cond=None, hidden_states=None, mini_batch_size=256, target_hidden_states=None, get_return=False):
        """
        A faster forward pass of DiT with sliding window size 1.
        x: (B, T, C, H, W) tensor of spatial inputs
        t: (B, T,) tensor of diffusion timesteps
        Note: this function will not return the output, but only the new hidden states.
        """

        x = x[:, :target_hidden_states[-1], :, :, :]
        t = t[:, :target_hidden_states[-1]]
        external_cond = external_cond[:, :target_hidden_states[-1], :] if torch.is_tensor(external_cond) else None
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
        hidden_states = [None] * len(self.blocks) if hidden_states is None else hidden_states
        all_hidden_states = [[] for _ in range(len(target_hidden_states))]
        for block, hidden_state in zip(self.blocks, hidden_states):
            if self.gradient_checkpointing and self.training:
                x, new_hidden_state = checkpoint(block.window_size_1_forward, x, c, hidden_state, mini_batch_size, target_hidden_states, use_reentrant=False)
            else:
                x, new_hidden_state = block.window_size_1_forward(x, c, hidden_state=hidden_state, mini_batch_size=mini_batch_size, target_hidden_states=target_hidden_states)  # (N, T, H, W, D)
            for t in range(len(target_hidden_states)):
                all_hidden_states[t].append(new_hidden_state[t])
        if not get_return:
            return all_hidden_states
        
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x, all_hidden_states

    def inference(self, x, t, external_cond=None, hidden_states=None, get_new_hidden_states=False):
        """
        Inference pass of DiT without returning hidden states.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """
        # forward with window size 1
        if get_new_hidden_states:
            _, new_hidden_states = self.forward(x[:, :1], t[:, :1], external_cond=external_cond[:, :1] if torch.is_tensor(external_cond) else None, hidden_states=hidden_states)
        else:
            new_hidden_states = None
        x, _ = self.forward(x, t, external_cond=external_cond, hidden_states=hidden_states)
        return x, new_hidden_states