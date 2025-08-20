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

@torch.no_grad()
def pose_embeddings(poses: torch.Tensor, embed_dim: int = 8):
    """
    Generate positional embeddings for a sequence of poses.
    Args:
        poses: Tensor of shape (B, 4) where B is batch size, and 4 represents x, y, z coordinates and yaw.
        embed_dim: Dimension of each.
    Returns:
        pos_emb: Tensor of shape (B, embed_dim * 2 * 4 + 4) containing the positional embeddings.
    """
    assert poses.shape[-1] == 4, "Poses should have shape (B, 4) with x, y, z, yaw."
    B = poses.shape[0]
    pos_emb = torch.zeros(B, embed_dim * 2 * 4 + 4, device=poses.device, dtype=poses.dtype)

    poses[:, 3] = poses[:, 3] / 360.0  # Convert yaw from degrees to [0, 1] range
    for i in range(4):
        pos_emb[:, i * (embed_dim * 2 + 1)] = poses[:, i]
        for freq in range(embed_dim):
            pos_emb[:, 1 + i * (embed_dim * 2 + 1) + freq * 2] = torch.sin(poses[:, i] * (2 ** freq * torch.pi))
            pos_emb[:, 2 + i * (embed_dim * 2 + 1) + freq * 2] = torch.cos(poses[:, i] * (2 ** freq * torch.pi))

    return pos_emb

class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        attn_drop: float = 0.0,
        retrieve_num: int = 10,
        pos_emb_strategy: str = "add",
        total_pos_emb_dim: int = 3,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        assert pos_emb_strategy in ["add", "concat"], "pos_emb_strategy must be either 'add' or 'concat'"
        self.pos_emb_strategy = pos_emb_strategy
        self.total_pos_emb_dim = total_pos_emb_dim
        if pos_emb_strategy == "add":
            self.pos_emb_linear = nn.Linear(total_pos_emb_dim, dim_head * heads, bias=False)

        self.rotary_emb = rotary_emb
        self.is_causal = is_causal

        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5
        self.retrieve_num = retrieve_num
        self.offset = 100

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        if self.pos_emb_strategy == "add":
            pos_emb = self.pos_emb_linear(pos_emb) # (B, T, D)
            pos_emb = pos_emb.unsqueeze(2).unsqueeze(3)  # (B, T, 1, 1, D)
            pos_emb = pos_emb.expand(B, T, H, W, self.heads * self.head_dim)
            # q += pos_emb
            # k += pos_emb
            q = q + pos_emb
            k = k + pos_emb

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        if self.pos_emb_strategy == "concat":
            # (B, T, C)
            pos_emb = pos_emb.unsqueeze(2)  # (B, T, 1, C)
            pos_emb = pos_emb.expand(B, T, H * W * self.heads, self.total_pos_emb_dim)
            pos_emb = rearrange(pos_emb, "B T (M h) d -> (B M) h T d", B=B, T=T, M=H*W, h=self.heads)
            q = torch.cat([q, pos_emb], dim=-1)
            k = torch.cat([k, pos_emb], dim=-1)

        retrieve_q = q[:, :, :self.retrieve_num, :]
        retrieve_k = k[:, :, :self.retrieve_num, :]
        condition_q = q[:, :, self.retrieve_num:, :]
        condition_k = k[:, :, self.retrieve_num:, :]

        retrieve_q = self.rotary_emb.rotate_queries_or_keys(retrieve_q, self.rotary_emb.freqs, offset=self.offset)
        retrieve_k = self.rotary_emb.rotate_queries_or_keys(retrieve_k, self.rotary_emb.freqs, offset=self.offset)
        condition_q = self.rotary_emb.rotate_queries_or_keys(condition_q, self.rotary_emb.freqs)
        condition_k = self.rotary_emb.rotate_queries_or_keys(condition_k, self.rotary_emb.freqs)

        q = torch.cat([retrieve_q, condition_q], dim=2)
        k = torch.cat([retrieve_k, condition_k], dim=2)
        # q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        # k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
        retrieve_num=10,
        pos_emb_strategy="add",
        total_pos_emb_dim=28,
    ):
        super().__init__()
        self.is_causal = is_causal
        self.retrieve_num = retrieve_num
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
            retrieve_num=retrieve_num,
            pos_emb_strategy=pos_emb_strategy,
            total_pos_emb_dim=total_pos_emb_dim,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c, pos_emb):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa), pos_emb), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


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
        pos_condition_dim=4,
        action_condition_dim=4,
        max_frames=20,
        retrieve_num=10,
        pos_emb_strategy="add",
        pos_emb_dim=3,
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_condition_dim = pos_condition_dim
        self.action_condition_dim = action_condition_dim
        self.max_frames = max_frames
        self.retrieve_num = retrieve_num
        self.pos_emb_strategy = pos_emb_strategy
        self.pos_emb_dim = pos_emb_dim
        total_pos_emb_dim = pos_emb_dim * 2 * pos_condition_dim + pos_condition_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = nn.Linear(action_condition_dim, hidden_size) if action_condition_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                    retrieve_num=retrieve_num,
                    pos_emb_strategy=pos_emb_strategy,
                    total_pos_emb_dim=total_pos_emb_dim,
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
        assert external_cond.shape[2] == (self.action_condition_dim + self.pos_condition_dim), "External condition should have shape (B, T, action_condition_dim + pos_condition_dim)"
        pose = external_cond[:, :, self.action_condition_dim:]
        external_cond = external_cond[:, :, :self.action_condition_dim]

        pose = rearrange(pose, "b t c -> (b t) c")  # (B*T, pos_condition_dim)
        pos_emb = pose_embeddings(pose, embed_dim=self.pos_emb_dim)  # (B * T, pos_emb_dim * 2 * pos_condition_dim + pos_condition_dim)
        pos_emb = rearrange(pos_emb, "(b t) d -> b t d", b=B, t=T)  # (B, T, pos_emb_dim * 2 * pos_condition_dim + pos_condition_dim)

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
                x = checkpoint(block, x, c, pos_emb, use_reentrant=False)
            else:
                x = block(x, c, pos_emb)  # (N, T, H, W, D)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x
