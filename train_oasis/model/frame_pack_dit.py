"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Mlp
import numpy as np
from timm.layers.helpers import to_2tuple

from .attention import (
    Attention,
    get_layernorm,
    approx_gelu,
)
from .blocks import (
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = rearrange(x, "B C H W -> B (H W) C")
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        enable_layernorm_kernel=False,
        use_causal_mask=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            use_causal_mask=use_causal_mask,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.attn(modulate(self.norm1(x), shift_msa, scale_msa)), gate_msa)
        x = x + gate(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x

class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
        max_temporal_pos_emb=40,
        dtype=torch.float32,
        enable_layernorm_kernel=False,
        use_causal_mask=False,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.input_size = (max_frames, input_h, input_w)

        self.x_embedder_1 = PatchEmbed(2, in_channels, hidden_size, flatten=False)
        self.x_embedder_2 = PatchEmbed(4, in_channels, hidden_size, flatten=False)
        self.x_embedder_3 = PatchEmbed(8, in_channels, hidden_size, flatten=False)

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)

        self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed(max_temporal_pos_emb))

        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                    use_causal_mask=use_causal_mask,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, 2, self.out_channels)
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
        w = self.x_embedder_1.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_1.proj.bias, 0)

        w = self.x_embedder_2.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_2.proj.bias, 0)

        w = self.x_embedder_3.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder_3.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        p = 2
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    @torch.no_grad()
    def resize(self, x, size):
        """
        Resize the input tensor x to the given size.
        x: (B, C, H, W)
        size: (H', W')
        """
        return nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def patchify(self, x):
        """
        x: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        f0 = x[:, -2:, :, :, :]  # (B, 2, C, H, W)
        f0 = rearrange(f0, "b t c h w -> (b t) c h w")  # (B*T, C, H, W)
        f0 = self.x_embedder_1(f0)  # (B*2, H/2, W/2, D)
        f0 = rearrange(f0, "(b t) h w d -> b t h w d", t=2)  # (B, 2, 9, 16, D)
        f1 = x[:, -3, :, :, :]  # (B, C, 18, 32)
        f1 = self.resize(f1, (18, 16))
        f1 = self.x_embedder_1(f1)  # (B, 9, 8, D)
        f2 = x[:, -4, :, :, :]  # (B, C, 18, 32)
        f2 = self.resize(f2, (10, 16))
        f2 = self.x_embedder_1(f2)  # (B, 5, 8, D)
        f3 = x[:, -5, :, :, :]  # (B, C, 18, 32)
        f3 = self.resize(f3, (16, 16))
        f3 = self.x_embedder_2(f3)  # (B, 4, 4, D)
        f4 = x[:, -6, :, :, :]  # (B, C, 18, 32)
        f4 = self.resize(f4, (8, 16))
        f4 = self.x_embedder_2(f4)  # (B, 2, 4, D)
        f5 = x[:, -7, :, :, :] # (B, C, 18, 32)
        f5 = self.resize(f5, (16, 16))
        f5 = self.x_embedder_3(f5)  # (B, 2, 2, D)
        f6 = x[:, -8, :, :, :]  # (B, C, 18, 32)
        f6 = self.resize(f6, (8, 16))
        f6 = self.x_embedder_3(f6)  # (B, 1, 2, D)
        f7 = x[:, -9, :, :, :]  # (B, C, 18, 32)
        f7 = self.resize(f7, (8, 8))
        f7 = self.x_embedder_3(f7)  # (B, 1, 1, D)
        f8 = x[:, -10, :, :, :]  # (B, C, 18, 32)
        f8 = self.resize(f8, (8, 8))
        f8 = self.x_embedder_3(f8)  # (B, 1, 1, D)

        c = torch.cat([f8, f7], dim=2) # (B, 1, 2, D)
        c = torch.cat([c, f6], dim=1) # (B, 2, 2, D)
        c = torch.cat([c, f5], dim=2) # (B, 2, 4, D)
        c = torch.cat([c, f4], dim=1) # (B, 4, 4, D)
        c = torch.cat([c, f3], dim=2) # (B, 4, 8, D)
        c = torch.cat([c, f2], dim=1) # (B, 9, 8, D)
        c = torch.cat([c, f1], dim=2) # (B, 9, 16, D)
        c = c.unsqueeze(1)  # (B, 1, 9, 16, D)
        c = torch.cat([c, f0], dim=1)  # (B, 3, 9, 16, D)

        return c
    
    def get_spatial_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (self.input_size[1] // 2, self.input_size[2] // 2),
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self, max_temporal_pos_emb=40):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            max_temporal_pos_emb,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        # add spatial embeddings
        x = self.patchify(x)  # (B, T, H/2, W/2, D)
        B, T, H, W, D = x.shape
        t = t[:, -T:]
        external_cond = external_cond[:, -T:] if external_cond is not None else None
        # restore shape
        x = rearrange(x, "b t h w d -> b t (h w) d")
        # embed x
        x = x + self.pos_embed_spatial
        x = rearrange(x, "b t s d -> b s t d")
        x = x + self.pos_embed_temporal[:, :T, :]
        x = rearrange(x, "b (h w) t d -> b t h w d", b=B, t=T, h=H, w=W)
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
                x = block(x, c)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x