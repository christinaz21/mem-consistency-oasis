"""
Based on https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/attention.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        enable_flash_attn: bool = False,
        attn_drop: float = 0.0,
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

        self.enable_flash_attn = enable_flash_attn
        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q, self.rotary_emb.freqs)
        k = self.rotary_emb.rotate_queries_or_keys(k, self.rotary_emb.freqs)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        if self.enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
            x = rearrange(x, "B N H D -> B H N D", B=B * H * W, N=T, H=self.heads)
        else:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal)

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x


class SpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        enable_flash_attn: bool = False,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

        self.rotary_emb = rotary_emb
        self.enable_flash_attn = enable_flash_attn
        self.attn_drop = attn_drop
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        freqs = self.rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        if self.enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            x = rearrange(x, "B N H D -> B H N D", B=B * T, N= H * W, H=self.heads)
        else:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)

        # linear proj
        x = self.to_out(x)
        return x

approx_gelu = lambda: nn.GELU(approximate="tanh")

def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        rope=None,
        qk_norm_legacy: bool = False,
        use_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.use_causal_mask = use_causal_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = rearrange(x, "b t h w c -> b (t h w) c")
        N = x.shape[1]
        # flash attn is not memory efficient for small sequences, this is empirical
        qkv = self.qkv(x)
        # qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        # qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        qkv = rearrange(qkv, "b n (u h d) -> u b h n d", u=3, b=B, n=N, h=self.num_heads, d=self.head_dim)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if self.use_causal_mask and self.training:
            # query (B, ..., heads, N, dim)
            # attn_mask: bool (B,..., N, N)
            causal_mask = torch.tril(torch.ones(T, T))
            causal_mask = causal_mask.repeat_interleave(H * W, dim=0)  # Expand rows
            causal_mask = causal_mask.repeat_interleave(H * W, dim=1)  # Expand columns
            causal_mask = causal_mask.bool().to(q.device)
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=causal_mask, is_causal=False) # (B, H, N, D)
        else:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x_output_shape = (B, N, C)
        # x = x.reshape(x_output_shape)
        x = rearrange(x, "b n h d -> b n (h d)", b=B, n=N, h=self.num_heads, d=self.head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, "b (t h w) c -> b t h w c", t=T, h=H, w=W)
        return x