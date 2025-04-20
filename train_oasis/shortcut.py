import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Mlp

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import wandb
from model.blocks import (
    PatchEmbed, 
    FinalLayer,
    TimestepEmbedder,
)
from einops import rearrange
import random
from torch.nn.utils import clip_grad_value_

# export cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            is_causal: bool = False,
            spatial_rotary_emb: RotaryEmbedding = None,

    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.spatial_rotary_emb = spatial_rotary_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k = self.q_norm(q), self.k_norm(k)

        q = rearrange(q, "B H W (h d) -> B h H W d", h=self.head_dim)
        k = rearrange(k, "B H W (h d) -> B h H W d", h=self.head_dim)
        v = rearrange(v, "B H W (h d) -> B h H W d", h=self.head_dim)

        freqs = self.spatial_rotary_emb.get_axial_freqs(H, W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "B h H W d -> B h (H W) d", B=B, h=self.head_dim)
        k = rearrange(k, "B h H W d -> B h (H W) d", B=B, h=self.head_dim)
        v = rearrange(v, "B h H W d -> B h (H W) d", B=B, h=self.head_dim)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )

        x = rearrange(x, "B h (H W) d -> B H W (h d)", B=B, H=H, W=W)
        x = x.type_as(q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1).unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1).unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

def create_targets_naive(images, batch_t, n_T, device):
    x_0 = torch.randn_like(images).to(device)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * batch_t) * x_0 + batch_t * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = int(math.log2(n_T))
    dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(device)

    return x_t, v_t, dt_base


# create batch, consisting of different timesteps and different dts(depending on total step sizes)
def create_targets(images, batch_t, labels, context_mask, model, n_T, device, bootstrap_every=8):
    current_batch_size = images.shape[0]

    # 1. create step sizes dt
    bootstrap_batch_size = current_batch_size // bootstrap_every
    log2_sections = int(math.log2(n_T))

    dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), bootstrap_batch_size // log2_sections)
    # print(f"dt_base: {dt_base}")

    dt_base = torch.cat([dt_base, torch.zeros(bootstrap_batch_size-dt_base.shape[0],)]).to(device)
    # print(f"dt_base: {dt_base}")
    
    dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]
    # print(f"dt: {dt}")

    dt_base_bootstrap = dt_base + 1
    dt_bootstrap = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]
    # print(f"dt_bootstrap: {dt_bootstrap}")

    # 2. sample timesteps t
    dt_sections = 2**dt_base

    # print(f"dt_sections: {dt_sections}")

    t = torch.cat([
        torch.randint(low=0, high=int(val.item()), size=(1,)).float()
        for val in dt_sections
        ]).to(device)
    
    t = t / dt_sections
    t_full = t[:, None, None, None]

    # 3. generate bootstrap targets:
    x_1 = images[:bootstrap_batch_size]
    x_0 = torch.randn_like(x_1)

    # get dx at timestep t
    x_t = (1 - (1-1e-5) * t_full)*x_0 + t_full*x_1

    bst_labels = labels[:bootstrap_batch_size]


    with torch.no_grad():
        v_b1 = model(x_t, t, dt_base_bootstrap, bst_labels)

    t2 = t + dt_bootstrap
    x_t2 = x_t + dt_bootstrap[:, None, None, None] * v_b1
    x_t2 = torch.clip(x_t2, -4, 4)
    
    with torch.no_grad():
        v_b2 = model(x_t2, t2, dt_base_bootstrap, bst_labels)

    v_target = (v_b1 + v_b2) / 2

    v_target = torch.clip(v_target, -4, 4)
    
    bst_v = v_target
    bst_dt = dt_base
    bst_xt = x_t

    # 4. generate flow-matching targets
    # sample t(normalized)
    # sample flow pairs x_t, v_t
    x_0 = torch.randn_like(images).to(device)
    x_1 = images
    x_t = (1 - (1 - 1e-5) * batch_t) * x_0 + batch_t * x_1
    v_t = x_1 - (1 - 1e-5) * x_0

    dt_flow = int(math.log2(n_T))
    dt_base = (torch.ones(images.shape[0], dtype=torch.int32) * dt_flow).to(device)

    # 5. merge flow and bootstrap
    bst_size_data = current_batch_size - bootstrap_batch_size

    x_t = torch.cat([bst_xt, x_t[-bst_size_data:]], dim=0)
    t = torch.cat([t_full, batch_t[-bst_size_data:]], dim=0)
    dt_base = torch.cat([bst_dt, dt_base[-bst_size_data:]], dim=0)
    v_t = torch.cat([bst_v, v_t[-bst_size_data:]], dim=0)

    # set context mask to 1 for first bootstrap_batch_size samples
    unmask = torch.ones(context_mask[:bootstrap_batch_size].shape).to(context_mask.device)
    context_mask = torch.cat([unmask, context_mask[-bst_size_data:]], dim=0)

    return x_t, v_t, t, dt_base, context_mask

def flow_matching_schedules(T):
    """
    Returns pre-computed schedules for flow matching sampling and training process.
    """
    # Linear time steps from 0 to 1
    t = torch.linspace(0, 1, T + 1)
    
    return {
        "t": t,  # time steps from 0 to 1
    }

def extract(a, t, x_shape):
    f, b = t.shape
    out = a[t]
    return out.reshape(f, b, *((1,) * (len(x_shape) - 2)))

class Shortcut(nn.Module):
    """
    Modified from Flow Matching
    """
    def __init__(self, model, n_T, device, drop_prob=0.1, add_velocity_direction_loss=False, 
                 lognorm_t=False, target_std=1.0, lognorm_mu=0.0, lognorm_sigma=1.0, training_type="shortcut", bootstrap_every=4):
        super(Shortcut, self).__init__()
        self.model = model.to(device)

        # register_buffer allows accessing dictionary produced by flow_matching_schedules
        for k, v in flow_matching_schedules(n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T  # allow only: [1, 2, 4, 8, 16, 32, 128]
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()
        self.add_velocity_direction_loss = add_velocity_direction_loss
        self.lognorm_t = lognorm_t
        self.target_std = target_std
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.training_type = training_type
        self.bootstrap_every = bootstrap_every

    def lognorm_sample(self, size):
        """
        Sample from logit-normal distribution with configurable mu and sigma
        """
        # Sample from normal distribution with specified mu and sigma
        samples = torch.randn(size).to(self.device) * self.lognorm_sigma + self.lognorm_mu
        
        # Transform to 0 to 1 using sigmoid
        samples = 1.0 / (1.0 + torch.exp(-samples))
        return samples

    def get_flow_field(self, x0, x1, t):
        """
        Compute the straight-line flow field between x0 and x1 at time t
        """
        return x1 - (1-1e-5)*x0  # (1-1e-5) is added by shortcut official implementation
    
    def get_xt(self, x0, x1, t):
        """
        Compute the intermediate point at time t between x0 and x1
        """
        return x0 + t * (x1 - (1-1e-5)*x0)  # (1-1e-5) is added by shortcut official implementation

    def forward(self, x0, x1, c):
        """
        This method is used in training, samples t randomly and computes loss
        """
        # Apply data shifting: scale data to target standard deviation
        if self.target_std != 1.0:
            # Calculate current standard deviation
            # https://github.com/hustvl/LightningDiT/blob/959d2ca76f238023bc9ff17ff2dcd094fe62be9b/vavae/ldm/models/diffusion/ddpm.py#L489
            data_std = x1.flatten().std()
            # Scale data to target standard deviation
            x1 = x1 * (self.target_std / (data_std + 1e-6))
        
        if x0.ndim == 4:  # (B, C, H, W)
            if self.lognorm_t:
                t = self.lognorm_sample((x0.shape[0], 1, 1, 1))
            else:   
                # continuous time
                # t = torch.rand(x0.shape[0], 1, 1, 1).to(self.device)  # t ~ Uniform(0, 1)
                # discrete time
                t = torch.randint(low=0, high=self.n_T, size=(x0.shape[0], 1, 1, 1), dtype=torch.float32).to(self.device)
                t /= self.n_T
        elif x0.ndim == 5:  # (B, T, C, H, W)
            if self.lognorm_t:
                t = self.lognorm_sample((x0.shape[0], x0.shape[1], 1, 1, 1))
            else:
                # continuous time
                # t = torch.rand(x0.shape[0], x0.shape[1], 1, 1, 1).to(self.device)  # t ~ Uniform(0, 1)
                # discrete time
                t = torch.randint(low=0, high=self.n_T, size=(x0.shape[0], x0.shape[1], 1, 1, 1), dtype=torch.float32).to(self.device)
                t /= self.n_T
        else:
            raise ValueError(f"x.ndim must be 4 or 5, but got {x0.ndim}")
        
        # Compute x_t at the sampled time steps
        # x_t = self.get_xt(x0, x1, t)
        
        # Compute the target vector field (straight line from x0 to x1)
        # target_field = self.get_flow_field(x0, x1, t)
        
        # Dropout context with some probability
        context_mask = torch.bernoulli(torch.ones_like(c) - self.drop_prob).to(self.device)

        self.model.eval()
        if self.training_type=="naive":
            x_t, target_field, dt = create_targets_naive(x1, t, self.n_T, self.device)
        elif self.training_type=="shortcut":
            x_t, target_field, t, dt, context_mask = create_targets(x1, t, c, context_mask, self.model, self.n_T, self.device, bootstrap_every=self.bootstrap_every)
        self.model.train()

        # Get model prediction
        pred_field = self.model(x_t, t.squeeze(), dt, c, context_mask)

        # MSE loss
        mse_loss = self.loss_mse(pred_field, target_field)
        
        # Add velocity direction loss if enabled
        if self.add_velocity_direction_loss:
            # Compute cosine similarity between predicted and target fields
            # Flatten spatial dimensions for cosine similarity
            target_flat = target_field.view(target_field.shape[0], -1)
            pred_flat = pred_field.view(pred_field.shape[0], -1)
            
            # Normalize vectors for cosine similarity
            target_norm = torch.nn.functional.normalize(target_flat, dim=1)
            pred_norm = torch.nn.functional.normalize(pred_flat, dim=1)
            
            # Compute cosine similarity
            cos_sim = (target_norm * pred_norm).sum(dim=1)
            
            # Direction loss: 1 - cosine similarity
            direction_loss = 1.0 - cos_sim.mean()
            
            # Combined loss
            return mse_loss + direction_loss
        else:
            return mse_loss

    def sample(self, n_sample, size, device, guide_w=0.0, cond=None, steps=None):
        """
        Sample from the flow model using Euler integration
        """
        if steps is None:
            steps = self.n_T
            
        # Start with random noise
        x_i = torch.randn(n_sample, *size).to(device)
        n_frames = x_i.shape[1] if len(size) > 3 else 1
        
        # Don't drop context at test time
        context_mask = torch.ones_like(cond).to(device)
        
        # Double the batch for classifier-free guidance
        cond = cond.repeat(2, 1)  # (2B, cond_dim)
        context_mask = context_mask.repeat(2, 1)  # (2B, 1)
        context_mask[n_sample:] = 0.  # Makes second half of batch context free
        
        # Time step size for Euler integration
        dt = 1.0 / steps
        
        x_i_store = []  # Keep track of generated steps in case want to plot something
        
        # Euler integration
        for i in range(steps):
            t_i = i * dt
            # print(f'sampling timestep {i+1}/{steps}, t={t_i:.4f}', end='\r')
 
            # Current time step tensor
            t_is = torch.ones(n_sample) * t_i
            t_is = t_is.to(device)
            
            # Double batch
            if x_i.ndim == 4:
                x_i = x_i.repeat(2, 1, 1, 1)
                t_is = t_is.repeat(2)
            elif x_i.ndim == 5:
                x_i = x_i.repeat(2, 1, 1, 1, 1)
                t_is = t_is.repeat(2).unsqueeze(1).repeat(1, n_frames)
            else:
                raise ValueError(f"x_i.ndim must be 4 or 5, but got {x_i.ndim}")
            
            dt_base = torch.ones_like(t_is).to(device) * math.log2(steps)

            # Get vector field prediction
            v = self.model(x_i, t_is, dt_base, cond, context_mask)
            
            # Split predictions and compute weighting for classifier-free guidance
            v1 = v[:n_sample]
            v2 = v[n_sample:]
            v = (1 + guide_w) * v1 - guide_w * v2
            
            # Keep only the first half of the batch
            x_i = x_i[:n_sample]
            
            # Euler step
            x_i = x_i + v * dt
            
            # Store intermediate results
            if i % 20 == 0 or i == steps - 1 or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

'''
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
'''
'''
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

'''

class ShortcutTransformer(nn.Module):
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
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.dt_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        # number of denoising steps can be applied
        self.denoise_timesteps = [1, 2, 4, 8, 16, 32, 128]

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    spatial_rotary_emb=self.spatial_rotary_emb,
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
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, dt, external_cond=None, context_mask=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        """

        B, C, H, W = x.shape

        # add spatial embeddings
        x = self.x_embedder(x)  # (B, C, H, W) -> (B, H/2, W/2, D) , C = 16, D = d_model
        # embed noise steps
        c = self.t_embedder(t)  # (B, D)
        dt = self.dt_embedder(dt)  # (B, D)
        c += dt
        if torch.is_tensor(external_cond):
            # to float
            external_cond = external_cond.to(self.dtype)
            if context_mask is not None:
                # context_mask is a tensor of 1s and 0s
                # we want to zero out the external_cond for the sample where context_mask is 0
                # external_cond is a tensor of shape (B, D)
                external_cond = external_cond * context_mask
            out = self.external_cond(external_cond)
            c += out
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
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


def shortcut_transformer_mnist(input_h=28, input_w=28, in_channels=1, patch_size=1, external_cond_dim=1):
    return ShortcutTransformer(
        input_h=input_h,
        input_w=input_w,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=256,
        depth=6,
        num_heads=16,
        external_cond_dim=external_cond_dim,
        gradient_checkpointing=False,
    )

def train_mnist():
    RUN_NAME = 'shortcut_mnist'
    PROJECT_NAME = 'shortcut'
    WANDB_ONLINE = True # turn this on to pipe experiment to cloud
    save_dir = f'./outputs/shortcut_outputs/{RUN_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=PROJECT_NAME, dir=save_dir, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
    # hardcoding these here
    n_epoch = 20  # 20
    batch_size = 256
    n_T = 128  # number of training time steps
    sample_T_list = [1, 2, 4, 128]
    device = "cuda:0"
    n_classes = 10
    lrate = 1e-4
    save_model = True
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    input_h = 28
    input_w = 28
    input_c = 1
    patch_size = 1  # critical: MNIST is 28x28, needs patch size 1
    USE_ONE_HOT_CLASS = True # critical for class labels as external condition
    # New parameters for improved flow matching
    add_velocity_direction_loss = False # technique from FasterDiT
    lognorm_t = False # technique from FasterDiT
    target_std = 1.0  # technique from FasterDiT: default 0.82, Target standard deviation for data shifting; disable with 1.0
    lognorm_mu = 0.0  # technique from FasterDiT: default 0.0, Mean for logit-normal distribution
    lognorm_sigma = 1.0  # technique from FasterDiT: default 1.0, Standard deviation for logit-normal distribution
    bootstrap_every = 8 # shortcut model: default 4, 1/4 portion of batch used for bootstrap self-consistency objective
    
    if USE_ONE_HOT_CLASS:
        external_cond_dim = 10  # MNIST has 10 classes
    else:
        external_cond_dim = 1

    # load model
    model = shortcut_transformer_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, 
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    flow_model = Shortcut(model, n_T=n_T, device=device, drop_prob=0.1,
                             add_velocity_direction_loss=add_velocity_direction_loss,
                             lognorm_t=lognorm_t, target_std=target_std,
                             lognorm_mu=lognorm_mu, lognorm_sigma=lognorm_sigma,
                             bootstrap_every=bootstrap_every)
    flow_model.to(device)

    # optionally load a model
    # flow_model.load_state_dict(torch.load("./data/flow_outputs/flow_model_19.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data/mnist", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(flow_model.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        flow_model.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            
            # Generate random noise as the starting point
            x_noise = torch.randn_like(x).to(device)
            
            # one-hot encoding
            if USE_ONE_HOT_CLASS:
                one_hot_c = torch.zeros(c.size(0), n_classes).to(device)
                one_hot_c.scatter_(1, c.unsqueeze(1), 1)
            else:
                one_hot_c = c.unsqueeze(1)
                
            # Train flow matching from noise to data
            loss = flow_model(x_noise, x, one_hot_c)
            loss.backward()
            
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            wandb.log({"loss": loss_ema})
            
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        flow_model.eval()
        with torch.no_grad():
            n_sample = 2 * n_classes
            for steps in sample_T_list:
                for w_i, w in enumerate(ws_test):
                    # get condition
                    int_c_i = torch.arange(0, 10).to(device)  # context cycles through mnist labels
                    if USE_ONE_HOT_CLASS:
                        c_i = torch.zeros(int_c_i.shape[0], 10).to(device)
                        c_i.scatter_(1, int_c_i.unsqueeze(1), 1)
                    else:
                        c_i = int_c_i.unsqueeze(1)
                    c_i = c_i.repeat(int(n_sample / c_i.shape[0]), 1)
                    
                    # Sample using flow matching
                    x_gen, x_gen_store = flow_model.sample(n_sample, (input_c, input_h, input_w), 
                                                        device, guide_w=w, cond=c_i, steps=steps)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(n_classes):
                        for j in range(int(n_sample / n_classes)):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k + (j * n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all * -1 + 1, nrow=10)
                    os.makedirs(os.path.join(save_dir, f"epoch-{ep}"), exist_ok=True)
                    save_image(grid, os.path.join(save_dir, f"epoch-{ep}", f"w{w}_steps{steps}.png"))
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}_steps{steps}.png")

                    if ep % 10 == 0 or ep == int(n_epoch - 1):
                        # create gif of images evolving over time, based on x_gen_store
                        fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, 
                                            sharex=True, sharey=True, figsize=(8, 3))
                        def animate_flow(i, x_gen_store):
                            print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                            plots = []
                            for row in range(int(n_sample / n_classes)):
                                for col in range(n_classes):
                                    axs[row, col].clear()
                                    axs[row, col].set_xticks([])
                                    axs[row, col].set_yticks([])
                                    plots.append(axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], 
                                                                    cmap='gray', 
                                                                    vmin=(-x_gen_store[i]).min(), 
                                                                    vmax=(-x_gen_store[i]).max()))
                            return plots
                        
                        ani = FuncAnimation(fig, animate_flow, fargs=[x_gen_store], interval=200, 
                                        blit=False, repeat=True, frames=x_gen_store.shape[0])
                        ani.save(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"), dpi=100, writer=PillowWriter(fps=5))
                        wandb.log({f"gif_ep{ep}_w{w}_steps{steps}.gif": wandb.Video(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"))})
                        print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}_steps{steps}.gif")
                    
        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(flow_model.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            print('saved model at:', os.path.join(save_dir, f"model_{ep}.pth"))

class ImageDiscriminator(nn.Module):
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
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    spatial_rotary_emb=self.spatial_rotary_emb,
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
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        """

        B, C, H, W = x.shape

        # add spatial embeddings
        x = self.x_embedder(x)  # (B, C, H, W) -> (B, H/2, W/2, D) , C = 16, D = d_model
        # embed noise steps
        if torch.is_tensor(external_cond):
            # to float
            external_cond = external_cond.to(self.dtype)
            c = self.external_cond(external_cond)
        else:
            c = torch.zeros(B, self.hidden_size).to(self.dtype)
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
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

def shortcut_discriminator_mnist(input_h=28, input_w=28, in_channels=1, patch_size=1, external_cond_dim=1):
    return ImageDiscriminator(
        input_h=input_h,
        input_w=input_w,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=256,
        depth=6,
        num_heads=16,
        external_cond_dim=external_cond_dim,
        gradient_checkpointing=False,
    )

def train_gan():
    RUN_NAME = 'pretrain_coeff0.1_gstep5'
    PROJECT_NAME = 'shortcut'
    WANDB_ONLINE = True # turn this on to pipe experiment to cloud
    save_dir = f'./outputs/shortcut_outputs/{RUN_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=PROJECT_NAME, dir=save_dir, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
    n_epoch = 20  # 20
    batch_size = 64
    n_T = 128  # number of training time steps
    sample_T_list = [1, 2, 4, 128]
    device = "cuda:0"
    n_classes = 10
    lrate_d = 1e-4
    lrate_g = 1e-5
    save_model = False
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    input_h = 28
    input_w = 28
    input_c = 1
    patch_size = 1  # critical: MNIST is 28x28, needs patch size 1
    USE_ONE_HOT_CLASS = True # critical for class labels as external condition
    # New parameters for improved flow matching
    add_velocity_direction_loss = False # technique from FasterDiT
    lognorm_t = False # technique from FasterDiT
    target_std = 1.0  # technique from FasterDiT: default 0.82, Target standard deviation for data shifting; disable with 1.0
    lognorm_mu = 0.0  # technique from FasterDiT: default 0.0, Mean for logit-normal distribution
    lognorm_sigma = 1.0  # technique from FasterDiT: default 1.0, Standard deviation for logit-normal distribution
    bootstrap_every = 8 # shortcut model: default 4, 1/4 portion of batch used for bootstrap self-consistency objective
    gan_loss_coeff = 0.1
    discriminator_train_steps = 1
    discriminator_warmup_steps = 100
    generator_train_steps = 5

    if USE_ONE_HOT_CLASS:
        external_cond_dim = 10  # MNIST has 10 classes
    else:
        external_cond_dim = 1
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = shortcut_transformer_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, 
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    flow_model = Shortcut(model, n_T=n_T, device=device, drop_prob=0.1,
                             add_velocity_direction_loss=add_velocity_direction_loss,
                             lognorm_t=lognorm_t, target_std=target_std,
                             lognorm_mu=lognorm_mu, lognorm_sigma=lognorm_sigma,
                             bootstrap_every=bootstrap_every)
    flow_model.to(device)

    discriminator = shortcut_discriminator_mnist(input_h=input_h, input_w=input_w, in_channels=input_c,
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    discriminator.to(device)

    # optionally load a model
    flow_model.load_state_dict(torch.load("outputs/shortcut_outputs/shortcut_mnist/model_19.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data/mnist", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    opt_g = torch.optim.Adam(flow_model.parameters(), lr=lrate_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lrate_d)

    global_step = 0
    loss_ema = None
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        flow_model.train()
        discriminator.train()

        # linear lrate decay
        opt_g.param_groups[0]['lr'] = lrate_g * (1 - ep / n_epoch)
        opt_d.param_groups[0]['lr'] = lrate_d * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        for x, c in pbar:
            x = x.to(device)
            c = c.to(device)
            
            # Generate random noise as the starting point
            x_noise = torch.randn_like(x).to(device)
            
            # one-hot encoding
            if USE_ONE_HOT_CLASS:
                one_hot_c = torch.zeros(c.size(0), n_classes).to(device)
                one_hot_c.scatter_(1, c.unsqueeze(1), 1)
            else:
                one_hot_c = c.unsqueeze(1)
            
            n_sample = x.shape[0]
            w = random.choice([0, 0.5, 2])
            x_gen_1, _ = flow_model.sample(n_sample // 2, (input_c, input_h, input_w), 
                                            device, guide_w=w, cond=one_hot_c[:n_sample // 2], steps=1)
            x_gen_2, _ = flow_model.sample(n_sample // 2, (input_c, input_h, input_w),
                                            device, guide_w=w, cond=one_hot_c[n_sample // 2:], steps=2)
            # Train discriminator
            if global_step % discriminator_train_steps == 0 or global_step < discriminator_warmup_steps:
                discriminator_pred = discriminator(torch.cat([x, x_gen_1.detach(), x_gen_2.detach()], dim=0), one_hot_c.repeat(2, 1))
                discriminator_pred = torch.sigmoid(discriminator_pred)
                real_pred = discriminator_pred[:n_sample]
                fake_pred = discriminator_pred[n_sample:]
                loss_d = -torch.mean(torch.log(real_pred) + torch.log(1 - fake_pred))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                real_pred = real_pred.mean().item()
                fake_pred = fake_pred.mean().item()
                loss_d = loss_d.item()

            # Train generator
            if global_step % generator_train_steps == 0:
                discriminator_pred = discriminator(torch.cat([x_gen_1, x_gen_2], dim=0), one_hot_c)
                discriminator_pred = torch.sigmoid(discriminator_pred)
                generator_loss = -torch.mean(torch.log(discriminator_pred))
                shortcut_loss = flow_model(x_noise, x, one_hot_c)
                loss = generator_loss * gan_loss_coeff + shortcut_loss
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()
                generator_loss = generator_loss.item()
                shortcut_loss = shortcut_loss.item()
                if loss_ema is None:
                    loss_ema = shortcut_loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * shortcut_loss
            pbar.set_description(f"loss: {loss_ema:.4f}")
            wandb.log({
                "loss_ema": loss_ema,
                "discriminator_loss": loss_d,
                "generator_loss": generator_loss,
                "shortcut_loss": shortcut_loss,
                "real_pred": real_pred,
                "fake_pred": fake_pred
            })
            global_step += 1
            
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        flow_model.eval()
        with torch.no_grad():
            n_sample = 2 * n_classes
            for steps in sample_T_list:
                for w_i, w in enumerate(ws_test):
                    # get condition
                    int_c_i = torch.arange(0, 10).to(device)  # context cycles through mnist labels
                    if USE_ONE_HOT_CLASS:
                        c_i = torch.zeros(int_c_i.shape[0], 10).to(device)
                        c_i.scatter_(1, int_c_i.unsqueeze(1), 1)
                    else:
                        c_i = int_c_i.unsqueeze(1)
                    c_i = c_i.repeat(int(n_sample / c_i.shape[0]), 1)
                    
                    # Sample using flow matching
                    x_gen, x_gen_store = flow_model.sample(n_sample, (input_c, input_h, input_w), 
                                                        device, guide_w=w, cond=c_i, steps=steps)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(n_classes):
                        for j in range(int(n_sample / n_classes)):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k + (j * n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all * -1 + 1, nrow=10)
                    os.makedirs(os.path.join(save_dir, f"epoch-{ep}"), exist_ok=True)
                    save_image(grid, os.path.join(save_dir, f"epoch-{ep}", f"w{w}_steps{steps}.png"))
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}_steps{steps}.png")

                    if ep % 10 == 0 or ep == int(n_epoch - 1):
                        # create gif of images evolving over time, based on x_gen_store
                        fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, 
                                            sharex=True, sharey=True, figsize=(8, 3))
                        def animate_flow(i, x_gen_store):
                            print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                            plots = []
                            for row in range(int(n_sample / n_classes)):
                                for col in range(n_classes):
                                    axs[row, col].clear()
                                    axs[row, col].set_xticks([])
                                    axs[row, col].set_yticks([])
                                    plots.append(axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], 
                                                                    cmap='gray', 
                                                                    vmin=(-x_gen_store[i]).min(), 
                                                                    vmax=(-x_gen_store[i]).max()))
                            return plots
                        
                        ani = FuncAnimation(fig, animate_flow, fargs=[x_gen_store], interval=200, 
                                        blit=False, repeat=True, frames=x_gen_store.shape[0])
                        ani.save(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"), dpi=100, writer=PillowWriter(fps=5))
                        wandb.log({f"gif_ep{ep}_w{w}_steps{steps}.gif": wandb.Video(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"))})
                        print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}_steps{steps}.gif")
                    
        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(flow_model.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            print('saved model at:', os.path.join(save_dir, f"model_{ep}.pth"))

def train_wgan():
    RUN_NAME = 'wgan_coeff0.1_gstep5_dlr1e-5_glr1e-5'
    PROJECT_NAME = 'shortcut'
    WANDB_ONLINE = True # turn this on to pipe experiment to cloud
    save_dir = f'./outputs/shortcut_outputs/{RUN_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=PROJECT_NAME, dir=save_dir, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
    n_epoch = 20  # 20
    batch_size = 64
    n_T = 128  # number of training time steps
    sample_T_list = [1, 2, 4, 128]
    device = "cuda:0"
    n_classes = 10
    lrate_d = 1e-5
    lrate_g = 1e-5
    save_model = False
    ws_test = [0.0, 0.5, 2.0]  # strength of generative guidance
    input_h = 28
    input_w = 28
    input_c = 1
    patch_size = 1  # critical: MNIST is 28x28, needs patch size 1
    USE_ONE_HOT_CLASS = True # critical for class labels as external condition
    # New parameters for improved flow matching
    add_velocity_direction_loss = False # technique from FasterDiT
    lognorm_t = False # technique from FasterDiT
    target_std = 1.0  # technique from FasterDiT: default 0.82, Target standard deviation for data shifting; disable with 1.0
    lognorm_mu = 0.0  # technique from FasterDiT: default 0.0, Mean for logit-normal distribution
    lognorm_sigma = 1.0  # technique from FasterDiT: default 1.0, Standard deviation for logit-normal distribution
    bootstrap_every = 8 # shortcut model: default 4, 1/4 portion of batch used for bootstrap self-consistency objective
    gan_loss_coeff = 0.1
    discriminator_train_steps = 1
    discriminator_warmup_steps = 100
    generator_train_steps = 5

    if USE_ONE_HOT_CLASS:
        external_cond_dim = 10  # MNIST has 10 classes
    else:
        external_cond_dim = 1
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = shortcut_transformer_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, 
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    flow_model = Shortcut(model, n_T=n_T, device=device, drop_prob=0.1,
                             add_velocity_direction_loss=add_velocity_direction_loss,
                             lognorm_t=lognorm_t, target_std=target_std,
                             lognorm_mu=lognorm_mu, lognorm_sigma=lognorm_sigma,
                             bootstrap_every=bootstrap_every)
    flow_model.to(device)

    discriminator = shortcut_discriminator_mnist(input_h=input_h, input_w=input_w, in_channels=input_c,
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    discriminator.to(device)

    # optionally load a model
    flow_model.load_state_dict(torch.load("outputs/shortcut_outputs/shortcut_mnist/model_19.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data/mnist", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    opt_g = torch.optim.Adam(flow_model.parameters(), lr=lrate_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lrate_d)

    global_step = 0
    loss_ema = None
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        flow_model.train()
        discriminator.train()

        # linear lrate decay
        opt_g.param_groups[0]['lr'] = lrate_g * (1 - ep / n_epoch)
        opt_d.param_groups[0]['lr'] = lrate_d * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        for x, c in pbar:
            x = x.to(device)
            c = c.to(device)
            
            # Generate random noise as the starting point
            x_noise = torch.randn_like(x).to(device)
            
            # one-hot encoding
            if USE_ONE_HOT_CLASS:
                one_hot_c = torch.zeros(c.size(0), n_classes).to(device)
                one_hot_c.scatter_(1, c.unsqueeze(1), 1)
            else:
                one_hot_c = c.unsqueeze(1)
            
            n_sample = x.shape[0]
            w = random.choice([0, 0.5, 2])
            x_gen_1, _ = flow_model.sample(n_sample // 2, (input_c, input_h, input_w), 
                                            device, guide_w=w, cond=one_hot_c[:n_sample // 2], steps=1)
            x_gen_2, _ = flow_model.sample(n_sample // 2, (input_c, input_h, input_w),
                                            device, guide_w=w, cond=one_hot_c[n_sample // 2:], steps=2)
            # Train discriminator
            if global_step % discriminator_train_steps == 0 or global_step < discriminator_warmup_steps:
                discriminator_pred = discriminator(torch.cat([x, x_gen_1.detach(), x_gen_2.detach()], dim=0), one_hot_c.repeat(2, 1))
                real_pred = discriminator_pred[:n_sample]
                fake_pred = discriminator_pred[n_sample:]
                loss_d = -torch.mean(real_pred) + torch.mean(fake_pred)
                opt_d.zero_grad()
                loss_d.backward()
                clip_grad_value_(discriminator.parameters(), 1)
                opt_d.step()
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)
                real_pred = real_pred.mean().item()
                fake_pred = fake_pred.mean().item()
                loss_d = loss_d.item()

            # Train generator
            if global_step % generator_train_steps == 0:
                discriminator_pred = discriminator(torch.cat([x_gen_1, x_gen_2], dim=0), one_hot_c)
                generator_loss = -torch.mean(discriminator_pred)
                shortcut_loss = flow_model(x_noise, x, one_hot_c)
                loss = generator_loss * gan_loss_coeff + shortcut_loss
                opt_g.zero_grad()
                loss.backward()
                clip_grad_value_(flow_model.parameters(), 1)
                opt_g.step()
                generator_loss = generator_loss.item()
                shortcut_loss = shortcut_loss.item()
                if loss_ema is None:
                    loss_ema = shortcut_loss
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * shortcut_loss
            pbar.set_description(f"loss: {loss_ema:.4f}")
            wandb.log({
                "loss_ema": loss_ema,
                "discriminator_loss": loss_d,
                "generator_loss": generator_loss,
                "shortcut_loss": shortcut_loss,
                "real_pred": real_pred,
                "fake_pred": fake_pred
            })
            global_step += 1
            
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        flow_model.eval()
        with torch.no_grad():
            n_sample = 2 * n_classes
            for steps in sample_T_list:
                for w_i, w in enumerate(ws_test):
                    # get condition
                    int_c_i = torch.arange(0, 10).to(device)  # context cycles through mnist labels
                    if USE_ONE_HOT_CLASS:
                        c_i = torch.zeros(int_c_i.shape[0], 10).to(device)
                        c_i.scatter_(1, int_c_i.unsqueeze(1), 1)
                    else:
                        c_i = int_c_i.unsqueeze(1)
                    c_i = c_i.repeat(int(n_sample / c_i.shape[0]), 1)
                    
                    # Sample using flow matching
                    x_gen, x_gen_store = flow_model.sample(n_sample, (input_c, input_h, input_w), 
                                                        device, guide_w=w, cond=c_i, steps=steps)

                    # append some real images at bottom, order by class also
                    x_real = torch.Tensor(x_gen.shape).to(device)
                    for k in range(n_classes):
                        for j in range(int(n_sample / n_classes)):
                            try:
                                idx = torch.squeeze((c == k).nonzero())[j]
                            except:
                                idx = 0
                            x_real[k + (j * n_classes)] = x[idx]

                    x_all = torch.cat([x_gen, x_real])
                    grid = make_grid(x_all * -1 + 1, nrow=10)
                    os.makedirs(os.path.join(save_dir, f"epoch-{ep}"), exist_ok=True)
                    save_image(grid, os.path.join(save_dir, f"epoch-{ep}", f"w{w}_steps{steps}.png"))
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}_steps{steps}.png")

                    if ep % 10 == 0 or ep == int(n_epoch - 1):
                        # create gif of images evolving over time, based on x_gen_store
                        fig, axs = plt.subplots(nrows=int(n_sample / n_classes), ncols=n_classes, 
                                            sharex=True, sharey=True, figsize=(8, 3))
                        def animate_flow(i, x_gen_store):
                            print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
                            plots = []
                            for row in range(int(n_sample / n_classes)):
                                for col in range(n_classes):
                                    axs[row, col].clear()
                                    axs[row, col].set_xticks([])
                                    axs[row, col].set_yticks([])
                                    plots.append(axs[row, col].imshow(-x_gen_store[i, (row * n_classes) + col, 0], 
                                                                    cmap='gray', 
                                                                    vmin=(-x_gen_store[i]).min(), 
                                                                    vmax=(-x_gen_store[i]).max()))
                            return plots
                        
                        ani = FuncAnimation(fig, animate_flow, fargs=[x_gen_store], interval=200, 
                                        blit=False, repeat=True, frames=x_gen_store.shape[0])
                        ani.save(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"), dpi=100, writer=PillowWriter(fps=5))
                        wandb.log({f"gif_ep{ep}_w{w}_steps{steps}.gif": wandb.Video(os.path.join(save_dir, f"epoch-{ep}", f"gif_w{w}_steps{steps}.gif"))})
                        print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}_steps{steps}.gif")
                    
        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(flow_model.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            print('saved model at:', os.path.join(save_dir, f"model_{ep}.pth"))

if __name__ == "__main__":
    train_wgan()