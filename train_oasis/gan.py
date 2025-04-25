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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def shortcut_transformer_mnist(input_h=28, input_w=28, in_channels=1, patch_size=1, external_cond_dim=1):
    return ShortcutTransformer(
        input_h=input_h,
        input_w=input_w,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=128,
        depth=4,
        num_heads=16,
        external_cond_dim=external_cond_dim,
        gradient_checkpointing=False,
    )

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

# Discriminator
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Linear(10, 28 * 28)  # 嵌入层用于条件
        self.initial = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.resnet_blocks = nn.Sequential(
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 512, stride=2),
        )
        self.fc =  nn.Linear(512 * 4 * 4, 1)

    def forward(self, x, labels):
        c = self.label_embedding(labels).view(labels.size(0), 1, 28, 28)  # 嵌入条件
        x = torch.cat([x, c], dim=1)  # 将图像和条件拼接
        x = self.initial(x)
        x = self.resnet_blocks(x)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc(x)

def shortcut_discriminator_mnist(input_h=28, input_w=28, in_channels=1, patch_size=1, external_cond_dim=1):
    return ImageDiscriminator(
        input_h=input_h,
        input_w=input_w,
        in_channels=in_channels,
        patch_size=patch_size,
        hidden_size=64,
        depth=2,
        num_heads=8,
        external_cond_dim=external_cond_dim,
        gradient_checkpointing=False,
    )

def train_gan():
    RUN_NAME = 'gan_gstep5_dlr1e-5_ResNet'
    PROJECT_NAME = 'shortcut'
    WANDB_ONLINE = False # turn this on to pipe experiment to cloud
    save_dir = f'./outputs/shortcut_outputs/{RUN_NAME}'
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=PROJECT_NAME, dir=save_dir, mode='disabled' if not WANDB_ONLINE else 'online')
    wandb.run.name = RUN_NAME
    wandb.run.save()
    n_epoch = 20  # 20
    batch_size = 256
    device = "cuda:0"
    n_classes = 10
    lrate_d = 1e-5
    lrate_g = 1e-4
    save_model = False
    input_h = 28
    input_w = 28
    input_c = 1
    patch_size = 1  # critical: MNIST is 28x28, needs patch size 1
    USE_ONE_HOT_CLASS = True # critical for class labels as external condition
    discriminator_train_steps = 1
    generator_train_steps = 5

    if USE_ONE_HOT_CLASS:
        external_cond_dim = 10  # MNIST has 10 classes
    else:
        external_cond_dim = 1
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = shortcut_transformer_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, 
                     patch_size=patch_size, external_cond_dim=external_cond_dim)
    model.to(device)

    # discriminator = shortcut_discriminator_mnist(input_h=input_h, input_w=input_w, in_channels=input_c, patch_size=patch_size, external_cond_dim=external_cond_dim)
    discriminator = Discriminator()
    discriminator.to(device)

    # optionally load a model
    # model.load_state_dict(torch.load("outputs/shortcut_outputs/shortcut_mnist/model_19.pth"))

    tf = transforms.Compose([transforms.ToTensor()])  # mnist is already normalised 0 to 1

    dataset = MNIST("./data/mnist", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    opt_g = torch.optim.Adam(model.parameters(), lr=lrate_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lrate_d)

    global_step = 0
    for ep in range(n_epoch):
        print(f'epoch {ep}')
        model.train()
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
            
            x_pred = model(x_noise, one_hot_c)
            # Train discriminator
            if global_step % discriminator_train_steps == 0:
                discriminator_pred = discriminator(torch.cat([x, x_pred.detach()], dim=0), one_hot_c.repeat(2, 1))
                discriminator_pred = torch.sigmoid(discriminator_pred)
                real_pred = discriminator_pred[:x.shape[0]]
                fake_pred = discriminator_pred[x.shape[0]:]
                loss_d = -torch.mean(torch.log(real_pred) + torch.log(1 - fake_pred))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                real_pred = real_pred.mean().item()
                fake_pred = fake_pred.mean().item()
                loss_d = loss_d.item()

            # Train generator
            if global_step % generator_train_steps == 0:
                discriminator_pred_2 = discriminator(x_pred, one_hot_c)
                discriminator_pred_2 = torch.sigmoid(discriminator_pred_2)
                generator_loss = -torch.mean(torch.log(discriminator_pred_2))
                opt_g.zero_grad()
                generator_loss.backward()
                opt_g.step()
                generator_loss = generator_loss.item()
                fake_pred2 = discriminator_pred_2.mean().item()
            # with torch.no_grad():
            #     print("After", torch.sigmoid(discriminator(x_pred, one_hot_c)).mean().item())
            pbar.set_description(f"g_l: {generator_loss:.4f} d_l: {loss_d:.4f} r_p: {real_pred:.4f} f_p: {fake_pred:.4f} f_p2: {fake_pred2:.4f}")
            wandb.log({
                "discriminator_loss": loss_d,
                "generator_loss": generator_loss,
                "real_pred": real_pred,
                "fake_pred": fake_pred
            })
            global_step += 1
            
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        model.eval()
        with torch.no_grad():
            n_sample = 2 * n_classes
            # get condition
            int_c_i = torch.arange(0, 10).to(device)  # context cycles through mnist labels
            if USE_ONE_HOT_CLASS:
                c_i = torch.zeros(int_c_i.shape[0], 10).to(device)
                c_i.scatter_(1, int_c_i.unsqueeze(1), 1)
            else:
                c_i = int_c_i.unsqueeze(1)
            c_i = c_i.repeat(int(n_sample / c_i.shape[0]), 1)
            
            # Sample using flow matching
            x_noise = torch.randn(n_sample, input_c, input_h, input_w).to(device)
            x_gen = model(x_noise, c_i)

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
            os.makedirs(save_dir, exist_ok=True)
            save_image(grid, os.path.join(save_dir, f"epoch-{ep}.png"))
            print(os.path.join(save_dir, f"epoch-{ep}.png"))

        # optionally save model
        if save_model and ep == int(n_epoch - 1):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{ep}.pth"))
            print('saved model at:', os.path.join(save_dir, f"model_{ep}.pth"))

if __name__ == "__main__":
    train_gan()