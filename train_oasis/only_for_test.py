from einops import rearrange
import torch

a = torch.randn(2, 3, 256, 256)

a = rearrange(a, "b c h w -> 1 (b c) h w")

print(a.shape)  # (1, 6, 256, 256)