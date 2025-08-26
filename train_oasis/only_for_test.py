from torchvision.io import write_video
import torch

a = torch.randn(20, 64, 64, 3)  # Random video tensor with shape (T, C, H, W)
a = torch.clamp(a, 0, 1)  # Ensure values are in the range [0, 1]
a = (a * 255).byte()
write_video("/home/tc0786/Project/train-oasis/test.mp4", a, fps=20)