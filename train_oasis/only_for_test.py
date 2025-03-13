import torch.nn.functional as F
import torch

pred = torch.randn(3,)
label = torch.ones(3,)

pred = F.sigmoid(pred)
print(F.binary_cross_entropy_with_logits(pred, label))