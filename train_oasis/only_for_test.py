import numpy as np
import time
import cv2
import torch
import torch.nn.functional as F

output = torch.tensor([1, 2, 3, 4, 5]).float()
target = torch.tensor([1, 1, 0, 0, 0]).float()

a = torch.sigmoid(output)
loss = F.binary_cross_entropy(a, target)
print(loss)

loss = F.binary_cross_entropy_with_logits(output, target)
print(loss)