import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir_path)
import torch
from train_oasis.model.mamba_dit import Mamba2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rnn = Mamba2(
    d_model=64,
    d_state=16,
    headdim=16,
    device=device,
)

input_tensor = torch.randn(2, 32, 64, device=device)

b1 = rnn(input_tensor)

b2 = []
conv_state, ssm_state = rnn.allocate_inference_cache(input_tensor.shape[0], input_tensor.shape[1], dtype=input_tensor.dtype)
for i in range(input_tensor.shape[1]):
    out, conv_state, ssm_state = rnn.step(input_tensor[:, i:i+1, :], conv_state, ssm_state)
    b2.append(out)

b2 = torch.cat(b2, dim=1)

print(b1)
print(b2)