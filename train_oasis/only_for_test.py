from model.rotary_embedding_torch import RotaryEmbedding
import torch


q = torch.ones(1, 1, 12, 8)
k = torch.ones(1, 1, 12, 8)

print(q.dtype)

rope = RotaryEmbedding(dim=8)
print(rope.freqs)
q = rope.rotate_queries_or_keys(q, rope.freqs)
k = rope.rotate_queries_or_keys(k, rope.freqs)

print(q.shape)
print(k.shape)

print(q)
print(k)