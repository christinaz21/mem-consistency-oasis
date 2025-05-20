import torch

pred_action = torch.randn(2, 4, 5)  # (B, R, D)
actions = torch.randn(2, 40, 5)  # (B, N, D)
similarity_func = "euclidean"
retrieve_num = 4

pred_action = pred_action.unsqueeze(2)  # (B, R, 1, D)
actions = actions.unsqueeze(1)  # (B, 1, N, D)

if similarity_func == "cosine":
    similarity = 1 - torch.nn.functional.cosine_similarity(actions, pred_action, dim=-1)
elif similarity_func == "euclidean":
    similarity = torch.norm(actions - pred_action, dim=-1)
print(similarity.shape)  # (B, R, N)

# retrieve the top-k most similar actions
topk_idx = torch.topk(similarity, 1, dim=-1, largest=False).indices.squeeze(-1)  # (B, R)
print(topk_idx.shape)  # (B, R)

# (B, retrieve_num)
topk_idx, _ = torch.sort(topk_idx, dim=-1)

print(topk_idx.shape)  # (B, R)
print(topk_idx)  # (B, R)