import numpy as np

path = "/data/taiye/Project/train-oasis/data/minecraft_easy/0/000000.npz"
data = np.load(path)
print(data["actions"])
print(data["actions"].dtype)