import numpy as np

path = "data/maze/eval/20220920T045050-1000.npz"

data = np.load(path, allow_pickle=True)
for d in data['action']:
    if d[0] != 0:
        print(d)
