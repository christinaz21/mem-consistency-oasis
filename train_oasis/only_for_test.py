<<<<<<< HEAD
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
=======
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

def get_data_paths(save_dir):
    all_path = []
    for sub_dir in save_dir.glob("*/"):
        paths = list(sub_dir.glob("*.npz"))
        all_path.extend(paths)
    return all_path

def get_data_lengths(save_dir):
    paths = get_data_paths(save_dir)
    total_files = len(paths)

    def process_file(path):
        try:
            data = np.load(path)["actions"]
            line_count = len(data)
            return str(path), line_count
        except Exception as e:
            print(f"Skipping file {path} due to error: {e}")
            return None
    lengths = []
    for path in tqdm(paths, total=total_files, desc="Processing files"):
        result = process_file(path)
        if result is not None:
            lengths.append({
                "file": result[0],
                "length": result[1]
            })
    return lengths

save_dir = "/home/tc0786/Project/train-oasis/data/mc_collect_data"
metadata_path = "/home/tc0786/Project/train-oasis/data/mc_collect_data/metadata.json"
validation_size = 100
save_dir = Path(save_dir)
all_data = get_data_lengths(save_dir)
json.dump(
    {
        "training": all_data[validation_size:],
        "validation": all_data[: validation_size],
    },
    open(metadata_path, "w"),
    indent=4,
)
>>>>>>> refs/remotes/origin/master
