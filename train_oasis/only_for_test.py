import numpy as np
import json

metadata_path = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/metadata_mem.json"
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

for item in metadata[4:]:
    path = item['file']
    data = np.load(path)["actions"][:, 4:]
    for d in data:
        print(d)
    break