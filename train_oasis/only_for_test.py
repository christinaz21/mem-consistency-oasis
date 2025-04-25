import os

path = "/data/taiye/Project/train-oasis/data/minecraft_pos"

for file_name in os.listdir(path):
    if file_name.endswith(".tar.gz"):
        file_path = os.path.join(path, file_name)
        os.system(f"rm {file_path}")