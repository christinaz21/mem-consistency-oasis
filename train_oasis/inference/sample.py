"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import json
import numpy as np
import torch
from train_oasis.model.vae import VAE_models
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import os
import json
from safetensors.torch import load_model
from torchvision.io import read_video

def gt():
    assert torch.cuda.is_available()
    device = "cuda"

    save_dir = ""
    metadata_path = "data/oasis500m/metadata.json"
    new_meta_data_path = "data/oasis500m/metadata_gt.json"
    os.makedirs(save_dir, exist_ok=True)
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae, "models/oasis500m/vit-l-20.safetensors")
    vae = vae.to(device).eval()

    with open(metadata_path, "r") as f:
        meta_data = json.load(f)
    all_prompt_paths = meta_data["training"]
    scaling_factor = 0.07843137255
    id = 0
    new_meta_data = []
    pbar = tqdm(range(len(all_prompt_paths)))
    for prompt_path in all_prompt_paths:
        save_path = os.path.join(save_dir, f"{id:04d}.pt")
        action_path = prompt_path["file"]
        video_path = prompt_path["file"].replace("npz", "mp4")
        assert os.path.exists(video_path), f"Video file {video_path} does not exist."
        assert os.path.exists(action_path), f"Action file {action_path} does not exist."
        action_save_path = save_path.replace("pt", "npz")
        os.system(f"cp {action_path} {action_save_path}")
        video, _, _ = read_video(str(video_path), pts_unit="sec")
        video = video.contiguous().numpy()
        video = torch.from_numpy(video).float() / 255.0
        video = video.to(torch.half).to(device)
        video = video.permute(0, 3, 1, 2).contiguous()
        B = video.shape[0]
        H, W = video.shape[-2:]
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                video = vae.encode(video * 2 - 1).mean * scaling_factor
        video = rearrange(video, "b (h w) c -> b c h w", b=B, h=H // vae.patch_size, w=W // vae.patch_size)
        torch.save(video, save_path)
        id += 1
        new_meta_data.append({
            "file": action_save_path,
            "length": B,
        })
        pbar.update(1)
    pbar.close()
    with open(new_meta_data_path, "w") as f:
        json.dump(new_meta_data, f, indent=4)

def reload():
    metadata_path = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/metadata.json"
    original_dir = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/data"
    with open(metadata_path, "r") as f:
        meta_data = json.load(f)

    new_dir = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/20k"
    os.makedirs(new_dir, exist_ok=True)

    new_meta_data = []
    for idx, item in tqdm(enumerate(meta_data)):
        action_path = item["file"]
        assert action_path == f"{original_dir}/{idx:04d}.npz", f"Action path mismatch: {action_path}"
        video_path = action_path.replace("npz", "pt")
        actions = np.load(action_path)["actions"]
        video = torch.load(video_path, weights_only=True)
        actions = actions[1:]
        video = video[1:]
        new_action_path = os.path.join(new_dir, f"{idx:06d}.npz")
        new_video_path = os.path.join(new_dir, f"{idx:06d}.pt")
        np.savez_compressed(new_action_path, actions=actions)
        torch.save(video, new_video_path)
        new_meta_data.append({
            "file": new_action_path,
            "length": len(actions),
        })
        assert len(actions) == 1200

    new_metadata_path = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/20k.json"
    with open(new_metadata_path, "w") as f:
        json.dump(new_meta_data, f, indent=4)

def handle():
    original_metadata_path = "/home/tc0786/Project/train-oasis/data/mc_mem_data/metadata_old.json"
    with open(original_metadata_path, "r") as f:
        handled_file = json.load(f)["training"]
    root_dir = "/home/tc0786/Project/train-oasis/"
    handle_file_set = set()
    for item in handled_file:
        file_path = item["file"]
        file_path = os.path.join(root_dir, file_path)
        handle_file_set.add(file_path)
    print(f"Total handled files: {len(handle_file_set)}")

    data_dir = "/home/tc0786/Project/train-oasis/data/mc_mem_data"
    file_to_handle = []
    for i in range(24):
        subset_dir = f"{data_dir}/{i}"
        print(i, len(os.listdir(subset_dir)))
        for file_name in os.listdir(subset_dir):
            if file_name.endswith(".npz"):
                file_path = os.path.join(subset_dir, file_name)
                if file_path not in handle_file_set:
                    file_to_handle.append(file_path)

    print(f"Total files to handle: {len(file_to_handle)}")
    # assert len(file_to_handle) == (20000 - 5900), f"Detected {len(file_to_handle)} files, expected {20000 - 5900}"

    new_metadata_path = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/20k.json"
    with open(new_metadata_path, "r") as f:
        new_metadata = json.load(f)
    save_dir = "/home/tc0786/Project/train-oasis/data/mem_encoded_data/20k"

    assert torch.cuda.is_available()
    device = "cuda"
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    load_model(vae, "models/oasis500m/vit-l-20.safetensors")
    vae = vae.to(device).eval()
    scaling_factor = 0.07843137255

    idx = len(new_metadata)
    for file_path in tqdm(file_to_handle):
        actions = np.load(file_path)["actions"]
        video_path = file_path.replace("npz", "mp4")
        assert os.path.exists(video_path), f"Video file {video_path} does not exist."
        actions = actions[1:]
        actions_save_path = f"{save_dir}/{idx:06d}.npz"
        save_path = f"{save_dir}/{idx:06d}.pt"
        np.savez_compressed(actions_save_path, actions=actions)

        video, _, _ = read_video(video_path, pts_unit="sec")
        video = video.contiguous().numpy()
        video = torch.from_numpy(video).float() / 255.0
        video = video.to(torch.half).to(device)
        video = video.permute(0, 3, 1, 2).contiguous()
        B = video.shape[0]
        H, W = video.shape[-2:]
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                video = vae.encode(video * 2 - 1).mean * scaling_factor
        video = rearrange(video, "b (h w) c -> b c h w", b=B, h=H // vae.patch_size, w=W // vae.patch_size)
        torch.save(video[1:], save_path)
        new_metadata.append({
            "file": actions_save_path,
            "length": len(actions),
        })
        idx += 1
    with open(new_metadata_path, "w") as f:
        json.dump(new_metadata, f, indent=4)

if __name__ == "__main__":
    handle()