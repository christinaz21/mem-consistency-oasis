"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
import argparse
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from train_oasis.model.vae import VAE_models
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import os
import json
from safetensors.torch import load_model
from torchvision.io import read_video

assert torch.cuda.is_available()
device = "cuda"

def gt(save_dir, metadata_path, new_meta_data_path):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save encoded data")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to input metadata JSON")
    parser.add_argument("--new_meta_data_path", type=str, required=True, help="Path to output new metadata JSON")
    args = parser.parse_args()
    gt(args.save_dir, args.metadata_path, args.new_meta_data_path)