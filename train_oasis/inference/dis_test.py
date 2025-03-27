"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import numpy as np
import torch
import torch.nn.functional as F
from train_oasis.model.video_discriminator import VideoDiscriminator
from train_oasis.model.vae import VAE_models
import cv2
from train_oasis.utils import load_actions
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
import time

assert torch.cuda.is_available()
device = "cuda:0"

def load_video_to_numpy(video_path):
    # 打开视频文件
    start_time = time.time()
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    frames = []
    
    # 逐帧读取视频
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将 BGR 格式转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    # 释放视频捕获对象
    cap.release()
    
    # 将帧列表转换为 numpy 数组
    video_np = np.array(frames)

    # 打印视频加载时间
    end_time = time.time()
    print(f"Video loaded in {end_time - start_time:.2f} seconds, shape: {video_np.shape}")
    
    return video_np

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = VideoDiscriminator()
    print(f"loading Video discriminator from oasis-ckpt={os.path.abspath(args.dis_ckpt)}...")
    ckpt = torch.load(args.dis_ckpt)
    state_dict = {}
    for key, value in ckpt['state_dict'].items():
        if key.startswith("discriminator."):
            state_dict[key[14:]] = value
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
    if args.vae_ckpt.endswith(".pt"):
        vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
        vae.load_state_dict(vae_ckpt)
    elif args.vae_ckpt.endswith(".safetensors"):
        load_model(vae, args.vae_ckpt)
    vae = vae.to(device).eval()

    # load video
    print(f"loading video from video-path={os.path.abspath(args.video_path)}...")
    video = load_video_to_numpy(args.video_path)
    video = torch.from_numpy(video).float() / 255.0
    video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # (1, T, C, H, W)
    video = video.to(device)
    total_frames = video.shape[1]
    print(f"video shape: {video.shape}")
    # get input action stream
    actions = load_actions(args.actions_path, action_offset=None)[:, :total_frames]
    if actions.shape[1] < total_frames:
        copy_actions_list = [actions for _ in range(total_frames // actions.shape[1] + 1)]
        actions = torch.cat(copy_actions_list, dim=1)
        actions = actions[:, :total_frames]
    assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"
    actions = actions.to(device)
    print(f"actions shape: {actions.shape}")
    # sampling inputs
    B = video.shape[0]
    H, W = video.shape[-2:]

    # vae encoding
    vae_batch_size = 128
    scaling_factor = 0.07843137255
    video = rearrange(video, "b t c h w -> (b t) c h w")
    with torch.no_grad():
        all_frames = []
        for idx in tqdm(range(0, video.shape[0], vae_batch_size), desc="vae encoding frames"):
            x_clip = video[idx:idx + vae_batch_size]
            x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
            all_frames.append(x_clip)
        video = torch.cat(all_frames, dim=0)

    video = rearrange(video, "(b t) (h w) c -> b t c h w", b=1, t=total_frames, h=H // vae.patch_size, w=W // vae.patch_size)

    # sampling loop
    all_probs = []
    for start in tqdm(range(total_frames - model.max_frames + 1)):
        end = start + model.max_frames
        x_clip = video[:, start:end]  # (1, T, C, H, W)
        actions_clip = actions[:, start:end]  # (1, T, A)

        prob = F.sigmoid(model(x_clip, actions_clip))  # (1, )
        all_probs.append(prob.item())
    print(all_probs)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--video-path",
        type=str,
        help="Path to Video.",
        default="outputs/video/easy_gan_20_3000.mp4",
    )
    parse.add_argument(
        "--dis-ckpt",
        type=str,
        help="Path to Discriminator checkpoint.",
        default="outputs/2025-03-21/08-27-10/checkpoints/epoch=1-step=40000.ckpt",
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="models/vit/vit-l-20.safetensors",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="data/minecraft_easy/5/000038.npz",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=None,
    )

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
