import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from torchvision.io import read_video, write_video
from train_oasis.utils import load_prompt, load_actions, sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import os
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
import json
import numpy as np
from diffusers.models import AutoencoderKL

@torch.no_grad()
def vanilla():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vanilla_20"
    ddim_noise_steps = 20
    n_prompt_frames = 100
    total_frames = 1001
    batch_size = 32
    limited_validation_number = 64
    file_path = "data/maze/metadata.json"
    with open(file_path, "r") as f:
        paths = json.load(f)["validation"][:limited_validation_number]

    save_dir = f"outputs/df/maze_eval/{model_name}/validation_videos"
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "vanilla_20":
        oasis_ckpt = "outputs/df/maze_vae/checkpoints/epoch=0-step=33000.ckpt"
        window_size = 20
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.dit import DiT
    if model_name == "vanilla_20":
        model = DiT(
            input_h=8,
            input_w=8,
            in_channels=4,
            patch_size=2,
            hidden_size=1024,
            depth=12,
            num_heads=16,
            external_cond_dim=6,
            max_frames=20
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    # print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    if os.path.isdir(oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(oasis_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = AutoencoderKL.from_pretrained(vae_ckpt)
    vae.eval()
    vae = vae.to(device).eval()

    # sampling params
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15
    fps = 20

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model.max_frames = window_size

    prompts = []
    all_actions = []
    save_paths = []
    for info in paths:
        prompt_path = info["file"]
        data = np.load(prompt_path, allow_pickle=True)

        x = data["image"][:n_prompt_frames] # (T, H, W, C)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
        actions = torch.from_numpy(data["action"][:total_frames]).unsqueeze(0).float() # (1, T, 6)

        prompts.append(x)
        all_actions.append(actions)
        file_name = os.path.basename(prompt_path).replace(".npz", ".mp4")
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    prompts = torch.cat(prompts, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    assert prompts.shape[0] == all_actions.shape[0], f"{prompts.shape[0]} != {actions.shape[0]}"
    
    for start_idx in range(0, prompts.shape[0], batch_size):
        # sampling inputs
        x = prompts[start_idx : start_idx + batch_size]
        actions = all_actions[start_idx : start_idx + batch_size]
        B = x.shape[0]
        H, W = x.shape[-2:]
        # assert B == batch_size, f"{B} != {batch_size}"
        # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        x = x.to(device)
        actions = actions.to(device)

        # vae encoding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        vae_batch_size = 128
        all_frames = []
        
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = x_clip * 2 - 1
                    x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames)):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)

            for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                # set up noise values
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
                t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                # sliding window
                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])
                if model_name == "pred_x":
                    x_start = v
                else:
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                # get frame prediction
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        # vae decoding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        vae_batch_size = 128
        with torch.no_grad():
            all_frames = []
            for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                x_clip = x[idx:idx + vae_batch_size]
                x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
                x_clip = (x_clip + 1) / 2
                all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_frames)

        for idx in range(B):
            # save video
            video = x[idx]
            video = video.cpu()
            video = torch.clamp(video, 0, 1)
            video = (video * 255).byte()
            output_path = save_paths[start_idx + idx]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, video, fps=fps)

@torch.no_grad()
def memory_token_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "memory_token_1"
    ddim_noise_steps = 20
    n_prompt_frames = 100
    total_frames = 1001
    batch_size = 1
    limited_validation_number = 1
    file_path = "data/maze/metadata.json"
    with open(file_path, "r") as f:
        paths = json.load(f)["validation"][:limited_validation_number]

    save_dir = f"outputs/df/maze_eval/{model_name}/validation_videos"
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "memory_token_1":
        oasis_ckpt = "outputs/df/maze_100/checkpoints/epoch=0-step=30000.ckpt"
        memory_token_ckpt = "outputs/df/memory_token/checkpoints/epoch=2999-step=3000.ckpt"
        window_size = 100
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.memory_token_dit import DiT
    if model_name == "memory_token_1":
        model = DiT(
            input_h=8,
            input_w=8,
            in_channels=4,
            patch_size=2,
            hidden_size=1024,
            depth=12,
            num_heads=16,
            external_cond_dim=6,
            max_frames=100
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    # print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    if os.path.isdir(oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(oasis_ckpt, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    if os.path.isdir(memory_token_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(memory_token_ckpt)
        memory_token = ckpt["memory_token"]
    else:
        ckpt = torch.load(memory_token_ckpt, map_location="cpu")
        memory_token = ckpt["memory_token"]
    memory_token = memory_token.to(device)
    print("Loaded memory token:", memory_token.shape)

    # load VAE checkpoint
    vae = AutoencoderKL.from_pretrained(vae_ckpt)
    vae.eval()
    vae = vae.to(device).eval()

    # sampling params
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15
    fps = 20

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model.max_frames = window_size

    prompts = []
    all_actions = []
    save_paths = []
    for info in paths:
        prompt_path = info["file"]
        data = np.load(prompt_path, allow_pickle=True)

        x = data["image"][:n_prompt_frames] # (T, H, W, C)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
        actions = torch.from_numpy(data["action"][:total_frames]).unsqueeze(0).float() # (1, T, 6)

        prompts.append(x)
        all_actions.append(actions)
        file_name = os.path.basename(prompt_path).replace(".npz", ".mp4")
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    prompts = torch.cat(prompts, dim=0)
    all_actions = torch.cat(all_actions, dim=0)
    assert prompts.shape[0] == all_actions.shape[0], f"{prompts.shape[0]} != {actions.shape[0]}"
    
    for start_idx in range(0, prompts.shape[0], batch_size):
        # sampling inputs
        x = prompts[start_idx : start_idx + batch_size]
        actions = all_actions[start_idx : start_idx + batch_size]
        B = x.shape[0]
        H, W = x.shape[-2:]
        # assert B == batch_size, f"{B} != {batch_size}"
        # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        x = x.to(device)
        actions = actions.to(device)

        # vae encoding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        vae_batch_size = 128
        all_frames = []
        
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = x_clip * 2 - 1
                    x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames)):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)

            for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                # set up noise values
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
                t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                # sliding window
                x_curr = x.clone()
                x_curr = x_curr[:, start_frame:]
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, memory_token, actions[:, start_frame : i + 1])
                if model_name == "pred_x":
                    x_start = v
                else:
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                # get frame prediction
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                if noise_idx == 1:
                    alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, -1:] = x_pred[:, -1:]

        # vae decoding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        vae_batch_size = 128
        with torch.no_grad():
            all_frames = []
            for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                x_clip = x[idx:idx + vae_batch_size]
                x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
                x_clip = (x_clip + 1) / 2
                all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_frames)

        for idx in range(B):
            # save video
            video = x[idx]
            video = video.cpu()
            video = torch.clamp(video, 0, 1)
            video = (video * 255).byte()
            output_path = save_paths[start_idx + idx]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, video, fps=fps)

if __name__ == "__main__":
    memory_token_inference()