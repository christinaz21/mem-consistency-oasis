import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)
import yaml
import torch
from types import SimpleNamespace
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from torchvision.io import read_video, write_video
from train_oasis.utils import sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import os
import json
import numpy as np
from diffusers.models import AutoencoderKL
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
    universal_image_quality_index
)
import argparse
from train_oasis.utils import load_prompt, load_actions
import os, sys
sys.path.append("/n/fs/videogen/train-oasis/train_oasis/inference")
from worldscore import DroidReprojectionScorer

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import subprocess
import tempfile
import shutil
import re
# import cv2

@torch.no_grad()
def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :return: a tuple of metrics
    """
    frame, batch, channel, height, width = observation_hat.shape
    observation_gt = observation_gt * 2 - 1
    observation_hat = observation_hat * 2 - 1
    observation_gt = observation_gt.contiguous()
    observation_hat = observation_hat.contiguous()
    output_dict = {}
    observation_gt = observation_gt.type_as(observation_hat)  # some metrics don't fully support fp16

    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    @torch.no_grad()
    def mse_frame(pred, gt):
        mse = (pred - gt) ** 2
        return mse.mean(dim=[1, 2, 3])
    @torch.no_grad()
    def uiqi_frame(pred, gt):
        uiqi = universal_image_quality_index(pred, gt, reduction="none")
        return uiqi.mean(dim=[1, 2, 3])
    
    output_dict["mse"] = mse_frame(observation_hat, observation_gt).cpu().tolist()
    output_dict["psnr"] = peak_signal_noise_ratio(observation_hat, observation_gt, data_range=2.0, reduction="none", dim=[1, 2, 3]).cpu().tolist()
    ssim_batch_size = 200
    output_dict["ssim"] = []
    output_dict["uiqi"] = []
    for i in range(0, observation_hat.shape[0], ssim_batch_size):
        observation_hat_batch = observation_hat[i : i + ssim_batch_size]
        observation_gt_batch = observation_gt[i : i + ssim_batch_size]
        ssim = structural_similarity_index_measure(observation_hat_batch, observation_gt_batch, data_range=2.0, reduction="none").cpu().tolist()
        output_dict["ssim"].extend(ssim)
        uiqi = uiqi_frame(observation_hat_batch, observation_gt_batch).cpu().tolist()
        output_dict["uiqi"].extend(uiqi)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    output_dict["lpips"] = []
    lpips_batch_size = 16
    device = observation_hat.device
    global _lpips_model
    if '_lpips_model' not in globals():
        _lpips_model = lpips.LPIPS(net="vgg").to(device)
    else:
        _lpips_model = _lpips_model.to(device)
    lpips_model = _lpips_model
    # lpips_model = lpips.LPIPS(net="vgg").to(device)
    for i in range(0, observation_hat.shape[0], lpips_batch_size):
        observation_hat_batch = observation_hat[i : i + lpips_batch_size]
        observation_gt_batch = observation_gt[i : i + lpips_batch_size]
        lpips_metrics = lpips_model(observation_hat_batch, observation_gt_batch).flatten().cpu().tolist()
        output_dict["lpips"].extend(lpips_metrics)
    return output_dict

def get_data(args):
    with open(args.metadata_file_path, 'r') as f:
        paths = json.load(f)
    
    if args.dataset in ['maze15', 'maze9']:
        action_types = ["action", "agent_pos", "agent_dir"]
        prompts = []
        gt_videos = []
        all_actions = []
        save_paths = []
        for info in paths:
            prompt_path = info["file"]
            # length = info["length"]
            data = np.load(prompt_path, allow_pickle=True)
            length = data["image"].shape[0]
            n_prompt_frames = max(10, int(length * 0.6))

            x = data["image"][:n_prompt_frames] # (T, H, W, C)
            gt_video = data["image"]  # (T, H, W, C)
            gt_video = torch.from_numpy(gt_video).float() / 255.  # (T, H, W, C)
            gt_videos.append(gt_video)
            x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
            actions = torch.cat([torch.from_numpy(data[atype]) for atype in action_types], dim=-1)
            actions = actions.unsqueeze(0).float() # (1, T, 10)

            prompts.append(x)
            all_actions.append(actions)
            file_name = os.path.basename(prompt_path).replace(".npz", ".mp4")
            save_path = os.path.join(args.save_dir, "videos", file_name)
            save_paths.append(save_path)
        return prompts, gt_videos, all_actions, save_paths
    elif args.dataset == 'maze15_batch':
        action_types = ["action", "agent_pos", "agent_dir"]
        prompts = []
        gt_videos = []
        all_actions = []
        save_paths = []
        for info in paths:
            prompt_path = info["file"]
            # length = info["length"]
            data = np.load(prompt_path, allow_pickle=True)
            length = data["image"].shape[0]
            n_prompt_frames = args.prompt_frames

            x = data["image"][:n_prompt_frames] # (T, H, W, C)
            gt_video = data["image"][:args.total_frames]  # (T, H, W, C)
            gt_video = torch.from_numpy(gt_video).float() / 255.  # (T, H, W, C)
            gt_videos.append(gt_video)
            x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
            actions = torch.cat([torch.from_numpy(data[atype]) for atype in action_types], dim=-1)
            actions = actions[:args.total_frames]
            actions = actions.unsqueeze(0).float() # (1, T, 10)

            prompts.append(x)
            all_actions.append(actions)
            file_name = os.path.basename(prompt_path).replace(".npz", ".mp4")
            save_path = os.path.join(args.save_dir, "videos", file_name)
            save_paths.append(save_path)
        return prompts, gt_videos, all_actions, save_paths
    elif args.dataset == 'minecraft':
        memory_meta_data = paths["memory"]
        prompts = []
        all_actions = []
        save_paths = []
        gt_videos = []
        for info in memory_meta_data:
            prompt_path = info["video_path"]
            video = read_video(prompt_path, pts_unit='sec')[0]  # (T, H, W, C)
            gt_video = video[:args.total_frames].float() / 255.
            gt_videos.append(gt_video)

            x = video[:args.prompt_frames].permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
            # get input action stream
            actions_path = info["action_path"]
            actions = load_actions(actions_path, action_offset=None)[:, :args.total_frames, :4]
            assert actions.shape[1] == args.total_frames, f"{actions.shape[1]} != {args.total_frames}"

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(args.save_dir, "videos", info["save_relative_path"])
            save_paths.append(save_path)

        return prompts, gt_videos, all_actions, save_paths
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

@torch.no_grad()
def rnn_chunk_inference(args):
    from train_oasis.model.rnn_chunk_dit import DiT

    ddim_noise_steps = 50
    window_size = 20
    dtype = torch.float32
    vae_batch_size = 128
    predict_v = True
    external_cond_dim = 10 if "maze" in args.dataset else 4

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(args.save_dir, exist_ok=True)
    video_save_dir = os.path.join(args.save_dir, "videos")
    os.makedirs(video_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.vae_name == "oasis":
        from train_oasis.model.vae import AutoencoderKL
        from safetensors.torch import load_model
        vae = AutoencoderKL(
            latent_dim=16,
            patch_size=20,
            enc_dim=1024,
            enc_depth=6,
            enc_heads=16,
            dec_dim=1024,
            dec_depth=12,
            dec_heads=16,
            input_height=360,
            input_width=640,
        )
        assert args.vae_ckpt, "VAE checkpoint is required for oasis VAE."
        load_model(vae, args.vae_ckpt)
        vae.eval()
    elif args.vae_name == "sd_vae":
        assert args.vae_ckpt
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_ckpt)
        vae.eval()
    else:
        raise ValueError(f"Unknown VAE name: {args.vae_name}")
    vae = vae.to(device).eval()

    cfg_rel_dir = {
        "LSTM": "config/model/rnn_lstm_maze.yaml",
        "Mamba": "config/model/rnn_mamba_maze.yaml",
        "TTT": "config/model/rnn_ttt_maze.yaml",
    }

    cfg_path = os.path.join(dir_path, cfg_rel_dir[args.rnn_type])
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    rnn_config = config["rnn_config"]
    if args.combine_actions:
        rnn_config["combine_action_dim"] = 128
    if args.onemem:
        rnn_config["one_mem"] = True
    if args.mask_last_hidden_state:
        rnn_config["mask_last_hidden_state"] = True
    # 将 rnn_config（dict）转为支持属性访问的对象
    rnn_config = SimpleNamespace(**rnn_config)

    if args.dataset == "minecraft":
        cfg_path = os.path.join(dir_path, "config/model/dit.yaml")
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
    
    model = DiT(
        input_h=config["input_h"],
        input_w=config["input_w"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        external_cond_dim=external_cond_dim,
        max_frames=window_size,
        rnn_config=rnn_config,
        gradient_checkpointing=config["gradient_checkpointing"],
        dtype=dtype,
    )
    if os.path.isdir(args.ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("Only support deepspeed zero checkpoints for RNN chunk inference.")
    model = model.to(device).eval()
    print(f"Model {args.rnn_type} loaded from {args.ckpt_path} and moved to {device}.")

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

    all_metrics = []
    prompts, gt_videos, all_actions, save_paths = get_data(args)

    for start_idx in range(0, len(prompts), args.batch_size):
        # sampling inputs
        # x = prompts[start_idx]
        x = torch.cat(prompts[start_idx : start_idx + args.batch_size], dim=0)
        # actions = all_actions[start_idx]
        actions = torch.cat(all_actions[start_idx : start_idx + args.batch_size], dim=0)
        B = x.shape[0]
        H, W = x.shape[-2:]
        # assert B == batch_size, f"{B} != {batch_size}"
        # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        x = x.to(device)
        actions = actions.to(device)

        # vae encoding
        n_prompt_frames = x.shape[1]
        total_frames = actions.shape[1]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        all_frames = []
        
        if args.vae_name == "sd_vae":
            with torch.no_grad():
                with autocast("cuda", dtype=dtype):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = x_clip * 2 - 1
                        x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        elif args.vae_name == "oasis":
            scaling_factor = 0.07843137255
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
                x = rearrange(x, "b (h w) c -> b c h w", h=H // vae.patch_size, w=W // vae.patch_size)
        else:
            raise ValueError(f"Unknown VAE name: {args.vae_name}")
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

        # get hidden states for prompt frames
        hidden_states = None
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
                for start_frame in range(0, n_prompt_frames, model.max_frames):
                    if start_frame + model.max_frames > n_prompt_frames:
                        break
                    end_frame = start_frame + model.max_frames
                    x_chunk = x[:, start_frame:end_frame]
                    t_chunk = torch.full((B, end_frame - start_frame), stabilization_level - 1, dtype=torch.long, device=device)
                    actions_chunk = actions[:, start_frame:end_frame]
                    _, hidden_states = model.inference(x_chunk, t_chunk, actions_chunk, hidden_states=hidden_states, start_ids=start_frame)

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames), desc="sampling frames"):
            start_frame = (i // model.max_frames) * model.max_frames
            end_frame = min(start_frame + model.max_frames, i + 1)
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)

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
                    with autocast("cuda", dtype=dtype):
                        # v = model(x_curr, t, actions[:, start_frame : i + 1])
                        if noise_idx == 1 and (i + 1) % model.max_frames == 0:
                            v, hidden_states = model.inference(x_curr, t, actions[:, start_frame : end_frame], hidden_states=hidden_states, start_ids=start_frame)
                        else:
                            v, _ = model.inference(x_curr, t, actions[:, start_frame : end_frame], hidden_states=hidden_states, start_ids=start_frame)
                if predict_v:
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                else:
                    x_start = v
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
        if args.vae_name == "sd_vae":
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
                    x_clip = (x_clip + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        elif args.vae_name == "oasis":
            x = rearrange(x, "b c h w -> b (h w) c")
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = (vae.decode(x_clip / scaling_factor) + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        else:
            raise ValueError(f"Unknown VAE name: {args.vae_name}")
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_frames)

        for idx in range(B):
            # save video
            video = x[idx]
            video = torch.clamp(video, 0, 1)
            gt_video = gt_videos[start_idx + idx].to(device)
            metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
            all_metrics.append({
                "prompt_frames": n_prompt_frames,
                "metrics": metrics
            })
            video = video.cpu()
            video = (video * 255).byte()
            output_path = save_paths[start_idx + idx]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, video, fps=fps)

    # save metrics
    metrics_save_path = os.path.join(args.save_dir, "metrics.json")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

@torch.no_grad()
def rnn_frame_inference(args):
    from train_oasis.model.rnn_dit import DiT

    ddim_noise_steps = 50
    window_size = 20
    dtype = torch.float32
    vae_batch_size = 128
    predict_v = True
    external_cond_dim = 10 if "maze" in args.dataset else 4

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(args.save_dir, exist_ok=True)
    video_save_dir = os.path.join(args.save_dir, "videos")
    os.makedirs(video_save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the droid-slam scorer
    scorer = DroidReprojectionScorer(
        weights_path="/u/cz5047/videogen/data/models/droid_models/droid.pth",          # add CLI arg
        stride=getattr(args, "droid_stride", 2),
        max_frames=getattr(args, "droid_max_frames", 200),
        resize_long_side=getattr(args, "droid_resize", 256),
        quiet=True,
    )

    if args.vae_name == "oasis":
        from train_oasis.model.vae import AutoencoderKL
        from safetensors.torch import load_model
        vae = AutoencoderKL(
            latent_dim=16,
            patch_size=20,
            enc_dim=1024,
            enc_depth=6,
            enc_heads=16,
            dec_dim=1024,
            dec_depth=12,
            dec_heads=16,
            input_height=360,
            input_width=640,
        )
        assert args.vae_ckpt, "VAE checkpoint is required for oasis VAE."
        load_model(vae, args.vae_ckpt)
        vae.eval()
    elif args.vae_name == "sd_vae":
        assert args.vae_ckpt
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.vae_ckpt)
        vae.eval()
    else:
        raise ValueError(f"Unknown VAE name: {args.vae_name}")
    vae = vae.to(device).eval()

    cfg_rel_dir = {
        "LSTM": "config/model/rnn_lstm_maze.yaml",
        "Mamba": "config/model/rnn_mamba_maze.yaml",
        "TTT": "config/model/rnn_ttt_maze.yaml",
    }

    cfg_path = os.path.join(dir_path, cfg_rel_dir[args.rnn_type])
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    rnn_config = config["rnn_config"]
    if args.combine_actions:
        rnn_config["combine_action_dim"] = 128
    if args.onemem:
        rnn_config["one_mem"] = True
    if args.mask_last_hidden_state:
        rnn_config["mask_last_hidden_state"] = True
    # 将 rnn_config（dict）转为支持属性访问的对象
    rnn_config = SimpleNamespace(**rnn_config)

    if args.dataset == "minecraft":
        cfg_path = os.path.join(dir_path, "config/model/dit.yaml")
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
    
    model = DiT(
        input_h=config["input_h"],
        input_w=config["input_w"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        hidden_size=config["hidden_size"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        external_cond_dim=external_cond_dim,
        max_frames=window_size,
        rnn_config=rnn_config,
        gradient_checkpointing=config["gradient_checkpointing"],
        dtype=dtype,
    )
    if os.path.isdir(args.ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError("Only support deepspeed zero checkpoints for RNN chunk inference.")
    model = model.to(device).eval()
    print(f"Model {args.rnn_type} loaded from {args.ckpt_path} and moved to {device}.")

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

    all_metrics = []
    prompts, gt_videos, all_actions, save_paths = get_data(args)

    ### GENERATE ONE CANDIDATE VIDEO FOR EACH PROMPT
    def _generate_one_candidate(x_prompt, actions, seed: int):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        B = x_prompt.shape[0]
        H, W = x_prompt.shape[-2:]
        n_prompt_frames = x_prompt.shape[1]
        total_frames = actions.shape[1]

        x = x_prompt.to(device)
        actions = actions.to(device)

        x = rearrange(x, "b t c h w -> (b t) c h w")
        all_frames = []
        
        # for start_idx in range(0, len(prompts), args.batch_size):
        #     # sampling inputs
        #     # x = prompts[start_idx]
        #     x = torch.cat(prompts[start_idx : start_idx + args.batch_size], dim=0)
        #     # actions = all_actions[start_idx]
        #     actions = torch.cat(all_actions[start_idx : start_idx + args.batch_size], dim=0)
        #     B = x.shape[0]
        #     H, W = x.shape[-2:]
        #     # assert B == batch_size, f"{B} != {batch_size}"
        #     # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        #     # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        #     x = x.to(device)
        #     actions = actions.to(device)

        #     # vae encoding
        #     n_prompt_frames = x.shape[1]
        #     total_frames = actions.shape[1]
        #     x = rearrange(x, "b t c h w -> (b t) c h w")
        #     all_frames = []
        
        if args.vae_name == "sd_vae":
            with torch.no_grad():
                with autocast("cuda", dtype=dtype):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = x_clip * 2 - 1
                        x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        elif args.vae_name == "oasis":
            scaling_factor = 0.07843137255
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
                x = rearrange(x, "b (h w) c -> b c h w", h=H // vae.patch_size, w=W // vae.patch_size)
        else:
            raise ValueError(f"Unknown VAE name: {args.vae_name}")
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

        # get hidden states for prompt frames
        first_start_frame = max(0, n_prompt_frames + 1 - model.max_frames)
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
                all_hidden_states = model.window_size_1_forward(x[:, :first_start_frame], torch.full((B, first_start_frame), stabilization_level - 1, dtype=torch.long, device=device), actions[:, :first_start_frame], mini_batch_size=256, target_hidden_states=[first_start_frame])[0]

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames), desc="sampling frames"):
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
                    with autocast("cuda", dtype=dtype):
                        # v = model(x_curr, t, actions[:, start_frame : i + 1])
                        if noise_idx == 1:
                            v, all_hidden_states = model.inference(x_curr, t, actions[:, start_frame : i + 1], hidden_states=all_hidden_states, get_new_hidden_states=True)
                        else:
                            v, _ = model.inference(x_curr, t, actions[:, start_frame : i + 1], hidden_states=all_hidden_states, get_new_hidden_states=False)
                if predict_v:
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                else:
                    x_start = v
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
        if args.vae_name == "sd_vae":
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
                    x_clip = (x_clip + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        elif args.vae_name == "oasis":
            x = rearrange(x, "b c h w -> b (h w) c")
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = (vae.decode(x_clip / scaling_factor) + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
        else:
            raise ValueError(f"Unknown VAE name: {args.vae_name}")
        x_dec = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_frames)
        return x_dec


        # for idx in range(B):
        #     # save video
        #     video = x[idx]
        #     video = torch.clamp(video, 0, 1)
        #     gt_video = gt_videos[start_idx + idx].to(device)
        #     metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
        #     all_metrics.append({
        #         "prompt_frames": n_prompt_frames,
        #         "metrics": metrics
        #     })
        #     video = video.cpu()
        #     video = (video * 255).byte()
        #     output_path = save_paths[start_idx + idx]
        #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        #     write_video(output_path, video, fps=fps)

    # -------------------------
    # Main loop with Best-of-N
    # -------------------------
    num_particles = 4
    base_seed = 0

    for start_idx in range(0, len(prompts), args.batch_size):
        x_prompt = torch.cat(prompts[start_idx : start_idx + args.batch_size], dim=0)   # [B,Tp,C,H,W]
        actions = torch.cat(all_actions[start_idx : start_idx + args.batch_size], dim=0)  # [B,T,A]
        B = x_prompt.shape[0]
        n_prompt_frames = x_prompt.shape[1]

        # Generate N candidates
        candidate_videos = []
        candidate_scores = np.zeros((num_particles, B), dtype=np.float32)

        for p in range(num_particles):
            seed = base_seed + p + 1000 * (start_idx // args.batch_size)
            x_dec = _generate_one_candidate(x_prompt, actions, seed=seed)  # [B,T,H,W,C]
            candidate_videos.append(x_dec)

            # score each sample in batch
            for b in range(B):
                candidate_scores[p, b] = scorer.reward_from_video(x_dec[b])
                print(f"Droid reward: {candidate_scores[p, b]}")

        # Pick best per sample
        best_p = candidate_scores.argmax(axis=0)  # [B]
        best_scores = candidate_scores[best_p, np.arange(B)]

        # Save + metrics using chosen candidate
        for b in range(B):
            x_best = candidate_videos[int(best_p[b])][b]
            x_best = torch.clamp(x_best, 0, 1)

            gt_video = gt_videos[start_idx + b].to(device)
            metrics = get_validation_metrics_for_videos(
                x_best.permute(0, 3, 1, 2).unsqueeze(1),
                gt_video.permute(0, 3, 1, 2).unsqueeze(1),
            )

            all_metrics.append({
                "prompt_frames": int(n_prompt_frames),
                "metrics": metrics,
                "droid_reward": float(best_scores[b]),
                "selected_particle": int(best_p[b]),
            })

            # write video
            out_path = save_paths[start_idx + b]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            v8 = (x_best.detach().cpu() * 255).byte()
            write_video(out_path, v8, fps=fps)

    # save metrics
    metrics_save_path = os.path.join(args.save_dir, "metrics.json")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)





def _write_video_frames_png(video_thwc: torch.Tensor, out_dir: str, max_frames: int = 200, resize_long_side: int = 256):
    """
    video_thwc: (T,H,W,C) float in [0,1], torch tensor (cpu or cuda)
    Writes PNG frames to out_dir. Returns out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    v = torch.clamp(video_thwc, 0, 1)
    if v.is_cuda:
        v = v.detach().cpu()
    v = (v * 255).byte().numpy()  # uint8 RGB
    T = min(v.shape[0], max_frames)

    for t in range(T):
        rgb = v[t]  # (H,W,3) RGB uint8
        if resize_long_side is not None and resize_long_side > 0:
            h, w = rgb.shape[:2]
            long_side = max(h, w)
            if long_side > resize_long_side:
                scale = resize_long_side / float(long_side)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        bgr = rgb[..., ::-1]  # cv2 wants BGR
        fp = os.path.join(out_dir, f"{t:06d}.png")
        ok = cv2.imwrite(fp, bgr)
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {fp}")

    return out_dir


def score_with_droid_eval(imagedir: str, calib_path: str, weights: str, stride: int = 1, buffer: int = 512, upsample: bool = True):
    """
    Runs scoring in the separate 'droid_eval' conda env and returns mean reprojection error (float).
    Assumes score_reproj.py prints something like: 'mean_reprojection_error: <float>'
    """
    upsample_flag = "--upsample" if upsample else ""

    # Use conda.sh (more reliable than relying on ~/.bashrc)
    bash_cmd = (
        "set -euo pipefail; "
        "source \"$(conda info --base)/etc/profile.d/conda.sh\"; "
        "conda activate droid; "
        "python /n/fs/videogen/train-oasis/train_oasis/inference/score_reproj.py "
        f"--imagedir \"{imagedir}\" "
        f"--calib \"{calib_path}\" "
        f"--weights \"{weights}\" "
        f"--stride {int(stride)} "
        f"--buffer {int(buffer)} "
        f"{upsample_flag}"
    )

    out = subprocess.check_output(["/bin/bash", "-lc", bash_cmd], text=True)

    # Parse the float robustly
    m = re.search(r"mean_reprojection_error:\s*([0-9]*\.?[0-9]+([eE][-+]?\d+)?)", out)
    if m:
        return float(m.group(1))

    # fallback: last token
    toks = out.strip().split()
    if len(toks) == 0:
        raise RuntimeError(f"Empty output from scorer. Raw output:\n{out}")
    return float(toks[-1])


def score_video_tensor_with_droid_eval(
    video_thwc: torch.Tensor,
    calib_path: str,
    weights: str,
    stride: int = 2,
    max_frames: int = 200,
    resize_long_side: int = 256,
    buffer: int = 512,
    upsample: bool = True,
) -> float:
    """
    Writes frames to a temp dir, calls droid_eval scorer, returns reward = -mean_error.
    """
    tmpdir = tempfile.mkdtemp(prefix="droid_eval_frames_")
    try:
        _write_video_frames_png(video_thwc, tmpdir, max_frames=max_frames, resize_long_side=resize_long_side)
        mean_err = score_with_droid_eval(
            imagedir=tmpdir,
            calib_path=calib_path,
            weights=weights,
            stride=stride,
            buffer=buffer,
            upsample=upsample,
        )
        if not np.isfinite(mean_err):
            return -1e9
        return -float(mean_err)  # reward
    except Exception as e:
        # If DROID fails on a sample, give a terrible reward so it won't be selected
        return -1e9
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)




if __name__ == "__main__":
    # get_metrics()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save outputs')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--metadata_file_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--inference_type', type=str, required=True, help='Inference type', choices=['chunkwise', 'framewise'])
    parser.add_argument('--vae_name', type=str, default='sd_vae', choices=['sd_vae', 'oasis'], help='VAE model name')
    parser.add_argument('--vae_ckpt', type=str, default='stabilityai/sd-vae-ft-mse', help='Path to VAE checkpoint if using oasis VAE')
    parser.add_argument('--combine_actions', action='store_true', help='Whether to combine actions')
    parser.add_argument('--dataset', type=str, default='maze15', help='Dataset name', choices=['maze15', 'maze9', "minecraft", "maze15_batch"])
    parser.add_argument('--onemem', action='store_true', help='Use one memory for RNN')
    parser.add_argument('--batch_size', type=int, default=1, help='Inference batch size')
    parser.add_argument('--mask_last_hidden_state', action='store_true', help='Whether to mask h')
    parser.add_argument('--rnn_type', type=str, default='LSTM', help='RNN type', choices=['LSTM', 'Mamba', 'TTT'])
    parser.add_argument('--total_frames', type=int, default=20, help='Total frames to generate during inference, will deprecate when using maze15 dataset')
    parser.add_argument('--prompt_frames', type=int, default=10, help='Number of clean frames during inference, will deprecate when using maze15 dataset')
    parser.add_argument("--droid_calib", type=str, default="/u/cz5047/videogen/DROID-SLAM/calib/test.txt")
    parser.add_argument("--droid_weights", type=str, default="/u/cz5047/videogen/data/models/droid_models/droid.pth")
    parser.add_argument("--droid_stride", type=int, default=2)
    parser.add_argument("--droid_max_frames", type=int, default=200)
    parser.add_argument("--droid_resize", type=int, default=256)
    parser.add_argument("--droid_buffer", type=int, default=512)
    parser.add_argument("--droid_upsample", action="store_true")
    args = parser.parse_args()
    if args.inference_type == 'chunkwise':
        rnn_chunk_inference(args)
    elif args.inference_type == 'framewise':
        rnn_frame_inference(args)
    else:
        raise ValueError(f"Unknown inference type: {args.inference_type}")