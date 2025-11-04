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
from torchvision.io import read_video
from fractions import Fraction

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
    lpips_batch_size = 64
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


@torch.no_grad()
def vanilla(args):
    from train_oasis.model.dit import DiT

    ddim_noise_steps = 50
    window_size = args.window_size
    metadata_file_path = args.metadata_file_path
    save_dir = args.save_dir
    metrics_save_path = os.path.join(save_dir, "metrics.json")
    dtype = torch.bfloat16
    vae_batch_size = 128
    batch_size = args.batch_size
    prompt_length = args.prompt_length
    total_length = args.total_length
    predict_v = True
    external_cond_dim = args.external_cond_dim

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    with open(metadata_file_path, 'r') as f:
        paths = json.load(f)["validation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"
    
    cfg_rel_path = "config/model/scannet_dit.yaml"
    cfg_path = os.path.join(dir_path, cfg_rel_path)
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
        gradient_checkpointing=config["gradient_checkpointing"],
        dtype=dtype,
    )
    ckpt_path = args.ckpt_path
    if os.path.isdir(ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    elif ckpt_path.endswith(".ckpt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"Model loaded from {ckpt_path} and moved to {device}.")

    vae = AutoencoderKL.from_pretrained(vae_ckpt)
    vae.eval()
    vae = vae.to(device).eval()

    # sampling params
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15
    fps = 30

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model.max_frames = window_size

    prompts = []
    gt_videos = []
    all_actions = []
    save_paths = []
    all_metrics = []
    for info in paths:
        action_path = info["file"]
        video_path = action_path.with_suffix(".mp4")
        start = Fraction(0, fps)
        end = Fraction((total_length - 1), fps)
        video, _, _ = read_video(str(video_path), start_pts=start, end_pts=end, pts_unit="sec")
        video = video.contiguous().float() / 255.0
        assert video.shape[0] == total_length, f"video.shape[0]={video.shape[0]} != {total_length}"
        gt_videos.append(video)
        video = video.permute(0, 3, 1, 2).contiguous() # (T, C, H, W)
        x = video[:prompt_length].unsqueeze(0)  # (1, T, C, H, W)
        assert x.shape[1] == prompt_length, f"x.shape[1]={x.shape[1]} != {prompt_length}"

        # load npy data
        actions = np.load(action_path, allow_pickle=True)
        actions = actions[: total_length]
        actions = torch.from_numpy(actions).float().unsqueeze(0)  # (1, T, D)
        assert actions.shape[0] == total_length, f"actions.shape[0]={actions.shape[0]} != {total_length}"

        prompts.append(x)
        all_actions.append(actions)
        file_name = os.path.basename(action_path).replace(".npy", ".mp4")
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    for start_idx in range(0, len(prompts), batch_size):
        # sampling inputs
        x = torch.cat(prompts[start_idx : start_idx + batch_size], dim=0)  # (B, T, C, H, W)
        actions = torch.cat(all_actions[start_idx : start_idx + batch_size], dim=0)  # (B, T, D)
        B = x.shape[0]
        H, W = x.shape[-2:]
        # assert B == batch_size, f"{B} != {batch_size}"
        # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        x = x.to(device)
        actions = actions.to(device)

        # vae encoding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        all_frames = []
        
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = x_clip * 2 - 1
                    x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=prompt_length)

        # sampling loop
        for i in tqdm(range(prompt_length, total_length), desc="sampling frames"):
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
                if not predict_v:
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
        with torch.no_grad():
            all_frames = []
            for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                x_clip = x[idx:idx + vae_batch_size]
                x_clip = vae.decode(x_clip / vae.config.scaling_factor).sample
                x_clip = (x_clip + 1) / 2
                all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_length)

        for idx in range(B):
            # save video
            video = x[idx]
            video = torch.clamp(video, 0, 1)
            gt_video = gt_videos[start_idx + idx].to(device)
            metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
            all_metrics.append({
                "file": paths[start_idx + idx]["file"],
                "prompt_frames": prompt_length,
                "metrics": metrics
            })
            video = video.cpu()
            video = (video * 255).byte()
            output_path = save_paths[start_idx + idx]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_video(output_path, video, fps=fps)

    # save metrics
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    with open(metrics_save_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    # get_metrics()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--metadata_file_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--external_cond_dim', type=int, required=True, help='Dimension of external condition')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for inference')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--prompt_length', type=int, default=8, help='Number of prompt frames')
    parser.add_argument('--total_length', type=int, default=20, help='Total number of frames to generate')
    args = parser.parse_args()
    vanilla(args)
    # vanilla("ws20", "outputs/rnn/df_maze/checkpoints/epoch=0-step=36000.ckpt")
    # rnn_check_in_context("LSTM_comb_sft_unfreeze_epoch6", "outputs/rnn/lstm_maze_pos_comb_sft_unfreeze/checkpoints/ckpt-epoch=5-step=4890.ckpt")