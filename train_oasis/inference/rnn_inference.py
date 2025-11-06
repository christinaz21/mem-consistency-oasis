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
def rnn_chunk_inference(model_name, ckpt_path):
    from train_oasis.model.rnn_chunk_dit import DiT

    ddim_noise_steps = 50
    window_size = 20
    metadata_file_path = "data/maze/metadata_self.json"
    combine_actions = "comb" in model_name
    save_dir = os.path.join("outputs/rnn/eval_outputs", f"rnn_chunk_{model_name}")
    metrics_save_path = os.path.join("outputs/rnn/eval_outputs/metrics", f"rnn_chunk_{model_name}.json")
    dtype = torch.bfloat16
    vae_batch_size = 128
    predict_v = True
    action_types = ["action", "agent_pos", "agent_dir"]
    external_cond_dim = 10

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    with open(metadata_file_path, 'r') as f:
        paths = json.load(f)[:40]
    paths = sorted(paths, key=lambda x: x["length"], reverse=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"

    if "LSTM" in model_name:
        cfg_rel_path = "config/model/rnn_lstm_maze.yaml"
    elif "Mamba" in model_name:
        cfg_rel_path = "config/model/rnn_mamba_maze.yaml"
    elif "TTT" in model_name:
        cfg_rel_path = "config/model/rnn_ttt_maze.yaml"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg_path = os.path.join(dir_path, cfg_rel_path)
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    rnn_config = config["rnn_config"]
    if combine_actions:
        rnn_config["combine_action_dim"] = 128
    # 将 rnn_config（dict）转为支持属性访问的对象
    rnn_config = SimpleNamespace(**rnn_config)
    
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
    if os.path.isdir(ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).to(dtype).eval()
    print(f"Model {model_name} loaded from {ckpt_path} and moved to {device}.")

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
    gt_videos = []
    all_actions = []
    save_paths = []
    all_metrics = []
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
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    for start_idx in range(len(prompts)):
        # sampling inputs
        x = prompts[start_idx]
        actions = all_actions[start_idx]
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
        
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = x_clip * 2 - 1
                    x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
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
            video = torch.clamp(video, 0, 1)
            gt_video = gt_videos[start_idx + idx].to(device)
            metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
            all_metrics.append({
                "file": paths[start_idx + idx]["file"],
                "prompt_frames": n_prompt_frames,
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

@torch.no_grad()
def rnn_inference(model_name, ckpt_path):
    from train_oasis.model.rnn_dit import DiT

    ddim_noise_steps = 50
    window_size = 20
    combine_actions = True
    metadata_file_path = "data/maze/metadata_self.json"
    save_dir = os.path.join("outputs/rnn/eval_outputs", f"rnn_{model_name}")
    metrics_save_path = os.path.join("outputs/rnn/eval_outputs/metrics", f"rnn_{model_name}.json")
    dtype = torch.bfloat16
    vae_batch_size = 128
    predict_v = True
    action_types = ["action", "agent_pos", "agent_dir"]
    external_cond_dim = 10

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    with open(metadata_file_path, 'r') as f:
        paths = json.load(f)[:40]
    paths = sorted(paths, key=lambda x: x["length"], reverse=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"

    if "LSTM" in model_name:
        cfg_rel_path = "config/model/rnn_lstm_maze.yaml"
    elif "Mamba" in model_name:
        cfg_rel_path = "config/model/rnn_mamba_maze.yaml"
    elif "TTT" in model_name:
        cfg_rel_path = "config/model/rnn_ttt_maze.yaml"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    cfg_path = os.path.join(dir_path, cfg_rel_path)
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    rnn_config = config["rnn_config"]
    if combine_actions:
        rnn_config["combine_action_dim"] = 128
    # 将 rnn_config（dict）转为支持属性访问的对象
    rnn_config = SimpleNamespace(**rnn_config)
    
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
    if os.path.isdir(ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"Model {model_name} loaded from {ckpt_path} and moved to {device}.")

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
    gt_videos = []
    all_actions = []
    save_paths = []
    all_metrics = []
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
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    for start_idx in range(len(prompts)):
        # sampling inputs
        x = prompts[start_idx]
        actions = all_actions[start_idx]
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
        
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = x_clip * 2 - 1
                    x_clip = vae.encode(x_clip).latent_dist.sample() * vae.config.scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
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
            video = torch.clamp(video, 0, 1)
            gt_video = gt_videos[start_idx + idx].to(device)
            metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
            all_metrics.append({
                "file": paths[start_idx + idx]["file"],
                "prompt_frames": n_prompt_frames,
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

@torch.no_grad()
def vanilla(model_name, ckpt_path):
    from train_oasis.model.dit import DiT

    ddim_noise_steps = 50
    window_size = 200
    metadata_file_path = "data/maze/metadata_self.json"
    save_dir = os.path.join("outputs/rnn/eval_outputs", f"df_{model_name}")
    metrics_save_path = os.path.join("outputs/rnn/eval_outputs/metrics", f"df_{model_name}.json")
    dtype = torch.bfloat16
    vae_batch_size = 128
    predict_v = True
    action_types = ["action", "agent_pos", "agent_dir"]
    external_cond_dim = 10

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    with open(metadata_file_path, 'r') as f:
        paths = json.load(f)[:40]
    paths = sorted(paths, key=lambda x: x["length"], reverse=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_paths = {
        "ws20": "config/model/latent_maze_dit.yaml",
        "ws200": "config/model/latent_maze_dit.yaml",
    }
    vae_ckpt = "stabilityai/sd-vae-ft-mse"
    if model_name not in config_paths:
        raise ValueError(f"Unknown model name: {model_name}")
    
    cfg_rel_path = config_paths[model_name]
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
    print(f"Model {model_name} loaded from {ckpt_path} and moved to {device}.")

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
    gt_videos = []
    all_actions = []
    save_paths = []
    all_metrics = []
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
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    for start_idx in range(len(prompts)):
        # sampling inputs
        x = prompts[start_idx]
        actions = all_actions[start_idx]
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
        
        with torch.no_grad():
            with autocast("cuda", dtype=dtype):
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
        x = rearrange(x, "(b t) c h w -> b t h w c", b=B, t=total_frames)

        for idx in range(B):
            # save video
            video = x[idx]
            video = torch.clamp(video, 0, 1)
            gt_video = gt_videos[start_idx + idx].to(device)
            metrics = get_validation_metrics_for_videos(video.permute(0, 3, 1, 2).unsqueeze(1), gt_video.permute(0, 3, 1, 2).unsqueeze(1))
            all_metrics.append({
                "file": paths[start_idx + idx]["file"],
                "prompt_frames": n_prompt_frames,
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

@torch.no_grad()
def rnn_check_in_context(model_name, ckpt_path):
    from train_oasis.model.rnn_dit import DiT

    ddim_noise_steps = 50
    window_size = 20
    combine_actions = True
    metadata_file_path = "data/maze/metadata.json"
    save_dir = os.path.join("outputs/rnn/abla_eval_outputs", f"rnn_{model_name}")
    metrics_save_path = os.path.join("outputs/rnn/abla_eval_outputs/metrics", f"rnn_{model_name}.json")
    dtype = torch.bfloat16
    vae_batch_size = 128
    predict_v = True
    action_types = ["action", "agent_pos", "agent_dir"]
    external_cond_dim = 10
    inference_batch_size = 100
    n_prompt_frames = 10
    total_length = 20

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    os.makedirs(save_dir, exist_ok=True)
    with open(metadata_file_path, 'r') as f:
        paths = json.load(f)["validation"][:100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_ckpt = "stabilityai/sd-vae-ft-mse"

    if "LSTM" in model_name:
        cfg_rel_path = "config/model/rnn_lstm_maze.yaml"
    elif "Mamba" in model_name:
        cfg_rel_path = "config/model/rnn_mamba_maze.yaml"
    elif "TTT" in model_name:
        cfg_rel_path = "config/model/rnn_ttt_maze.yaml"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    cfg_path = os.path.join(dir_path, cfg_rel_path)
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    rnn_config = config["rnn_config"]
    if combine_actions:
        rnn_config["combine_action_dim"] = 128
    # 将 rnn_config（dict）转为支持属性访问的对象
    rnn_config = SimpleNamespace(**rnn_config)
    
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
    if os.path.isdir(ckpt_path):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    print(f"Model {model_name} loaded from {ckpt_path} and moved to {device}.")

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
    gt_videos = []
    all_actions = []
    save_paths = []
    all_metrics = []
    for info in paths:
        prompt_path = info["file"]
        # length = info["length"]
        data = np.load(prompt_path, allow_pickle=True)

        x = data["image"][:n_prompt_frames] # (T, H, W, C)
        gt_video = data["image"][:total_length] # (T, H, W, C)
        gt_video = torch.from_numpy(gt_video).float() / 255.  # (T, H, W, C)
        gt_videos.append(gt_video)
        x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0).float() / 255. # (1, T, C, H, W)
        actions = torch.cat([torch.from_numpy(data[atype]) for atype in action_types], dim=-1)[:total_length]
        actions = actions.unsqueeze(0).float() # (1, T, 10)

        prompts.append(x)
        all_actions.append(actions)
        file_name = os.path.basename(prompt_path).replace(".npz", ".mp4")
        save_path = os.path.join(save_dir, file_name)
        save_paths.append(save_path)

    for start_idx in range(0, len(prompts), inference_batch_size):
        # sampling inputs
        end_idx = min(start_idx + inference_batch_size, len(prompts))
        x = torch.cat(prompts[start_idx:end_idx], dim=0)
        actions = torch.cat(all_actions[start_idx:end_idx], dim=0)
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
        x = rearrange(x, "(b t) ... -> b t ...", b=B, t=n_prompt_frames)

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_length), desc="sampling frames"):
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
                        v, _ = model.inference(x_curr, t, actions[:, start_frame : i + 1], hidden_states=None, get_new_hidden_states=False)
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
                "prompt_frames": n_prompt_frames,
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

def get_metrics():
    paths = [
        "outputs/rnn/eval_outputs/metrics/df_ws20.json",
        "outputs/rnn/eval_outputs/metrics/rnn_chunk_combine_LSTM.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_b256_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_pad_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch1.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch2.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch3.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch4.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch5.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_epoch6.json",
        "outputs/rnn/eval_outputs/metrics/rnn_LSTM_comb_onemem_aux_epoch3.json",
    ]

    for path in paths:
        with open(path, 'r') as f:
            all_metrics = json.load(f)
        metrics = {}
        for item in all_metrics:
            prompt_frames = item["prompt_frames"]
            for key, values in item["metrics"].items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].extend(values[prompt_frames:])
        avg_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        print(f"Metrics for {path}:")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")
        print()

if __name__ == "__main__":
    # get_metrics()
    # rnn_inference("LSTM_comb_onemem_b256_epoch3", "outputs/rnn/lstm_maze_pos_comb_onemem_b256/checkpoints/ckpt-epoch=2-step=1221.ckpt")
    # vanilla("ws200", "outputs/df/maze_200/checkpoints/epoch=0-step=54000.ckpt")
    # rnn_check_in_context("LSTM_comb_sft_unfreeze_epoch6", "outputs/rnn/lstm_maze_pos_comb_sft_unfreeze/checkpoints/ckpt-epoch=5-step=4890.ckpt")
    rnn_chunk_inference("Mamba", "outputs/rnn/rnn_chunk_mamba_maze_pos/checkpoints/epoch=2-step=2445.ckpt")