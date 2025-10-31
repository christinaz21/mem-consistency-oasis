"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
from train_oasis.model.dit import DiT_models
from train_oasis.model.vae import VAE_models
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

assert torch.cuda.is_available()
device = "cuda"

@torch.no_grad()
def vanilla():
    model_name = "vanilla_20_longer" # "vanilla_10", "vanilla_20", "world_coordinate", "pred_x"
    inference_splits = ["memory", "random"]
    ddim_noise_steps = 20
    n_prompt_frames = 100
    video_offset = None
    batch_size = 10
    file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(file_path, "r") as f:
        paths = json.load(f)

    save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    if model_name == "vanilla_10":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-07/03-28-14/checkpoints/epoch=2-step=13000.ckpt"
        window_size = 10
    elif model_name == "vanilla_20":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/checkpoints/1000data_3epoch_predictv.bin"
        window_size = 20
    elif model_name == "world_coordinate":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-08/02-24-21/checkpoints/epoch=2-step=6000.ckpt"
        window_size = 20
    elif model_name == "pred_x":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/checkpoints/1000data_3epoch_predictx.bin"
        window_size = 20
    elif model_name == "vanilla_20_longer":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-25/07-51-43/checkpoints/epoch=0-step=41000.ckpt"
        window_size = 20
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.dit import DiT
    if model_name == "vanilla_10":
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=4,
            max_frames=10
        )
    elif model_name == "vanilla_20" or model_name == "pred_x" or model_name == "vanilla_20_longer":
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=4,
            max_frames=20
        )
    elif model_name == "world_coordinate":
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=8,
            max_frames=20
        )
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
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
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

    if "memory" in inference_splits:
        # memory tasks
        memory_meta_data = paths["memory"]
        total_frames = 301
        prompts = []
        all_actions = []
        save_paths = []
        for info in memory_meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            if model_name == "vanilla_10" or model_name == "vanilla_20" or model_name == "pred_x" or model_name == "vanilla_20_longer":
                actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames, :4]
            elif model_name == "world_coordinate":
                actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames]
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

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
            x = rearrange(x, "b t c h w -> (b t) (h w) c")
            vae_batch_size = 128
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = (vae.decode(x_clip / scaling_factor) + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

            for idx in range(B):
                # save video
                video = x[idx]
                video = video.cpu()
                video = torch.clamp(video, 0, 1)
                video = (video * 255).byte()
                output_path = save_paths[start_idx + idx]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_video(output_path, video, fps=fps)

    if "random" in inference_splits:
        # random tasks
        random_meta_data = paths["random"]
        total_frames = 1201
        prompts = []
        all_actions = []
        save_paths = []
        for info in random_meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            if model_name == "vanilla_10" or model_name == "vanilla_20" or model_name == "pred_x" or model_name == "vanilla_20_longer":
                actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames, :4]
            elif model_name == "world_coordinate":
                actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames]
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

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
            x = rearrange(x, "b t c h w -> b t (h w) c")
            with torch.no_grad():
                for idx in tqdm(range(B)):
                    video = x[idx]
                    video = (vae.decode(video / scaling_factor) + 1) / 2
                    video = rearrange(video, "t c h w -> t h w c")
                    video = video.cpu()
                    video = torch.clamp(video, 0, 1)
                    video = (video * 255).byte()
                    output_path = save_paths[start_idx + idx]
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    write_video(output_path, video, fps=fps)


def retrieve_frame_idx(actions, retrieve_num, pred_action, similarity_func="euclidean"):
    """
    Retrieve the frame index of the action that is most similar to the predicted action.
    pred_action: (B, action_dim)
    actions: (B, num_actions, action_dim)
    retrieve_num: number of actions to retrieve
    """
    weights = torch.tensor([10,10,10,3], dtype=torch.float32, device=device)
    pred_action = pred_action * weights
    pred_action = pred_action.unsqueeze(1)  # (B, 1, D)
    actions = actions * weights

    if similarity_func == "cosine":
        similarity = 1 - torch.nn.functional.cosine_similarity(actions, pred_action, dim=-1)
    elif similarity_func == "euclidean":
        similarity = torch.norm(actions - pred_action, dim=-1)
    else:
        raise ValueError(f"unsupported similarity function: {similarity_func}")
    # retrieve the top-k most similar actions
    # similarity: (B, N)
    topk_idx = torch.topk(similarity, retrieve_num, largest=False).indices
    return topk_idx

def retrieve_frame_idx_multiple(actions, retrieve_num, pred_action, similarity_func="euclidean"):
    """
    Retrieve the frame index of the action that is most similar to the predicted action.
    pred_action: (B, num_condition, action_dim)
    actions: (B, num_actions, action_dim)
    retrieve_num: number of actions to retrieve
    """
    assert pred_action.shape[1] == retrieve_num, f"pred_action shape {pred_action.shape} does not match retrieve_num {retrieve_num}"
    weights = torch.tensor([10,10,10,3], dtype=torch.float32, device=device)
    pred_action = pred_action * weights
    actions = actions * weights

    pred_action = pred_action.unsqueeze(2)  # (B, R, 1, D)
    actions = actions.unsqueeze(1)  # (B, 1, N, D)

    if similarity_func == "cosine":
        similarity = 1 - torch.nn.functional.cosine_similarity(actions, pred_action, dim=-1)
    elif similarity_func == "euclidean":
        similarity = torch.norm(actions - pred_action, dim=-1)
    else:
        raise ValueError(f"unsupported similarity function: {similarity_func}")
    similarity += 1e-5 * torch.arange(similarity.shape[-1], device=similarity.device).unsqueeze(0).unsqueeze(0)  # to ensure unique indices
    # similarity: (B, R, N)
    # retrieve the top-k most similar actions
    topk_idx = torch.topk(similarity, 1, dim=-1, largest=False).indices.squeeze(-1)  # (B, R)
    # (B, retrieve_num)
    topk_idx, _ = torch.sort(topk_idx, dim=-1)
    return topk_idx

@torch.no_grad()
def rag():
    ddim_noise_steps = 20
    total_frames = 300
    video_offset = None
    batch_size = 10
    n_prompt_frames = 100
    # file_path = "/home/tc0786/Project/train-oasis/data/eval_data/pred_pose_paths.json"
    file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(file_path, "r") as f:
        paths = json.load(f)

    inference_splits = ["memory", "random"]

    model_name = "rag_multi" # "rag_wo_training" "rag" "rag_multi"

    if model_name == "rag" or model_name == "rag_pred_pose":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-12/12-57-27/checkpoints/epoch=2-step=12000.ckpt"
    elif model_name == "rag_wo_training":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-08/02-24-21/checkpoints/epoch=2-step=6000.ckpt"
    elif model_name == "rag_multi" or model_name == "rag_multi_pred_pose":
        # oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-18/03-14-16/checkpoints/epoch=1-step=10000.ckpt"
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/checkpoints/anjian/18-43-16/checkpoints/epoch=0-step=38000.ckpt"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    window_size = 20
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    if model_name == "rag" or model_name == "rag_pred_pose" or model_name == "rag_multi" or model_name == "rag_multi_pred_pose":
        from train_oasis.model.rag_dit import DiT
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=8,
            max_frames=20
        )
    elif model_name == "rag_wo_training":
        from train_oasis.model.dit import DiT
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=8,
            max_frames=20
        )
    # print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    ckpt = get_fp32_state_dict_from_zero_checkpoint(oasis_ckpt)
    state_dict = {}
    for key, value in ckpt.items():
        if key.startswith("diffusion_model."):
            state_dict[key[16:]] = value
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15
    fps = 20
    retrieve_num = 10

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model.max_frames = window_size

    for split in inference_splits:
        # random tasks
        random_meta_data = paths[split]
        total_frames = 1201 if split == "random" else 301
        prompts = []
        all_actions = []
        save_paths = []
        for info in random_meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames]
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"
            actions[:, 1:, 4:] = actions[:, 1:, 4:] - actions[:, 1, 4:]

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

            # sampling loop
            for i in tqdm(range(n_prompt_frames, total_frames)):
                chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
                x = torch.cat([x, chunk], dim=1)
                start_frame = max(0, i + 1 - model.max_frames)

                if i >= model.max_frames:
                    # retrieve actions
                    context_frame = start_frame + retrieve_num
                    candidate_actions = actions[:, :context_frame, 4:]
                    if model_name == "rag_multi":
                        retrieved_idx = retrieve_frame_idx_multiple(
                            candidate_actions,
                            retrieve_num=retrieve_num,
                            pred_action=actions[:, context_frame:i+1, 4:],
                            similarity_func="euclidean",
                        )
                    else:
                        retrieved_idx = retrieve_frame_idx(
                            candidate_actions,
                            retrieve_num=retrieve_num,
                            pred_action=actions[:, i, 4:],
                            similarity_func="euclidean",
                        )
                    batch_indices = torch.arange(actions.shape[0], device=actions.device).unsqueeze(-1).expand(-1, retrieved_idx.shape[1])
                    retrieved_actions = actions[batch_indices, retrieved_idx]
                    # retrieved_actions = actions[:, retrieved_idx]
                    retrieved_frames = x[batch_indices, retrieved_idx]
                    retrieved_actions[:, :, :4] = 0
                    context_actions = torch.cat([retrieved_actions, actions[:, context_frame:i+1]], dim=1)
                    context_frames = torch.cat([retrieved_frames, x[:, context_frame:i]], dim=1)
                else:
                    raise ValueError(f"i={i} < model.max_frames={model.max_frames}")
                
                for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                    # set up noise values
                    t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
                    t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
                    t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                    t_next = torch.where(t_next < 0, t, t_next)
                    t = torch.cat([t_ctx, t], dim=1)
                    t_next = torch.cat([t_ctx, t_next], dim=1)

                    # sliding window
                    x_curr = torch.cat([context_frames, x[:, -1:]], dim=1).clone()
                    t = t[:, start_frame:]
                    t_next = t_next[:, start_frame:]

                    # get model predictions
                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.half):
                            v = model(x_curr, t, context_actions)
                    
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
            x = rearrange(x, "b t c h w -> b t (h w) c")
            with torch.no_grad():
                for idx in tqdm(range(B)):
                    video = x[idx]
                    video = (vae.decode(video / scaling_factor) + 1) / 2
                    video = rearrange(video, "t c h w -> t h w c")
                    video = video.cpu()
                    video = torch.clamp(video, 0, 1)
                    video = (video * 255).byte()
                    output_path = save_paths[start_idx + idx]
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    write_video(output_path, video, fps=fps)

def infini_attention():
    ddim_noise_steps = 20
    total_frames = 120
    video_offset = None
    batch_size = 10
    n_prompt_frames = 100
    file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(file_path, "r") as f:
        paths = json.load(f)

    inference_splits = ["memory", "random"]

    model_name = "infini_attn" # "vanilla_10"

    oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-07/11-28-33/checkpoints/epoch=2-step=17000.ckpt"
    window_size = 20
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.attn_mem_dit import DiT
    model = DiT(
        input_h=18,
        input_w=32,
        in_channels=16,
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        external_cond_dim=4,
        max_frames=20,
        stride=10,
        delta_update=True
    )
    # print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    ckpt = get_fp32_state_dict_from_zero_checkpoint(oasis_ckpt)
    state_dict = {}
    for key, value in ckpt.items():
        if key.startswith("diffusion_model."):
            state_dict[key[16:]] = value
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    fps = 20

    # get alphas
    model.max_frames = window_size

    for split in inference_splits:
        # random tasks
        random_meta_data = paths[split]
        total_frames = 1200 if split == "random" else 300
        prompts = []
        all_actions = []
        save_paths = []
        for info in random_meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames, :4]
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"
            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

            with torch.no_grad():
                x = model.sample(x, n_context_frames=n_prompt_frames, n_frames=total_frames, sampling_timesteps=ddim_noise_steps, external_cond=actions)

            # vae decoding
            x = rearrange(x, "b t c h w -> b t (h w) c")
            with torch.no_grad():
                for idx in tqdm(range(B)):
                    video = x[idx]
                    video = (vae.decode(video / scaling_factor) + 1) / 2
                    video = rearrange(video, "t c h w -> t h w c")
                    video = video.cpu()
                    video = torch.clamp(video, 0, 1)
                    video = (video * 255).byte()
                    output_path = save_paths[start_idx + idx]
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    write_video(output_path, video, fps=fps)

@torch.no_grad()
def rag_folder():
    ddim_noise_steps = 20
    total_frames = 150
    video_offset = None
    batch_size = 20
    n_prompt_frames = 100
    save_dir = "/home/tc0786/Project/train-oasis/outputs/eval_outputs/"

    '''
    video_dir = "/home/tc0786/Project/train-oasis/data/eval_data/additional_mem"
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for i in range(30):
        video_path = os.path.join(video_dir, f"{i:06d}.mp4")
        action_path = os.path.join(video_dir, f"{i:06d}.npz")
        save_relative_path = os.path.join("additional_mem_rag", f"{i:06d}.mp4")
        paths.append({
            "video_path": video_path,
            "action_path": action_path,
            "save_relative_path": save_relative_path
        })
        '''
    paths = []
    video_dir = "/home/tc0786/Project/train-oasis/data/eval_data/additional_rotate"
    for i in range(300,600):
        video_path = os.path.join(video_dir, f"{i:06d}.mp4")
        action_path = os.path.join(video_dir, f"{i:06d}.npz")
        save_relative_path = os.path.join("rag/memory/additional_rotate", f"{i:06d}.mp4")
        paths.append({
            "video_path": video_path,
            "action_path": action_path,
            "save_relative_path": save_relative_path
        })

    model_name = "rag" # "rag_wo_training" "rag"

    if model_name == "rag":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-12/12-57-27/checkpoints/epoch=2-step=12000.ckpt"
    elif model_name == "rag_wo_training":
        oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-05-08/02-24-21/checkpoints/epoch=2-step=6000.ckpt"
    window_size = 20
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"


    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    if model_name == "rag":
        from train_oasis.model.rag_dit import DiT
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=8,
            max_frames=20
        )
    elif model_name == "rag_wo_training":
        from train_oasis.model.dit import DiT
        model = DiT(
            input_h=18,
            input_w=32,
            in_channels=16,
            patch_size=2,
            hidden_size=1024,
            depth=16,
            num_heads=16,
            external_cond_dim=8,
            max_frames=20
        )
    # print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(oasis_ckpt)}...")
    ckpt = get_fp32_state_dict_from_zero_checkpoint(oasis_ckpt)
    state_dict = {}
    for key, value in ckpt.items():
        if key.startswith("diffusion_model."):
            state_dict[key[16:]] = value
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # load VAE checkpoint
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
    vae = vae.to(device).eval()

    # sampling params
    max_noise_level = 1000
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
    noise_abs_max = 6 # open oasis use 20
    stabilization_level = 15
    fps = 20
    retrieve_num = 10

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")
    model.max_frames = window_size

    # random tasks
    for start_idx in range(0, len(paths), batch_size):
        random_meta_data = paths[start_idx : start_idx + batch_size]
        prompts = []
        all_actions = []
        save_paths = []
        for info in random_meta_data:
            try:
                prompt_path = info["video_path"]
                x = load_prompt(
                    prompt_path,
                    video_offset=video_offset,
                    n_prompt_frames=n_prompt_frames,
                )

                # get input action stream
                actions_path = info["action_path"]
                actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames]
                assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"
                assert x.shape[0] == actions.shape[0], f"{x.shape[0]} != {actions.shape[0]}"
            except Exception as e:
                print(f"Error loading video {info['video_path']}: {e}")
                continue

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
            save_paths.append(save_path)

        x = torch.cat(prompts, dim=0)
        actions = torch.cat(all_actions, dim=0)
        assert x.shape[0] == actions.shape[0], f"{x.shape[0]} != {actions.shape[0]}"
        
        # sampling inputs
        # x = prompts[start_idx : start_idx + batch_size]
        # actions = all_actions[start_idx : start_idx + batch_size]
        B = x.shape[0]
        H, W = x.shape[-2:]
        # assert B == batch_size, f"{B} != {batch_size}"
        # x = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        # x = torch.clamp(x, -noise_abs_max, +noise_abs_max)
        x = x.to(device)
        actions = actions.to(device)

        # vae encoding
        scaling_factor = 0.07843137255
        x = rearrange(x, "b t c h w -> (b t) c h w")
        vae_batch_size = 128
        all_frames = []
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                    all_frames.append(x_clip)
            x = torch.cat(all_frames, dim=0)
        x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
        x = x[:, :n_prompt_frames]

        # sampling loop
        for i in tqdm(range(n_prompt_frames, total_frames)):
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x = torch.cat([x, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)

            if i >= model.max_frames:
                # retrieve actions
                context_frame = start_frame + retrieve_num
                candidate_actions = actions[:, :context_frame, 4:]
                retrieved_idx = retrieve_frame_idx(
                    candidate_actions,
                    retrieve_num=retrieve_num,
                    pred_action=actions[:, i, 4:],
                    similarity_func="euclidean",
                )
                retrieved_actions = actions[:, retrieved_idx]
                retrieved_frames = x[:, retrieved_idx]
                retrieved_actions[:, :, :4] = 0
                context_actions = torch.cat([retrieved_actions, actions[:, context_frame:i+1]], dim=1)
                context_frames = torch.cat([retrieved_frames, x[:, context_frame:i]], dim=1)
            else:
                raise ValueError(f"i={i} < model.max_frames={model.max_frames}")
            
            for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                # set up noise values
                t_ctx = torch.full((B, i), stabilization_level - 1, dtype=torch.long, device=device)
                t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
                t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                t_next = torch.where(t_next < 0, t, t_next)
                t = torch.cat([t_ctx, t], dim=1)
                t_next = torch.cat([t_ctx, t_next], dim=1)

                # sliding window
                x_curr = torch.cat([context_frames, x[:, -1:]], dim=1).clone()
                t = t[:, start_frame:]
                t_next = t_next[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, context_actions)
                
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
        x = rearrange(x, "b t c h w -> b t (h w) c")
        with torch.no_grad():
            for idx in tqdm(range(B)):
                video = x[idx]
                video = (vae.decode(video / scaling_factor) + 1) / 2
                video = rearrange(video, "t c h w -> t h w c")
                video = video.cpu()
                video = torch.clamp(video, 0, 1)
                video = (video * 255).byte()
                output_path = save_paths[idx]
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                write_video(output_path, video, fps=fps)

def check_length():
    paths = []
    video_dir = "/home/tc0786/Project/train-oasis/data/eval_data/additional_rotate"
    for i in range(300,600):
        video_path = os.path.join(video_dir, f"{i:06d}.mp4")
        action_path = os.path.join(video_dir, f"{i:06d}.npz")
        save_relative_path = os.path.join("rag/memory/additional_rotate", f"{i:06d}.mp4")
        paths.append({
            "video_path": video_path,
            "action_path": action_path,
            "save_relative_path": save_relative_path
        })

    for info in paths:
        action_path = info["action_path"]
        actions = load_actions(action_path, action_offset=None)
        print(actions.shape)

@torch.no_grad()
def frame_pack():
    model_name = "frame_pack" # "vanilla_10", "vanilla_20", "world_coordinate", "pred_x"
    inference_splits = ["memory", "random"]
    ddim_noise_steps = 20
    n_prompt_frames = 100
    video_offset = None
    batch_size = 10
    file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(file_path, "r") as f:
        paths = json.load(f)

    save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/2025-07-26/11-34-04/checkpoints/epoch=0-step=27000.ckpt"
    window_size = 10
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.frame_pack_dit import DiT
    model = DiT(
        input_h=18,
        input_w=32,
        in_channels=16,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        external_cond_dim=4,
        max_frames=10
    )
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
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
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


    for inference_split in inference_splits:
        # memory tasks
        meta_data = paths[inference_split]
        total_frames = 301 if inference_split == "memory" else 1200
        prompts = []
        all_actions = []
        save_paths = []
        for info in meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames, :4]
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

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
                    x_start = alphas_cumprod[t].sqrt() * x_curr[:, -1:] - (1 - alphas_cumprod[t]).sqrt() * v[:, -1:]
                    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr[:, -1:] - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                    # get frame prediction
                    alpha_next = alphas_cumprod[t_next]
                    alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                    if noise_idx == 1:
                        alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, -1:] = x_pred[:, -1:]

            # vae decoding
            x = rearrange(x, "b t c h w -> (b t) (h w) c")
            vae_batch_size = 128
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = (vae.decode(x_clip / scaling_factor) + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

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
def lstm():
    model_name = "lstm_long_term" # "vanilla_10", "vanilla_20", "world_coordinate", "pred_x"
    inference_splits = ["memory", "random"]
    ddim_noise_steps = 20
    n_prompt_frames = 100
    video_offset = None
    batch_size = 10
    file_path = "/home/tc0786/Project/train-oasis/data/eval_data/paths.json"
    with open(file_path, "r") as f:
        paths = json.load(f)

    save_dir = f"/home/tc0786/Project/train-oasis/outputs/eval_outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/df/lstm_long_term/checkpoints/epoch=0-step=21000.ckpt"
    inner_window_size = 10
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    from train_oasis.model.lstm_dit import DiT
    model = DiT(
        input_h=18,
        input_w=32,
        in_channels=16,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        external_cond_dim=4,
        max_frames=40,
        inner_window_size=inner_window_size,
    )
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
    vae = VAE_models["vit-l-20-shallow-encoder"]()
    # print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(vae_ckpt)}...")
    load_model(vae, vae_ckpt)
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
    # model.max_frames = window_size

    for inference_split in inference_splits:
        # memory tasks
        meta_data = paths[inference_split]
        total_frames = 301 if inference_split == "memory" else 1200
        prompts = []
        all_actions = []
        save_paths = []
        for info in meta_data:
            prompt_path = info["video_path"]
            x = load_prompt(
                prompt_path,
                video_offset=video_offset,
                n_prompt_frames=n_prompt_frames,
            )

            # get input action stream
            actions_path = info["action_path"]
            actions = load_actions(actions_path, action_offset=video_offset)[:, :total_frames, :4]
            assert actions.shape[1] == total_frames, f"{actions.shape[1]} != {total_frames}"

            prompts.append(x)
            all_actions.append(actions)
            save_path = os.path.join(save_dir, info["save_relative_path"])
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
            scaling_factor = 0.07843137255
            x = rearrange(x, "b t c h w -> (b t) c h w")
            vae_batch_size = 128
            all_frames = []
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae encoding frames"):
                        x_clip = x[idx:idx + vae_batch_size]
                        x_clip = vae.encode(x_clip * 2 - 1).mean * scaling_factor
                        all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)
            x = x[:, :n_prompt_frames]

            # for lstm memory
            output_buffer_list = None
            state_buffer_list = None
            for i in tqdm(range(n_prompt_frames), desc="lstm memory prompt"):
                end_frame = i + 1
                start_frame = max(0, end_frame - inner_window_size)
                t = torch.full((B, end_frame - start_frame), stabilization_level, dtype=torch.long, device=device)
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        _, output_buffer_list, state_buffer_list = model.inference(x[:, start_frame:end_frame], t, actions[:, start_frame:end_frame], output_buffer_list, state_buffer_list, update=True)

            # sampling loop
            for i in tqdm(range(n_prompt_frames, total_frames)):
                chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
                x = torch.cat([x, chunk], dim=1)
                start_frame = max(0, i + 1 - inner_window_size)

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

                    update = True if noise_idx == 1 else False
                    # get model predictions
                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.half):
                            v, output_buffer_list, state_buffer_list = model.inference(x_curr, t, actions[:, start_frame : i + 1], output_buffer_list, state_buffer_list, update=update)
                    x_start = alphas_cumprod[t].sqrt() * x_curr[:, -1:] - (1 - alphas_cumprod[t]).sqrt() * v[:, -1:]
                    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr[:, -1:] - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                    # get frame prediction
                    alpha_next = alphas_cumprod[t_next]
                    alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
                    if noise_idx == 1:
                        alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, -1:] = x_pred[:, -1:]
                    assert torch.isnan(x).sum() == 0, f"NaN detected at frame {i}, noise_idx {noise_idx}"

            # vae decoding
            x = rearrange(x, "b t c h w -> (b t) (h w) c")
            vae_batch_size = 128
            with torch.no_grad():
                all_frames = []
                for idx in tqdm(range(0, x.shape[0], vae_batch_size), desc="vae decoding frames"):
                    x_clip = x[idx:idx + vae_batch_size]
                    x_clip = (vae.decode(x_clip / scaling_factor) + 1) / 2
                    all_frames.append(x_clip)
                x = torch.cat(all_frames, dim=0)
            x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)
            # assert torch.isnan(x).sum() == 0, f"NaN detected after vae decoding"

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
    rag()
    # check_length()
    # vanilla()
