import sys
import os
dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

import torch
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
from train_oasis.model.lstm_dit import DiT

assert torch.cuda.is_available()
device = "cuda"

@torch.no_grad()
def lstm():
    model_name = "lstm_chunk" # "vanilla_10", "vanilla_20", "world_coordinate", "pred_x"
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

    oasis_ckpt = "/home/tc0786/Project/train-oasis/outputs/df/lstm_long_2/checkpoints/epoch=0-step=41000.ckpt"
    inner_window_size = 10
    vae_ckpt = "/home/tc0786/Project/train-oasis/models/oasis500m/vit-l-20.safetensors"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
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
        total_frames = 300 if inference_split == "memory" else 1200
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
            state_buffer_list = None
            for i in tqdm(range(0, n_prompt_frames, inner_window_size), desc="lstm memory prompt"):
                end_frame = i + inner_window_size
                start_frame = i
                t = torch.full((B, end_frame - start_frame), stabilization_level, dtype=torch.long, device=device)
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        _, state_buffer_list = model.chunk_inference(x[:, start_frame:end_frame], t, actions[:, start_frame:end_frame], state_buffer_list)

            # sampling loop
            for i in tqdm(range(n_prompt_frames, total_frames, inner_window_size), desc="Sampling"):
                chunk = torch.randn((B, inner_window_size, *x.shape[-3:]), device=device) # (B, 10, C, H, W)
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
                x = torch.cat([x, chunk], dim=1)
                start_frame = i

                for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
                    # set up noise values
                    t = torch.full((B, inner_window_size), noise_range[noise_idx], dtype=torch.long, device=device)
                    t_next = torch.full((B, inner_window_size), noise_range[noise_idx - 1], dtype=torch.long, device=device)
                    t_next = torch.where(t_next < 0, t, t_next)

                    # sliding window
                    x_curr = x.clone()
                    x_curr = x_curr[:, start_frame:]

                    # get model predictions
                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.half):
                            v, new_state_buffer_list = model.chunk_inference(x_curr, t, actions[:, start_frame : start_frame + inner_window_size], state_buffer_list)
                            if noise_idx == 1:
                                state_buffer_list = new_state_buffer_list
                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

                    # get frame prediction
                    alpha_next = alphas_cumprod[t_next]
                    if noise_idx == 1:
                        alpha_next = torch.ones_like(alpha_next)
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, -inner_window_size:] = x_pred[:, -inner_window_size:]
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
    lstm()