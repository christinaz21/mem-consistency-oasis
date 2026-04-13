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
from torchvision.io import write_video
from train_oasis.utils import sigmoid_beta_schedule, load_actions
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
from pytorchvideo.data.encoded_video import EncodedVideo
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from worldscore import DroidReprojectionScorer
import numpy as np

assert torch.cuda.is_available()
device = torch.device("cuda")
print(f"using device: {device}")


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = DiT_models[args.model_name]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith((".pt", ".bin")):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    elif os.path.isdir(args.oasis_ckpt):
        ckpt = get_fp32_state_dict_from_zero_checkpoint(args.oasis_ckpt)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"unsupported checkpoint format: {args.oasis_ckpt}")
    model = model.to(device).eval()

    # sampling params
    max_frame = model.max_frames # 10
    n_prompt_frames = 1
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    assert ddim_noise_steps % (max_frame-1) == 0, "ddim_noise_steps must be divisible by 9"
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1, dtype=torch.long)
    print(noise_range)
    noise_abs_max = 6
    stabilization_level = 15

    # get prompt image/video
    # x = load_prompt(
    #     args.prompt_path,
    #     video_offset=args.video_offset,
    #     n_prompt_frames=n_prompt_frames,
    # )
    video = EncodedVideo.from_path(args.prompt_path, decode_audio=False)
    video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
    video = video.permute(1, 2, 3, 0).numpy()[args.video_offset:args.video_offset+2*max_frame]
    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    # get input action stream
    actions = load_actions(args.actions_path, action_offset=args.video_offset)[:, :total_frames, :4].to(device)

    # sampling inputs
    x = video.to(device)
    B = 1
    H, W = x.shape[-2:]
    x = x.reshape(1, -1, 3, H, W)
    x = (x - 0.5) / 0.5

    # vae encoding
    x = x[:, :n_prompt_frames]
    if args.vae_ckpt:
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
        if args.vae_ckpt.endswith(".pt"):
            vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
            vae.load_state_dict(vae_ckpt)
        elif args.vae_ckpt.endswith(".safetensors"):
            load_model(vae, args.vae_ckpt)
        vae = vae.to(device).eval()
        scaling_factor = 0.07843137255
        x = rearrange(x, "b t c h w -> (b t) c h w")
        with torch.no_grad():
            with autocast("cuda", dtype=torch.half):
                x = vae.encode(x).mean * scaling_factor
        x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H // vae.patch_size, w=W // vae.patch_size)

    print("x shape: ", x.shape)
    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # sampling loop
    ddim_one_step = ddim_noise_steps // (max_frame - 1)
    # The first frame is stabilized, noise level for the last 9 frames
    noise_level_matrix = torch.zeros(
        (ddim_one_step + 1, (max_frame - 1)), device=device
    )
    for i in range(max_frame - 1):
        for step in range(ddim_one_step + 1):
            noise_level_matrix[step, i] = noise_range[step + i * ddim_one_step]

    '''
    max_frame = 10, ddim_noise_steps = 36
    0 4 8 12 16 20 24 28 32
    1 5 9 13 17 21 25 29 33
    2 6 10 14 18 22 26 30 34
    3 7 11 15 19 23 27 31 35
    4 8 12 16 20 24 28 32 36
    '''
    def _generate_one_candidate(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        x_work = x.clone()
        bar = tqdm(range(n_prompt_frames, total_frames + max_frame - 2))
        bar.set_description(f"Generating frames (seed={seed})")
        for i in range(n_prompt_frames, total_frames):
            chunk = torch.randn((B, 1, *x_work.shape[-3:]), device=device)  # (B, 1, C, H, W)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x_work = torch.cat([x_work, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)
            horizon = min(max_frame, i + 1)

            for noise_idx in reversed(range(1, ddim_one_step + 1)):
                t = torch.full(
                    (B, horizon),
                    stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t[:, 1:] = noise_level_matrix[noise_idx, max_frame - horizon :]
                t_next = torch.full(
                    (B, horizon),
                    stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t_next[:, 1:] = noise_level_matrix[noise_idx - 1, max_frame - horizon :]
                t_next = torch.where(t_next < 0, t, t_next)

                # sliding window
                x_curr = x_work.clone()
                x_curr = x_curr[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])

                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
                    1 / alphas_cumprod[t] - 1
                ).sqrt()

                # get frame prediction
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1 and horizon == max_frame:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x_work[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]

            bar.update(1)

        # handle last max_frame - 1 frames
        for i in range(total_frames, total_frames + max_frame - 2):
            start_frame = max(0, i + 1 - max_frame)
            horizon = min(max_frame, max_frame + total_frames - 1 - i)

            for noise_idx in reversed(range(1, ddim_one_step + 1)):
                t = torch.full(
                    (B, horizon),
                    stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t[:, 1:] = noise_level_matrix[noise_idx, : horizon - 1]
                t_next = torch.full(
                    (B, horizon),
                    stabilization_level - 1,
                    dtype=torch.long,
                    device=device,
                )
                t_next[:, 1:] = noise_level_matrix[noise_idx - 1, : horizon - 1]
                t_next = torch.where(t_next < 0, t, t_next)

                # sliding window
                x_curr = x_work.clone()
                x_curr = x_curr[:, start_frame:]

                # get model predictions
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])

                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
                    1 / alphas_cumprod[t] - 1
                ).sqrt()

                # get frame prediction
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x_work[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]
            bar.update(1)
        bar.close()

        if args.vae_ckpt:
            x_dec = rearrange(x_work, "b t c h w -> (b t) (h w) c")
            with torch.no_grad():
                x_dec = vae.decode(x_dec / scaling_factor)
            x_dec = rearrange(x_dec, "(b t) c h w -> b t h w c", t=total_frames)
        else:
            x_dec = x_work
        x_dec = x_dec * 0.5 + 0.5
        x_dec = torch.clamp(x_dec, 0, 1)
        return x_dec

    scorer = DroidReprojectionScorer(
        weights_path=args.droid_weights,
        calib=tuple(float(v) for v in args.droid_calib.split(",")),
        stride=args.droid_stride,
        buffer=args.droid_buffer,
        filter_thresh=args.droid_filter_thresh,
        upsample=args.droid_upsample,
        quiet=True,
        resize_long_side=args.droid_resize,
        max_frames=args.droid_max_frames,
    )

    candidate_videos = []
    candidate_scores = []
    candidate_paths = []
    output_path = args.output_path
    base_root, base_ext = os.path.splitext(output_path)
    if base_ext == "":
        base_ext = ".mp4"
        output_path = base_root + base_ext
    for p in range(args.num_particles):
        seed = args.base_seed + p
        x_dec = _generate_one_candidate(seed=seed)  # [B,T,H,W,C]
        candidate_videos.append(x_dec)
        score = scorer.reward_from_video(x_dec[0])
        candidate_scores.append(score)
        cand_path = f"{base_root}_seed{seed}{base_ext}"
        candidate_paths.append(cand_path)
        print(f"Droid reward (seed={seed}): {score}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for cand_path, x_dec in zip(candidate_paths, candidate_videos):
        x_out = (x_dec[0] * 255).byte().cpu().numpy()
        write_video(cand_path, x_out, fps=args.fps)


    best_idx = int(np.argmax(candidate_scores))
    best_path = candidate_paths[best_idx]
    print("All candidate scores:")
    for cand_path, score in zip(candidate_paths, candidate_scores):
        print(f"{cand_path}\t{score}")
    print(f"Best candidate: {best_path} (score={candidate_scores[best_idx]})")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="/home/tc0786/Project/train-oasis/outputs/2025-05-07/03-28-14/checkpoints/epoch=2-step=13000.ckpt",
    )
    parse.add_argument(
        "--model-name",
        type=str,
        help="Model name",
        default="dit_easy",
    )
    parse.add_argument(
        "--vae-ckpt",
        type=str,
        help="Path to Oasis ViT-VAE checkpoint.",
        default="models/oasis500m/vit-l-20.safetensors",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=300,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="/home/tc0786/Project/train-oasis/data/eval_data/memory/rotate/000000.mp4",
    )
    parse.add_argument(
        "--actions-path",
        type=str,
        help="File to load actions from (.actions.pt or .one_hot_actions.pt)",
        default="/home/tc0786/Project/train-oasis/data/eval_data/memory/rotate/000000.npz",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=0,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="/home/tc0786/Project/train-oasis/outputs/eval_outputs/vanilla_10/memory/rotate/000000.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=18)
    parse.add_argument("--num-particles", type=int, default=4, help="Number of candidates for best-of-n.")
    parse.add_argument("--base-seed", type=int, default=0, help="Base seed for candidate sampling.")
    parse.add_argument("--droid-weights", type=str, default="/u/cz5047/videogen/data/models/droid_models/droid.pth")
    parse.add_argument("--droid-calib", type=str, default="500,500,256,256")
    parse.add_argument("--droid-stride", type=int, default=2)
    parse.add_argument("--droid-max-frames", type=int, default=200)
    parse.add_argument("--droid-resize", type=int, default=256)
    parse.add_argument("--droid-buffer", type=int, default=512)
    parse.add_argument("--droid-filter-thresh", type=float, default=0.01)
    parse.add_argument("--droid-upsample", action="store_true")

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
