"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""

import torch
from dit import DiT_models
from torchvision.io import read_video, write_video, write_png
from utils import sigmoid_beta_schedule, read_image, resize, extract
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import os
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms
from pathlib import Path

assert torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
print(f"using device: {device}")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
VIDEO_EXTENSIONS = {"mp4"}
def load_prompt(path, video_offset=None, n_prompt_frames=1):
    if path.lower().split(".")[-1] in IMAGE_EXTENSIONS:
        print("prompt is image; ignoring video_offset and n_prompt_frames")
        prompt = read_image(path)
        # add frame dimension
        prompt = rearrange(prompt, "c h w -> 1 c h w")
    elif path.lower().split(".")[-1] in VIDEO_EXTENSIONS:
        prompt = read_video(path, pts_unit="sec")[0]
        if video_offset is not None:
            prompt = prompt[video_offset:]
        prompt = prompt[:n_prompt_frames]
    else:
        raise ValueError(f"unrecognized prompt file extension; expected one in {IMAGE_EXTENSIONS} or {VIDEO_EXTENSIONS}")
    assert prompt.shape[0] == n_prompt_frames, f"input prompt {path} had less than n_prompt_frames={n_prompt_frames} frames"
    prompt = rearrange(prompt, "t h w c -> t c h w")
    prompt = resize(prompt, (32, 32))
    # add batch dimension
    prompt = rearrange(prompt, "t c h w -> 1 t c h w")
    prompt = prompt.float() / 255.0
    return prompt

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = DiT_models["self_train"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".pt") or args.oasis_ckpt.endswith(".ckpt"):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu")['state_dict']
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    model = model.to(device).eval()

    # sampling params
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
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
    video = video.permute(1, 2, 3, 0).numpy()[4:24]
    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    transform = transforms.Resize((64, 64), antialias=True)
    video = transform(video)[::2]
    # get input action stream
    actions = torch.zeros((1, args.num_frames, 4), device=device)
    actions[:, 0] = 1.0

    # sampling inputs
    x = video.to(device)
    x = x.reshape(1, -1, 3, 64, 64)
    x = (x - 0.5) / 0.5

    # vae encoding
    B = x.shape[0]
    H, W = x.shape[-2:]
    x = x[:, :n_prompt_frames]

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device) # (B, 1, C, H, W)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        # print(x[0][-1])
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
                    x_start = model(x_curr, t, actions[:, start_frame : i + 1])

            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            alpha_next = alphas_cumprod[t_next]
            alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])
            if noise_idx == 1:
                alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])
            x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    # save video
    x = x * 0.5 + 0.5
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    x = x[0].cpu()
    os.makedirs(args.output_path, exist_ok=True)
    for i in range(x.shape[0]):
        write_png(x[i], os.path.join(args.output_path, f"{i:04d}.png"))
    x = x.permute(0, 2, 3, 1).numpy()
    write_video(os.path.join(args.output_path, "video.mp4"), x, fps=args.fps)
    print(f"generation saved to {args.output_path}.")

import numpy as np

@torch.no_grad()
def same_as_validation(args):
    model = DiT_models["self_train"]()
    print(f"loading Oasis-500M from oasis-ckpt={os.path.abspath(args.oasis_ckpt)}...")
    if args.oasis_ckpt.endswith(".pt") or args.oasis_ckpt.endswith(".ckpt"):
        ckpt = torch.load(args.oasis_ckpt, map_location="cpu")['state_dict']
        data_mean = ckpt["data_mean"].to(device)
        data_std = ckpt["data_std"].to(device)
        alphas_cumprod = ckpt["alphas_cumprod"].to(device)
        sqrt_alphas_cumprod = ckpt["sqrt_alphas_cumprod"].to(device)
        sqrt_one_minus_alphas_cumprod = ckpt["sqrt_one_minus_alphas_cumprod"].to(device)
        sqrt_recip_alphas_cumprod = ckpt["sqrt_recip_alphas_cumprod"].to(device)
        sqrt_recipm1_alphas_cumprod = ckpt["sqrt_recipm1_alphas_cumprod"].to(device)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("diffusion_model."):
                state_dict[key[16:]] = value
        model.load_state_dict(state_dict)
    elif args.oasis_ckpt.endswith(".safetensors"):
        load_model(model, args.oasis_ckpt)
    model = model.to(device).eval()

    video_path = Path(args.prompt_path)
    action_path = video_path.with_suffix(".npz")
    video = EncodedVideo.from_path(video_path, decode_audio=False)
    video = video.get_clip(start_sec=0.0, end_sec=video.duration)["video"]
    video = video.permute(1, 2, 3, 0).numpy()
    actions = np.load(action_path)["actions"][1:]

    video = video[args.video_offset : args.video_offset + args.num_frames]  # (t, h, w, 3)
    actions = actions[args.video_offset : args.video_offset + args.num_frames]  # (t, )
    assert len(video) == len(actions)
    actions = np.eye(4)[actions]  # (t, 4)

    pad_len = args.num_frames - len(video)

    nonterminal = np.ones(args.num_frames)
    if len(video) < args.num_frames:
        video = np.pad(video, ((0, pad_len), (0, 0), (0, 0), (0, 0)))
        actions = np.pad(actions, ((0, pad_len),))
        nonterminal[-pad_len:] = 0

    video = torch.from_numpy(video / 255.0).float().permute(0, 3, 1, 2).contiguous()
    transform = transforms.Resize((64, 64), antialias=True)
    video = transform(video).reshape(1, 10, 3, 64, 64)
    actions = actions.reshape(1, 10, 4)
    nonterminal = nonterminal.reshape(1, 10)


    xs = video.to(device)
    batch_size, n_frames = 1, args.num_frames
    masks = torch.ones(n_frames, batch_size).to(device)
    conditions = torch.tensor(actions).to(device)
    conditions = torch.cat([torch.zeros_like(conditions[:, :1]), conditions[:, 1:]], 1)
    conditions = rearrange(conditions, "b t d -> t b d").float().contiguous()
    xs = (xs - data_mean)/data_std
    xs = rearrange(xs, "b t c ... -> t b c ...").contiguous().float()
    print(xs.shape)


    curr_frame = 0
    n_context_frames = 2
    chunk_size = 1
    xs_pred = xs[:n_context_frames].clone()
    curr_frame += n_context_frames
    clip_noise = 6
    timesteps = 1000
    stabilization_level = 15

    pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
    while curr_frame < n_frames:
        horizon = min(n_frames - curr_frame, chunk_size)
        assert horizon <= n_frames, "horizon exceeds the number of tokens."

        height = args.ddim_steps + int((horizon - 1) * args.ddim_steps) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = args.ddim_steps + int(t * args.ddim_steps) - m

        scheduling_matrix = np.clip(scheduling_matrix, 0, args.ddim_steps)

        chunk = torch.randn((horizon, batch_size, 3, 64, 64), device=device)
        chunk = torch.clamp(chunk, -clip_noise, clip_noise)
        xs_pred = torch.cat([xs_pred, chunk], 0)

        # sliding window: only input the last n_frames frames
        start_frame = max(0, curr_frame + horizon - n_frames)

        pbar.set_postfix(
            {
                "start": start_frame,
                "end": curr_frame + horizon,
            }
        )

        for m in range(scheduling_matrix.shape[0] - 1):
            from_noise_levels = np.concatenate((np.zeros((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                :, None
            ].repeat(batch_size, axis=1)
            to_noise_levels = np.concatenate(
                (
                    np.zeros((curr_frame,), dtype=np.int64),
                    scheduling_matrix[m + 1],
                )
            )[
                :, None
            ].repeat(batch_size, axis=1)

            from_noise_levels = torch.from_numpy(from_noise_levels).to(device)
            to_noise_levels = torch.from_numpy(to_noise_levels).to(device)

            # update xs_pred by DDIM or DDPM sampling
            # input frames within the sliding window
            x = xs_pred[start_frame:]
            external_cond = conditions[start_frame : curr_frame + horizon]
            curr_noise_level = from_noise_levels[start_frame:]
            next_noise_level = to_noise_levels[start_frame:]

            real_steps = torch.linspace(-1, timesteps - 1, steps=args.ddim_steps + 1, device=x.device).long()

            # convert noise levels (0 ~ sampling_timesteps) to real noise levels (-1 ~ timesteps - 1)
            curr_noise_level = real_steps[curr_noise_level]
            next_noise_level = real_steps[next_noise_level]

            clipped_curr_noise_level = torch.where(
                curr_noise_level < 0,
                torch.full_like(curr_noise_level, stabilization_level - 1, dtype=torch.long),
                curr_noise_level,
            )

            # treating as stabilization would require us to scale with sqrt of alpha_cum
            orig_x = x.clone().detach()
            noise = torch.zeros_like(x)
            scaled_context =  (
                extract(sqrt_alphas_cumprod, clipped_curr_noise_level, x.shape) * x
                + extract(sqrt_one_minus_alphas_cumprod, clipped_curr_noise_level, x.shape) * noise
            )
            x = torch.where(rearrange(curr_noise_level < 0, f"... -> ...{' 1' * 3}"), scaled_context, orig_x)

            alpha_next = torch.where(
                next_noise_level < 0,
                torch.ones_like(next_noise_level),
                alphas_cumprod[next_noise_level],
            )
            c = (1 - alpha_next).sqrt()

            alpha_next = rearrange(alpha_next, f"... -> ...{' 1' * 3}")
            c = rearrange(c, f"... -> ...{' 1' * 3}")

            model_pred = model(
                x=rearrange(x, "t b ... -> b t ..."),
                t=rearrange(clipped_curr_noise_level, "t b -> b t"),
                external_cond=rearrange(external_cond, "t b ... -> b t ..."),
            )
            model_pred = rearrange(model_pred, "b t ... -> t b ...")

            x_start = model_pred
            pred_noise = (extract(sqrt_recip_alphas_cumprod, clipped_curr_noise_level, x.shape) * x - x_start) / extract(
                sqrt_recipm1_alphas_cumprod, clipped_curr_noise_level, x.shape
            )

            x_pred = x_start * alpha_next.sqrt() + pred_noise * c

            # only update frames where the noise level decreases
            mask = curr_noise_level == next_noise_level
            x_pred = torch.where(
                rearrange(mask, f"... -> ...{' 1' * 3}"),
                orig_x,
                x_pred,
            )
            xs_pred[start_frame:] = x_pred

        curr_frame += horizon
        pbar.update(horizon)
    print(xs_pred.shape)
    xs_pred = xs_pred.reshape(1, -1, 3, 64, 64)
    x = xs_pred * data_std + data_mean
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    x = x[0].cpu()
    os.makedirs(args.output_path, exist_ok=True)
    for i in range(x.shape[0]):
        write_png(x[i], os.path.join(args.output_path, f"{i:04d}.png"))
    x = x.permute(0, 2, 3, 1).numpy()
    write_video(os.path.join(args.output_path, "video.mp4"), x, fps=args.fps)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="outputs/2024-12-17/11-28-33/checkpoints/epoch=1-step=150000.ckpt",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=240,
    )
    parse.add_argument(
        "--prompt-path",
        type=str,
        help="Path to image or video to condition generation on.",
        default="data/minecraft_video/minecraft/validation/000720.mp4",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=4,
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        help="If the prompt is a video, how many frames to condition on.",
        default=1,
    )
    parse.add_argument(
        "--output-path",
        type=str,
        help="Path where generated video should be saved.",
        default="outputs/video/self-240",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=100)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
