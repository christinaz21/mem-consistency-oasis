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
from torchvision.io import write_video
from train_oasis.utils import sigmoid_beta_schedule
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision import transforms

assert torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
print(f"using device: {device}")


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # load DiT checkpoint
    model = DiT_models["dit_small"]()
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
    video = video.permute(1, 2, 3, 0).numpy()[args.video_offset:args.video_offset+20]
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
    x = x.permute(0, 2, 3, 1).numpy()
    write_video(args.output_path, x, fps=args.fps)
    print(f"generation saved to {args.output_path}.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument(
        "--oasis-ckpt",
        type=str,
        help="Path to Oasis DiT checkpoint.",
        default="outputs/2024-12-17/11-28-33/checkpoints/epoch=1-step=230000.ckpt",
    )
    parse.add_argument(
        "--num-frames",
        type=int,
        help="How many frames should the output be?",
        default=120,
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
        default=10,
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
        default="outputs/video/self-120.mp4",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )
    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=40)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
