"""
Oasis inference with best-of-N selection using spatial distance (world-eval-latest).

Same sampling loop as ``fast_inference_droid.py``, but scores each candidate with
``utils.spatial_distance`` (VGGT + one-sided Chamfer), matching:

    bash scripts/run_spatial_distance_single.sh \\
      --generated-video /path/to/gen.mp4 \\
      --ground-truth-video /path/to/gt.mp4 \\
      --device cuda:0 --max-frames 30

References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
import tempfile

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)

world_eval_root = os.path.join(os.path.dirname(dir_path), "world-eval-latest")
if world_eval_root not in sys.path:
    sys.path.insert(0, world_eval_root)

import torch
from train_oasis.model.dit import DiT_models
from train_oasis.model.vae import VAE_models
from torchvision.io import write_video
from train_oasis.utils import sigmoid_beta_schedule, load_actions, load_prompt
from tqdm import tqdm
from einops import rearrange
from torch import autocast
from safetensors.torch import load_model
import argparse
from pprint import pprint
import json
from pathlib import Path
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from utils.spatial_distance import (
    ReconstructionConfig,
    VGGTReconstructor,
    compute_spatial_distance,
)

assert torch.cuda.is_available()
device = torch.device("cuda")
print(f"using device: {device}")


def paths_from_eval_metadata(info):
    """Return (video_path, action_path, save_relative_path) for one metadata JSON item.

    Legacy: ``video_path``, ``action_path``, optional ``save_relative_path``.
    Cosmos-style: ``file`` is the action ``.npy`` path; video is the same path with ``.mp4``.
    """
    if "video_path" in info and "action_path" in info:
        save_rel = info.get("save_relative_path", os.path.basename(info["video_path"]))
        return info["video_path"], info["action_path"], save_rel
    if "file" in info:
        actions_path = info["file"]
        video_path = os.path.splitext(actions_path)[0] + ".mp4"
        save_rel = os.path.basename(video_path)
        return video_path, actions_path, save_rel
    raise KeyError(
        "metadata item needs 'video_path'+'action_path', or 'file' (action .npy path)"
    )


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

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
    if isinstance(model.external_cond, torch.nn.Linear):
        if model.external_cond.in_features != args.action_dim:
            raise ValueError(
                f"Model expects {model.external_cond.in_features} action dims per frame; "
                f"got --action-dim {args.action_dim}. Match --model-name (e.g. cosmos or dit_easy_6) and --action-dim."
            )

    max_frame = model.max_frames
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    assert ddim_noise_steps % (max_frame - 1) == 0, "ddim_noise_steps must be divisible by 9"
    noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1, dtype=torch.long)
    print(noise_range)
    noise_abs_max = 6
    stabilization_level = 15

    vae = None
    scaling_factor = 0.07843137255
    if args.vae_ckpt:
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        print(f"loading ViT-VAE-L/20 from vae-ckpt={os.path.abspath(args.vae_ckpt)}...")
        if args.vae_ckpt.endswith(".pt"):
            vae_ckpt = torch.load(args.vae_ckpt, weights_only=True)
            vae.load_state_dict(vae_ckpt)
        elif args.vae_ckpt.endswith(".safetensors"):
            load_model(vae, args.vae_ckpt)
        vae = vae.to(device).eval()
    betas = sigmoid_beta_schedule(max_noise_level).float().to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    ddim_one_step = ddim_noise_steps // (max_frame - 1)
    noise_level_matrix = torch.zeros((ddim_one_step + 1, (max_frame - 1)), device=device)
    for i in range(max_frame - 1):
        for step in range(ddim_one_step + 1):
            noise_level_matrix[step, i] = noise_range[step + i * ddim_one_step]

    sd_reconstructor = VGGTReconstructor(
        repo_dir=Path(args.vggt_repo_dir),
        model_path=Path(args.vggt_model_path),
        device=args.sd_device,
        config=ReconstructionConfig(
            frame_stride=args.sd_frame_stride,
            max_frames=args.sd_max_frames,
            vggt_batch_size=args.sd_vggt_batch_size,
            confidence_threshold=args.sd_confidence_threshold,
            max_points_per_frame=args.sd_max_points_per_frame,
            max_merged_points=args.sd_max_merged_points,
            seed=args.sd_seed,
        ),
    )

    def spatial_distance_scalar(result) -> float:
        key = args.sd_metric
        return float(result.spatial_distance[key])

    def score_for_argmax(sd_value: float) -> float:
        # Lower distance is better; maximize negated distance (or neg max).
        return -sd_value

    with open(args.metadata_path, "r") as f:
        meta = json.load(f)
    if args.split == "all":
        items = meta["memory"] + meta["random"]
    else:
        items = meta[args.split]

    os.makedirs(args.save_dir, exist_ok=True)

    for start_idx in tqdm(range(0, len(items), args.batch_size), desc="dataset batches"):
        batch_items = items[start_idx : start_idx + args.batch_size]
        prompts = []
        actions_list = []
        save_paths = []
        gt_video_paths = []

        for info in batch_items:
            prompt_path, actions_path, save_rel = paths_from_eval_metadata(info)
            save_path = os.path.join(args.save_dir, save_rel)
            gt_video_paths.append(os.path.abspath(prompt_path))

            x_prompt = load_prompt(
                prompt_path,
                video_offset=args.video_offset,
                n_prompt_frames=n_prompt_frames,
            )
            actions = load_actions(actions_path, action_offset=args.video_offset)[
                :, :total_frames, : args.action_dim
            ]

            prompts.append(x_prompt)
            actions_list.append(actions)
            save_paths.append(save_path)

        x = torch.cat(prompts, dim=0).to(device)
        actions = torch.cat(actions_list, dim=0).to(device)
        B = x.shape[0]
        H, W = x.shape[-2:]
        x = (x - 0.5) / 0.5

        if args.vae_ckpt:
            x_enc = rearrange(x, "b t c h w -> (b t) c h w")
            with torch.no_grad():
                with autocast("cuda", dtype=torch.half):
                    x_enc = vae.encode(x_enc).mean * scaling_factor
            x = rearrange(
                x_enc,
                "(b t) (h w) c -> b t c h w",
                t=n_prompt_frames,
                h=H // vae.patch_size,
                w=W // vae.patch_size,
            )

        x_init = x.clone()

        gt_point_clouds = []
        for b in range(B):
            gt_point_clouds.append(sd_reconstructor.reconstruct_video(gt_video_paths[b]))

        cand_scores = torch.full((args.num_particles, B), -float("inf"), dtype=torch.float32)
        cand_states = []
        for p in range(args.num_particles):
            seed = args.base_seed + p + 1000 * total_frames
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            x = x_init.clone()

            bar = tqdm(range(n_prompt_frames, total_frames + max_frame - 2))
            bar.set_description("Generating frames")
            for i in range(n_prompt_frames, total_frames):
                chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
                chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
                x = torch.cat([x, chunk], dim=1)
                start_frame = max(0, i + 1 - model.max_frames)
                horizon = min(max_frame, i + 1)

                for noise_idx in reversed(range(1, ddim_one_step + 1)):
                    t = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                    t[:, 1:] = noise_level_matrix[noise_idx, max_frame - horizon :]
                    t_next = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                    t_next[:, 1:] = noise_level_matrix[noise_idx - 1, max_frame - horizon :]
                    t_next = torch.where(t_next < 0, t, t_next)

                    x_curr = x.clone()
                    x_curr = x_curr[:, start_frame:]

                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.half):
                            v = model(x_curr, t, actions[:, start_frame : i + 1])

                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
                        1 / alphas_cumprod[t] - 1
                    ).sqrt()

                    alpha_next = alphas_cumprod[t_next]
                    alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                    if noise_idx == 1 and horizon == max_frame:
                        alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]

                bar.update(1)

            for i in range(total_frames, total_frames + max_frame - 2):
                start_frame = max(0, i + 1 - max_frame)
                horizon = min(max_frame, max_frame + total_frames - 1 - i)

                for noise_idx in reversed(range(1, ddim_one_step + 1)):
                    t = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                    t[:, 1:] = noise_level_matrix[noise_idx, : horizon - 1]
                    t_next = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                    t_next[:, 1:] = noise_level_matrix[noise_idx - 1, : horizon - 1]
                    t_next = torch.where(t_next < 0, t, t_next)

                    x_curr = x.clone()
                    x_curr = x_curr[:, start_frame:]

                    with torch.no_grad():
                        with autocast("cuda", dtype=torch.half):
                            v = model(x_curr, t, actions[:, start_frame : i + 1])

                    x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                    x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (
                        1 / alphas_cumprod[t] - 1
                    ).sqrt()

                    alpha_next = alphas_cumprod[t_next]
                    alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                    if noise_idx == 1:
                        alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                    x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                    x[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]
                bar.update(1)
            bar.close()

            x_out = x
            if args.vae_ckpt:
                x_out = rearrange(x_out, "b t c h w -> (b t) (h w) c")
                with torch.no_grad():
                    x_out = vae.decode(x_out / scaling_factor)
                x_out = rearrange(x_out, "(b t) c h w -> b t h w c", t=total_frames)
            x_out = x_out * 0.5 + 0.5
            x_out = torch.clamp(x_out, 0, 1)

            for b in range(B):
                tmp_path = None
                try:
                    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                    os.close(fd)
                    write_video(tmp_path, (x_out[b] * 255).byte().cpu().numpy(), fps=args.fps)
                    gen_pc = sd_reconstructor.reconstruct_video(tmp_path)
                    result = compute_spatial_distance(
                        gen_pc,
                        gt_point_clouds[b],
                        chunk_size=args.sd_chunk_size,
                    )
                    raw = spatial_distance_scalar(result)
                    cand_scores[p, b] = score_for_argmax(raw)
                    print(
                        f"{os.path.basename(save_paths[b])} particle {p}: seed={seed} "
                        f"sd_{args.sd_metric}={raw:.6f} score={cand_scores[p, b].item():.6f}"
                    )
                finally:
                    if tmp_path and os.path.isfile(tmp_path):
                        os.unlink(tmp_path)

            cand_states.append(x_out.detach().cpu())

        best_particles = torch.argmax(cand_scores, dim=0).cpu().tolist()
        for b, save_path in enumerate(save_paths):
            best_idx = int(best_particles[b])
            best_score = float(cand_scores[best_idx, b].item())
            x_final = cand_states[best_idx][b]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            write_video(save_path, (x_final * 255).byte().numpy(), fps=args.fps)
            print(f"Saved {save_path} (best particle={best_idx}, argmax_score={best_score:.6f})")


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
        help="Model name (use cosmos or dit_easy_6 with --action-dim 6 for 6-D action checkpoints).",
        default="dit_easy",
    )
    parse.add_argument(
        "--action-dim",
        type=int,
        default=4,
        help="Number of action dimensions per frame (must match checkpoint external_cond; e.g. 6 for dit_easy_6).",
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
        "--metadata-path",
        type=str,
        required=True,
        help="Path to metadata JSON with memory/random splits.",
    )
    parse.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Root directory for saving outputs.",
    )
    parse.add_argument(
        "--video-offset",
        type=int,
        help="If loading prompt from video, index of frame to start reading from.",
        default=0,
    )
    parse.add_argument(
        "--split",
        type=str,
        default="memory",
        choices=["memory", "random", "all", "rotate_wait"],
        help="Which split to run.",
    )
    parse.add_argument(
        "--n-prompt-frames",
        type=int,
        default=1,
        help="Number of prompt frames to condition on.",
    )
    parse.add_argument(
        "--fps",
        type=int,
        help="What framerate should be used to save the output?",
        default=20,
    )

    parse.add_argument("--ddim-steps", type=int, help="How many DDIM steps?", default=18)
    parse.add_argument("--num_particles", type=int, default=4, help="Number of candidates for best-of-n.")
    parse.add_argument("--base-seed", type=int, default=0, help="Base seed for candidate sampling.")
    parse.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")

    parse.add_argument(
        "--sd-device",
        type=str,
        default="cuda:0",
        help="Device for VGGT / spatial distance (e.g. cuda:0).",
    )
    parse.add_argument(
        "--sd-max-frames",
        type=int,
        default=30,
        help="Max decoded frames per video for VGGT (matches run_spatial_distance_single.sh).",
    )
    parse.add_argument("--sd-frame-stride", type=int, default=1, help="Frame stride for VGGT decoding.")
    parse.add_argument(
        "--sd-metric",
        type=str,
        choices=("mean", "max"),
        default="mean",
        help="Which spatial_distance aggregate to minimize (mean or max).",
    )
    parse.add_argument("--sd-chunk-size", type=int, default=4096, help="Chamfer chunk size.")
    parse.add_argument("--sd-seed", type=int, default=0, help="Seed for VGGT point subsampling.")
    parse.add_argument(
        "--sd-vggt-batch-size",
        type=int,
        default=None,
        help="VGGT batch size (None = one multi-view pass).",
    )
    parse.add_argument(
        "--sd-confidence-threshold",
        type=float,
        default=0.2,
        help="VGGT confidence threshold for keeping points.",
    )
    parse.add_argument(
        "--sd-max-points-per-frame",
        type=int,
        default=200000,
        help="Max points per reconstructed frame.",
    )
    parse.add_argument(
        "--sd-max-merged-points",
        type=int,
        default=200000,
        help="Max points in merged GT cloud.",
    )
    parse.add_argument(
        "--vggt-repo-dir",
        type=str,
        default=str(Path(world_eval_root) / "vggt"),
        help="VGGT repo path (world-eval-latest/vggt).",
    )
    parse.add_argument(
        "--vggt-model-path",
        type=str,
        default=str(Path(world_eval_root) / "vggt" / "model.pt"),
        help="VGGT checkpoint path.",
    )

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
