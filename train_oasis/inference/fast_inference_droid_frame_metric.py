"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import sys
import os
import tempfile
import shutil
import inspect
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dir_path)
_LOCAL_DROID_SLAM = os.path.join(os.path.dirname(os.path.realpath(__file__)), "droid_slam")
_WORLDSCORE_DROID_SLAM = "/u/cz5047/videogen/WorldScore/worldscore/benchmark/metrics/third_party/droid_slam"
if os.path.isdir(_WORLDSCORE_DROID_SLAM):
    # Prefer the same DROID implementation as run_reprojection_error_plots.py.
    sys.path.insert(0, _WORLDSCORE_DROID_SLAM)
else:
    sys.path.insert(0, _LOCAL_DROID_SLAM)

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
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from droid import Droid
import numpy as np
import cv2

assert torch.cuda.is_available()
device = torch.device("cuda")
print(f"using device: {device}")


def paths_from_eval_metadata(info):
    if "video_path" in info and "action_path" in info:
        save_rel = info.get("save_relative_path", os.path.basename(info["video_path"]))
        return info["video_path"], info["action_path"], save_rel
    if "file" in info:
        actions_path = info["file"]
        video_path = os.path.splitext(actions_path)[0] + ".mp4"
        save_rel = os.path.basename(video_path)
        return video_path, actions_path, save_rel
    raise KeyError("metadata item needs 'video_path'+'action_path', or 'file' (action .npy path)")


def image_stream(image_list, stride, calib):
    fx, fy, cx, cy = calib
    image_list = image_list[::stride]
    for t, imfile in enumerate(image_list):
        image = cv2.imread(imfile)
        if image is None:
            continue
        h0, w0, _ = image.shape
        scale = np.sqrt((512 * 512) / float(h0 * w0))
        h1 = int(h0 * scale)
        w1 = int(w0 * scale)
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy], dtype=torch.float32)
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        yield t, image[None], intrinsics


def write_video_frames_to_dir(video_thwc_u8, out_dir, resize_long_side=None):
    os.makedirs(out_dir, exist_ok=True)
    frame_paths = []
    for t in range(video_thwc_u8.shape[0]):
        rgb = video_thwc_u8[t]
        if resize_long_side is not None:
            h, w = rgb.shape[:2]
            long_side = max(h, w)
            if long_side > resize_long_side:
                scale = resize_long_side / float(long_side)
                nh, nw = int(round(h * scale)), int(round(w * scale))
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        bgr = rgb[..., ::-1]
        path = os.path.join(out_dir, f"{t:06d}.png")
        if cv2.imwrite(path, bgr):
            frame_paths.append(path)
    return frame_paths


class DroidFrameErrorScorer:
    """
    Frame-wise DROID reprojection metric.
    - iter_errors[t] is mean finite reprojection error emitted at frame t (or NaN if unavailable).
    - reward can either be:
      (1) -mean(iter_errors), or
      (2) -mean(abs(iter_errors - gt_iter_errors)).
    """

    def __init__(
        self,
        weights_path,
        calib,
        stride=1,
        buffer=512,
        filter_thresh=0.01,
        warmup=8,
        backend_every=10,
        backend_steps=12,
        backend_steps_no_ba=1,
        upsample=False,
        quiet=True,
        resize_long_side=256,
        max_frames=200,
    ):
        self.weights_path = weights_path
        self.calib = tuple(float(v) for v in calib)
        self.stride = int(stride)
        self.buffer = int(buffer)
        self.filter_thresh = float(filter_thresh)
        self.warmup = int(warmup)
        self.backend_every = int(backend_every)
        self.backend_steps = int(backend_steps)
        self.backend_steps_no_ba = int(backend_steps_no_ba)
        self.upsample = bool(upsample)
        self.quiet = bool(quiet)
        self.resize_long_side = resize_long_side
        self.max_frames = int(max_frames)

        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    def _new_droid_args(self):
        return argparse.Namespace(
            t0=0,
            stride=self.stride,
            weights=self.weights_path,
            buffer=self.buffer,
            beta=0.3,
            filter_thresh=self.filter_thresh,
            warmup=self.warmup,
            keyframe_thresh=4.0,
            frontend_thresh=16.0,
            frontend_window=25,
            frontend_radius=2,
            frontend_nms=1,
            backend_thresh=22.0,
            backend_radius=2,
            backend_nms=3,
            backend_every=self.backend_every,
            backend_steps=self.backend_steps,
            backend_steps_no_ba=self.backend_steps_no_ba,
            upsample=self.upsample,
            stereo=False,
            calib=list(self.calib),
        )

    @staticmethod
    def _track_with_compat(droid, t, image, intrinsics):
        sig = inspect.signature(droid.track)
        if "update" in sig.parameters:
            return droid.track(t, image, intrinsics=intrinsics, update=True)
        return droid.track(t, image, intrinsics=intrinsics)

    def frame_errors_from_frame_paths(self, frame_paths):
        if len(frame_paths) == 0:
            return None
        if len(frame_paths) > self.max_frames:
            frame_paths = frame_paths[: self.max_frames]

        droid_args = self._new_droid_args()
        droid = None
        stream = image_stream(frame_paths, droid_args.stride, droid_args.calib)
        iterator = tqdm(stream, disable=self.quiet)
        iter_errors = np.full((len(frame_paths),), np.nan, dtype=np.float32)

        def _run():
            nonlocal droid
            for (t, image, intrinsics) in iterator:
                if droid is None:
                    droid_args.image_size = [image.shape[2], image.shape[3]]
                    droid = Droid(droid_args)
                errors = self._track_with_compat(droid, t, image, intrinsics)
                if errors is None:
                    continue
                errors = errors[torch.isfinite(errors)]
                if errors.numel() > 0:
                    iter_errors[t] = float(errors.mean().item())
            stream2 = image_stream(frame_paths, droid_args.stride, droid_args.calib)
            droid.terminate(stream2)

        if self.quiet:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _run()
        else:
            _run()
        return iter_errors

    def frame_errors_from_video(self, video_thwc):
        v = torch.clamp(video_thwc, 0, 1)
        if v.is_cuda:
            v = v.detach().cpu()
        v = (v * 255).byte().numpy()
        if v.shape[0] > self.max_frames:
            v = v[: self.max_frames]
        tmpdir = tempfile.mkdtemp(prefix="droid_frame_metric_")
        try:
            frame_paths = write_video_frames_to_dir(v, tmpdir, resize_long_side=self.resize_long_side)
            return self.frame_errors_from_frame_paths(frame_paths)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def score_from_errors(iter_errors, gt_iter_errors=None, mode="match_gt_l1"):
        if iter_errors is None:
            return -1e9
        if mode == "gen_mean" or gt_iter_errors is None:
            valid = np.isfinite(iter_errors)
            if not np.any(valid):
                return -1e9
            return -float(np.mean(iter_errors[valid]))
        min_len = min(len(iter_errors), len(gt_iter_errors))
        if min_len <= 0:
            return -1e9
        x = iter_errors[:min_len]
        y = gt_iter_errors[:min_len]
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            return -1e9
        return -float(np.mean(np.abs(x[valid] - y[valid])))

    @staticmethod
    def score_at_frame(iter_errors, frame_idx, gt_iter_errors=None, mode="match_gt_l1"):
        if iter_errors is None or len(iter_errors) == 0:
            return -1e9
        idx = min(frame_idx, len(iter_errors) - 1)
        x = iter_errors[idx]
        if not np.isfinite(x):
            return -1e9
        if mode == "gen_mean" or gt_iter_errors is None:
            return -float(x)
        if len(gt_iter_errors) == 0:
            return -1e9
        idy = min(frame_idx, len(gt_iter_errors) - 1)
        y = gt_iter_errors[idy]
        if not np.isfinite(y):
            return -1e9
        return -float(abs(x - y))


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
                f"Model expects {model.external_cond.in_features} action dims per frame; got --action-dim {args.action_dim}."
            )

    max_frame = model.max_frames
    n_prompt_frames = args.n_prompt_frames
    total_frames = args.num_frames
    max_noise_level = 1000
    ddim_noise_steps = args.ddim_steps
    assert ddim_noise_steps % (max_frame - 1) == 0, "ddim_noise_steps must be divisible by max_frame-1"
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

    scorer = DroidFrameErrorScorer(
        weights_path=args.droid_weights,
        calib=tuple(float(v) for v in args.droid_calib.split(",")),
        stride=args.droid_stride,
        buffer=args.droid_buffer,
        filter_thresh=args.droid_filter_thresh,
        warmup=args.droid_warmup,
        backend_every=args.droid_backend_every,
        backend_steps=args.droid_backend_steps,
        backend_steps_no_ba=args.droid_backend_steps_no_ba,
        upsample=args.droid_upsample,
        quiet=not args.verbose_droid,
        resize_long_side=args.droid_resize,
        max_frames=args.droid_max_frames,
    )

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
        gt_frame_errors = []

        for info in batch_items:
            prompt_path, actions_path, save_rel = paths_from_eval_metadata(info)
            save_path = os.path.join(args.save_dir, save_rel)

            x_prompt = load_prompt(
                prompt_path,
                video_offset=args.video_offset,
                n_prompt_frames=args.n_prompt_frames,
            )
            actions = load_actions(actions_path, action_offset=args.video_offset)[:, :total_frames, : args.action_dim]

            prompts.append(x_prompt)
            actions_list.append(actions)
            save_paths.append(save_path)

            gt_prompt = load_prompt(prompt_path, video_offset=args.video_offset, n_prompt_frames=total_frames)
            gt_prompt = gt_prompt[0].permute(0, 2, 3, 1).contiguous()
            gt_errors = scorer.frame_errors_from_video(gt_prompt)
            gt_frame_errors.append(gt_errors)

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

        def decode_to_video(latents):
            if args.vae_ckpt:
                x_out = rearrange(latents, "b t c h w -> (b t) (h w) c")
                with torch.no_grad():
                    x_out = vae.decode(x_out / scaling_factor)
                x_out = rearrange(x_out, "(b t) c h w -> b t h w c", t=latents.shape[1])
            else:
                x_out = rearrange(latents, "b t c h w -> b t h w c")
            x_out = x_out * 0.5 + 0.5
            return torch.clamp(x_out, 0, 1)

        def append_one_frame(x_state, i, seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            x_next = x_state.clone()
            chunk = torch.randn((B, 1, *x_next.shape[-3:]), device=device)
            chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
            x_next = torch.cat([x_next, chunk], dim=1)
            start_frame = max(0, i + 1 - model.max_frames)
            horizon = min(max_frame, i + 1)
            for noise_idx in reversed(range(1, ddim_one_step + 1)):
                t = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                t[:, 1:] = noise_level_matrix[noise_idx, max_frame - horizon :]
                t_next = torch.full((B, horizon), stabilization_level - 1, dtype=torch.long, device=device)
                t_next[:, 1:] = noise_level_matrix[noise_idx - 1, max_frame - horizon :]
                t_next = torch.where(t_next < 0, t, t_next)

                x_curr = x_next.clone()
                x_curr = x_curr[:, start_frame:]
                with torch.no_grad():
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])
                x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1 and horizon == max_frame:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x_next[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]
            return x_next

        x_work = x.clone()
        frame_choice_history = []
        bar = tqdm(range(n_prompt_frames, total_frames), desc="Frame-wise BoN")
        for i in bar:
            best_state = None
            best_idx = 0
            best_scores = [-1e9] * B
            cand_states = []
            cand_scores = []

            for p in range(args.num_particles):
                seed = args.base_seed + p + 1000 * i
                x_cand = append_one_frame(x_work, i, seed=seed)
                x_dec = decode_to_video(x_cand)

                per_sample_scores = []
                for b in range(B):
                    prefix = x_dec[b, : i + 1]
                    iter_errors = scorer.frame_errors_from_video(prefix)
                    gt_errors = gt_frame_errors[b] if args.frame_score_mode == "match_gt_l1" else None
                    score = scorer.score_at_frame(iter_errors, i, gt_iter_errors=gt_errors, mode=args.frame_score_mode)
                    per_sample_scores.append(float(score))
                cand_states.append(x_cand)
                cand_scores.append(per_sample_scores)

            # batch-size is typically 1; for B>1 use mean score to choose one shared trajectory
            mean_scores = [float(np.mean(s)) for s in cand_scores]
            best_idx = int(np.argmax(mean_scores))
            best_state = cand_states[best_idx]
            x_work = best_state
            frame_choice_history.append((i, best_idx, mean_scores))
            bar.set_postfix({"frame": i, "best_p": best_idx, "score": f"{mean_scores[best_idx]:.4f}"})

        # tail refinement on the selected trajectory
        x = x_work
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
                x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()
                alpha_next = alphas_cumprod[t_next]
                alpha_next[:, :1] = torch.ones_like(alpha_next[:, :1])
                if noise_idx == 1:
                    alpha_next[:, 1:2] = torch.ones_like(alpha_next[:, 1:2])
                x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()
                x[:, start_frame + 1 :] = x_pred[:, -(horizon - 1) :]

        x_final_all = decode_to_video(x).detach().cpu()
        for b, save_path in enumerate(save_paths):
            x_final = x_final_all[b]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            write_video(save_path, (x_final * 255).byte().numpy(), fps=args.fps)
            print(f"Saved {save_path} (frame-wise BoN, particles={args.num_particles})")
        print("Frame choice history (first 10):")
        for i, best_idx, mean_scores in frame_choice_history[:10]:
            print(f"frame={i} best_particle={best_idx} mean_scores={mean_scores}")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--oasis-ckpt", type=str, default="/home/tc0786/Project/train-oasis/outputs/2025-05-07/03-28-14/checkpoints/epoch=2-step=13000.ckpt")
    parse.add_argument("--model-name", type=str, default="dit_easy")
    parse.add_argument("--action-dim", type=int, default=4)
    parse.add_argument("--vae-ckpt", type=str, default="models/oasis500m/vit-l-20.safetensors")
    parse.add_argument("--num-frames", type=int, default=300)
    parse.add_argument("--metadata-path", type=str, required=True)
    parse.add_argument("--save-dir", type=str, required=True)
    parse.add_argument("--video-offset", type=int, default=0)
    parse.add_argument("--split", type=str, default="memory", choices=["memory", "random", "all", "rotate_wait"])
    parse.add_argument("--n-prompt-frames", type=int, default=1)
    parse.add_argument("--fps", type=int, default=20)
    parse.add_argument("--ddim-steps", type=int, default=18)
    parse.add_argument("--num_particles", type=int, default=4)
    parse.add_argument("--base-seed", type=int, default=0)

    parse.add_argument("--droid-weights", type=str, default="/u/cz5047/videogen/data/models/droid_models/droid.pth")
    parse.add_argument("--droid-calib", type=str, default="500,500,256,256")
    parse.add_argument("--droid-stride", type=int, default=1)
    parse.add_argument("--droid-max-frames", type=int, default=200)
    parse.add_argument("--droid-resize", type=int, default=256)
    parse.add_argument("--droid-buffer", type=int, default=512)
    parse.add_argument("--droid-filter-thresh", type=float, default=0.01)
    parse.add_argument("--droid-upsample", action="store_true")
    parse.add_argument("--droid-warmup", type=int, default=8)
    parse.add_argument("--droid-backend-every", type=int, default=10)
    parse.add_argument("--droid-backend-steps", type=int, default=12)
    parse.add_argument("--droid-backend-steps-no-ba", type=int, default=1)
    parse.add_argument("--verbose-droid", action="store_true")

    parse.add_argument(
        "--frame-score-mode",
        type=str,
        default="match_gt_l1",
        choices=["match_gt_l1", "gen_mean"],
        help="match_gt_l1: maximize similarity to GT frame-wise reprojection curve; gen_mean: maximize negative mean generated reprojection error.",
    )
    parse.add_argument("--batch-size", type=int, default=1)

    args = parse.parse_args()
    print("inference args:")
    pprint(vars(args))
    main(args)
