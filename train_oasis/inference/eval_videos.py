import os
import csv
import json
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

try:
    import lpips  # type: ignore[import-not-found]
except ImportError:
    lpips = None

try:
    from worldscore import DroidReprojectionScorer
except ImportError:
    DroidReprojectionScorer = None

world_eval_root = Path(__file__).resolve().parents[3] / "world-eval-latest"
if world_eval_root.is_dir():
    world_eval_root_str = str(world_eval_root)
    if world_eval_root_str not in sys.path:
        sys.path.insert(0, world_eval_root_str)

try:
    from utils.spatial_distance import (
        ReconstructionConfig,
        VGGTReconstructor,
        compute_spatial_distance_for_videos,
    )  # pyright: ignore[reportMissingImports]
except ImportError:
    ReconstructionConfig = None
    VGGTReconstructor = None
    compute_spatial_distance_for_videos = None


def load_video(path: str) -> torch.Tensor:
    """
    Load video as float tensor in [0, 1] with shape [T, C, H, W].
    """
    video, _, _ = torchvision.io.read_video(path, pts_unit="sec")
    video = video.permute(0, 3, 1, 2).float() / 255.0
    return video


def gaussian_window(window_size: int, sigma: float, channel: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_batch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5):
    """
    x, y: [N, C, H, W] in [0, 1]
    Returns scalar tensor mean SSIM over batch.
    """
    device = x.device
    channel = x.size(1)
    window = gaussian_window(window_size, sigma, channel, device)
    padding = window_size // 2

    mu_x = F.conv2d(x, window, padding=padding, groups=channel)
    mu_y = F.conv2d(y, window, padding=padding, groups=channel)

    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12
    )
    return ssim_map.mean()


def psnr_batch(x: torch.Tensor, y: torch.Tensor):
    """
    x, y: [N, C, H, W] in [0, 1]
    Returns scalar tensor mean PSNR over batch.
    """
    mse = F.mse_loss(x, y, reduction="none")
    mse = mse.flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / (mse + 1e-12))
    return psnr.mean()


def center_crop_to_match(a: torch.Tensor, b: torch.Tensor):
    """
    Center crop both videos to smallest shared H, W.
    a, b: [T, C, H, W]
    """
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])

    def crop(x):
        _, _, H, W = x.shape
        top = (H - h) // 2
        left = (W - w) // 2
        return x[:, :, top:top + h, left:left + w]

    return crop(a), crop(b)


def maybe_resize(video: torch.Tensor, size):
    """
    video: [T, C, H, W]
    size: int or None
    """
    if size is None:
        return video
    return F.interpolate(video, size=(size, size), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate_video_pair(
    gen_path: str,
    ref_path: str,
    device: torch.device,
    lpips_model=None,
    resize: int = None,
    max_frames: int = None,
    batch_size: int = 16,
    skip_first_n_frames: int = 0,
):
    gen = load_video(gen_path)
    ref = load_video(ref_path)

    T = min(len(gen), len(ref))
    if max_frames is not None:
        T = min(T, max_frames)

    if T <= skip_first_n_frames:
        raise ValueError(
            f"Not enough overlapping frames after skipping {skip_first_n_frames} "
            f"for:\n{gen_path}\n{ref_path}"
        )

    gen = gen[skip_first_n_frames:T]
    ref = ref[skip_first_n_frames:T]

    gen, ref = center_crop_to_match(gen, ref)

    if resize is not None:
        gen = maybe_resize(gen, resize)
        ref = maybe_resize(ref, resize)

    gen = gen.to(device)
    ref = ref.to(device)

    ssim_vals = []
    psnr_vals = []
    lpips_vals = []

    for i in range(0, len(gen), batch_size):
        g = gen[i:i + batch_size]
        r = ref[i:i + batch_size]

        ssim_vals.append(ssim_batch(g, r).item())
        psnr_vals.append(psnr_batch(g, r).item())

        if lpips_model is not None:
            g_lp = g * 2.0 - 1.0
            r_lp = r * 2.0 - 1.0
            lp = lpips_model(g_lp, r_lp)
            lpips_vals.append(lp.mean().item())

    result = {
        "num_frames": len(gen),
        "ssim": float(sum(ssim_vals) / len(ssim_vals)),
        "psnr": float(sum(psnr_vals) / len(psnr_vals)),
        "lpips": float(sum(lpips_vals) / len(lpips_vals)) if lpips_model is not None else None,
    }
    return result


@torch.no_grad()
def evaluate_droid_score(
    gen_path: str,
    scorer,
    max_frames: int = None,
    skip_first_n_frames: int = 0,
):
    """
    Compute DROID reprojection score on generated video only.
    """
    video = load_video(gen_path)  # [T, C, H, W] in [0,1]

    if skip_first_n_frames > 0:
        if len(video) <= skip_first_n_frames:
            raise ValueError(f"Video too short after skipping frames: {gen_path}")
        video = video[skip_first_n_frames:]

    if max_frames is not None:
        video = video[:max_frames]

    # scorer.reward_from_video expects [T, H, W, C]
    video = video.permute(0, 2, 3, 1).contiguous()
    score = scorer.reward_from_video(video)
    return float(score)


@torch.no_grad()
def evaluate_spatial_distance(
    gen_path: str,
    ref_path: str,
    reconstructor,
    chunk_size: int = 4096,
):
    """
    Compute spatial distance between generated and reference videos.

    Returns a tuple: (mean_distance, max_distance).
    """
    result = compute_spatial_distance_for_videos(
        generated_video_path=gen_path,
        ground_truth_video_path=ref_path,
        reconstructor=reconstructor,
        chunk_size=chunk_size,
        point_cloud_example_count=0,
    )
    return (
        float(result.spatial_distance["mean"]),
        float(result.spatial_distance["max"]),
    )


def collect_video_files(root: str, exts=(".mp4", ".avi", ".mov", ".mkv")):
    root = Path(root)
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def default_match_fn(gen_file: Path, gen_root: Path, ref_root: Path) -> Path:
    rel = gen_file.relative_to(gen_root)
    return ref_root / rel


def av_ref_path_from_item(info: dict) -> Path:
    """Ground-truth video path from a Cosmos / AV eval JSON item."""
    if "video_path" in info:
        return Path(info["video_path"])
    if "file" not in info:
        raise KeyError("eval item must include 'file' (.npy or .mp4) or 'video_path'")
    p = Path(info["file"])
    if p.suffix.lower() == ".npy":
        return p.with_suffix(".mp4")
    return p


def resolve_generated_path(gen_root: Path, basename: str) -> Path | None:
    """
    Prefer gen_root / basename; if missing, use a unique rglob match under gen_root.
    """
    direct = gen_root / basename
    if direct.is_file():
        return direct
    matches = sorted(gen_root.rglob(basename))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple generated files named {basename!r}; using {matches[0]}")
    return matches[0]


def load_av_eval_split(metadata_path: str, split: str) -> list[dict]:
    with open(metadata_path, "r") as f:
        meta = json.load(f)
    if split == "all":
        return list(meta.get("memory", [])) + list(meta.get("random", []))
    if split not in meta:
        raise KeyError(
            f"Split {split!r} not in {metadata_path} (available: {list(meta.keys())})"
        )
    return list(meta[split])


def ref_path_for_csv(ref_file: Path, ref_root: Path | None) -> str:
    if ref_root is not None:
        try:
            return str(ref_file.resolve().relative_to(ref_root.resolve()))
        except ValueError:
            pass
    return str(ref_file)


def mean_or_none(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    if len(vals) == 0:
        return None
    return float(sum(vals) / len(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=str, required=True,
                        help="Root directory containing generated videos.")
    parser.add_argument(
        "--reference-root",
        type=str,
        default=None,
        help="Ground-truth root for relative path mirroring (Minecraft-style). "
        "Optional with --eval-metadata; if set, reference paths in CSV are relativized.",
    )
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save CSV and JSON results.")

    parser.add_argument(
        "--eval-metadata",
        type=str,
        default=None,
        help="Cosmos / AV eval JSON (e.g. eval_paths_sunny.json). "
        "Each item uses 'file' (.npy) or 'video_path'; GT video is the .mp4; "
        "generated clip is generated-root / <basename>.mp4 (or nested rglob).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="memory",
        choices=["memory", "random", "all"],
        help="Which list to read from --eval-metadata (ignored without it).",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resize", type=int, default=None,
                        help="Optional resize to size x size before SSIM/PSNR/LPIPS.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only evaluate first N frames.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-first-n-frames", type=int, default=0,
                        help="Skip first N frames for all metrics.")
    parser.add_argument("--lpips-net", type=str, default="alex",
                        choices=["alex", "vgg", "squeeze"])

    # DROID args
    parser.add_argument("--compute-droid", action="store_true",
                        help="Enable DROID reprojection score.")
    parser.add_argument("--droid-weights", type=str, default="/u/cz5047/videogen/data/models/droid_models/droid.pth")
    parser.add_argument("--droid-calib", type=str, default="500,500,256,256")
    parser.add_argument("--droid-stride", type=int, default=2)
    parser.add_argument("--droid-max-frames", type=int, default=200)
    parser.add_argument("--droid-resize", type=int, default=256)
    parser.add_argument("--droid-buffer", type=int, default=512)
    parser.add_argument("--droid-filter-thresh", type=float, default=0.01)
    parser.add_argument("--droid-upsample", action="store_true")
    # Spatial distance args
    parser.add_argument(
        "--compute-spatial-distance",
        action="store_true",
        help="Enable spatial-distance score using world-eval-latest/utils/spatial_distance.py",
    )
    parser.add_argument(
        "--sd-device",
        type=str,
        default=None,
        help="Device for VGGT reconstructor (e.g. cuda:0). Default follows spatial_distance.py behavior.",
    )
    parser.add_argument("--sd-frame-stride", type=int, default=1, help="Frame stride for spatial distance decoding.")
    parser.add_argument(
        "--sd-max-frames",
        type=int,
        default=None,
        help="Optional cap on decoded frames for spatial distance (uniformly sampled).",
    )
    parser.add_argument(
        "--sd-vggt-batch-size",
        type=int,
        default=None,
        help="VGGT frames per forward pass; None means single multi-view pass.",
    )
    parser.add_argument(
        "--sd-confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum VGGT confidence threshold for points.",
    )
    parser.add_argument("--sd-max-points-per-frame", type=int, default=200000)
    parser.add_argument("--sd-max-merged-points", type=int, default=200000)
    parser.add_argument("--sd-chunk-size", type=int, default=4096, help="Chunk size for Chamfer fallback.")
    parser.add_argument("--sd-seed", type=int, default=0)
    parser.add_argument(
        "--sd-vggt-repo-dir",
        type=str,
        default=str(world_eval_root / "vggt"),
        help="Path to local VGGT repository.",
    )
    parser.add_argument(
        "--sd-model-path",
        type=str,
        default=str(world_eval_root / "vggt" / "model.pt"),
        help="Path to local VGGT checkpoint.",
    )

    args = parser.parse_args()
    if args.eval_metadata is None and args.reference_root is None:
        parser.error("--reference-root is required unless --eval-metadata is set")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if lpips is None:
        print("Warning: lpips package not found. LPIPS will be skipped.")
        lpips_model = None
    else:
        lpips_model = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    droid_scorer = None
    if args.compute_droid:
        if DroidReprojectionScorer is None:
            raise ImportError(
                "Requested --compute-droid but could not import DroidReprojectionScorer "
                "from worldscore."
            )

        droid_scorer = DroidReprojectionScorer(
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

    sd_reconstructor = None
    if args.compute_spatial_distance:
        if (
            VGGTReconstructor is None
            or ReconstructionConfig is None
            or compute_spatial_distance_for_videos is None
        ):
            raise ImportError(
                "Requested --compute-spatial-distance but failed to import "
                "world-eval-latest utils.spatial_distance. Ensure dependencies "
                "and PYTHONPATH are available."
            )
        sd_reconstructor = VGGTReconstructor(
            repo_dir=Path(args.sd_vggt_repo_dir),
            model_path=Path(args.sd_model_path),
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

    gen_root = Path(args.generated_root)
    ref_root = Path(args.reference_root).resolve() if args.reference_root else None

    rows = []
    missing_refs = []
    missing_generated = []
    failed_files = []

    if args.eval_metadata:
        items = load_av_eval_split(args.eval_metadata, args.split)
        print(
            f"AV / Cosmos eval mode: {len(items)} entries from {args.eval_metadata} "
            f"(split={args.split!r})"
        )
        for info in tqdm(items):
            try:
                ref_file = av_ref_path_from_item(info)
            except KeyError as e:
                failed_files.append({"video": str(info), "error": str(e)})
                print(f"Bad metadata item {info!r}: {e}")
                continue

            gen_file = resolve_generated_path(gen_root, ref_file.name)
            if gen_file is None:
                missing_generated.append(str(gen_root / ref_file.name))
                print(f"Missing generated video for GT {ref_file.name}")
                continue
            if not ref_file.is_file():
                missing_refs.append(str(ref_file))
                print(f"Missing reference {ref_file}")
                continue

            try:
                pair_metrics = evaluate_video_pair(
                    gen_path=str(gen_file),
                    ref_path=str(ref_file),
                    device=device,
                    lpips_model=lpips_model,
                    resize=args.resize,
                    max_frames=args.max_frames,
                    batch_size=args.batch_size,
                    skip_first_n_frames=args.skip_first_n_frames,
                )

                droid_score = None
                if droid_scorer is not None:
                    droid_score = evaluate_droid_score(
                        gen_path=str(gen_file),
                        scorer=droid_scorer,
                        max_frames=args.max_frames,
                        skip_first_n_frames=args.skip_first_n_frames,
                    )
                sd_mean = None
                sd_max = None
                if sd_reconstructor is not None:
                    sd_mean, sd_max = evaluate_spatial_distance(
                        gen_path=str(gen_file),
                        ref_path=str(ref_file),
                        reconstructor=sd_reconstructor,
                        chunk_size=args.sd_chunk_size,
                    )

                try:
                    video_rel = str(gen_file.resolve().relative_to(gen_root.resolve()))
                except ValueError:
                    video_rel = str(gen_file)

                rows.append(
                    {
                        "video": video_rel,
                        "reference": ref_path_for_csv(ref_file, ref_root),
                        "num_frames": pair_metrics["num_frames"],
                        "ssim": pair_metrics["ssim"],
                        "psnr": pair_metrics["psnr"],
                        "lpips": pair_metrics["lpips"],
                        "droid_score": droid_score,
                        "spatial_distance_mean": sd_mean,
                        "spatial_distance_max": sd_max,
                    }
                )

            except Exception as e:
                failed_files.append({"video": str(gen_file), "error": str(e)})
                print(f"Failed on {gen_file}: {e}")

    else:
        assert ref_root is not None
        gen_files = collect_video_files(gen_root)
        print(f"Found {len(gen_files)} generated videos under {gen_root}")

        for gen_file in tqdm(gen_files):
            ref_file = default_match_fn(gen_file, gen_root, ref_root)

            if not ref_file.exists():
                missing_refs.append(str(ref_file))
                print(f"Missing reference for {gen_file} -> expected {ref_file}")
                continue

            try:
                pair_metrics = evaluate_video_pair(
                    gen_path=str(gen_file),
                    ref_path=str(ref_file),
                    device=device,
                    lpips_model=lpips_model,
                    resize=args.resize,
                    max_frames=args.max_frames,
                    batch_size=args.batch_size,
                    skip_first_n_frames=args.skip_first_n_frames,
                )

                droid_score = None
                if droid_scorer is not None:
                    droid_score = evaluate_droid_score(
                        gen_path=str(gen_file),
                        scorer=droid_scorer,
                        max_frames=args.max_frames,
                        skip_first_n_frames=args.skip_first_n_frames,
                    )
                sd_mean = None
                sd_max = None
                if sd_reconstructor is not None:
                    sd_mean, sd_max = evaluate_spatial_distance(
                        gen_path=str(gen_file),
                        ref_path=str(ref_file),
                        reconstructor=sd_reconstructor,
                        chunk_size=args.sd_chunk_size,
                    )

                row = {
                    "video": str(gen_file.relative_to(gen_root)),
                    "reference": str(ref_file.relative_to(ref_root)),
                    "num_frames": pair_metrics["num_frames"],
                    "ssim": pair_metrics["ssim"],
                    "psnr": pair_metrics["psnr"],
                    "lpips": pair_metrics["lpips"],
                    "droid_score": droid_score,
                    "spatial_distance_mean": sd_mean,
                    "spatial_distance_max": sd_max,
                }
                rows.append(row)

            except Exception as e:
                failed_files.append({"video": str(gen_file), "error": str(e)})
                print(f"Failed on {gen_file}: {e}")

    csv_path = os.path.join(args.output_dir, "per_video_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "reference",
                "num_frames",
                "ssim",
                "psnr",
                "lpips",
                "droid_score",
                "spatial_distance_mean",
                "spatial_distance_max",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_evaluated_videos": len(rows),
        "num_missing_references": len(missing_refs),
        "num_missing_generated": len(missing_generated),
        "num_failed_files": len(failed_files),
        "mean_ssim": mean_or_none(rows, "ssim"),
        "mean_psnr": mean_or_none(rows, "psnr"),
        "mean_lpips": mean_or_none(rows, "lpips"),
        "mean_droid_score": mean_or_none(rows, "droid_score"),
        "mean_spatial_distance_mean": mean_or_none(rows, "spatial_distance_mean"),
        "mean_spatial_distance_max": mean_or_none(rows, "spatial_distance_max"),
        "missing_references": missing_refs,
        "missing_generated": missing_generated,
        "failed_files": failed_files,
    }

    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Per-video results: {csv_path}")
    print(f"Summary: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()