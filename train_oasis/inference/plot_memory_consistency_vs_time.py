import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PLOT_METRIC_CHOICES = [
    "memory_error",
    "droid_score",
    "spatial_distance_mean",
    "spatial_distance_max",
    "ssim",
    "psnr",
    "lpips",
]


def summary_to_metrics(summary: dict) -> dict:
    return {
        "droid_score": summary.get("mean_droid_score"),
        "spatial_distance_mean": summary.get("mean_spatial_distance_mean"),
        "spatial_distance_max": summary.get("mean_spatial_distance_max"),
        "ssim": summary.get("mean_ssim"),
        "psnr": summary.get("mean_psnr"),
        "lpips": summary.get("mean_lpips"),
    }


def load_existing_metrics(step_output_dir: Path) -> dict | None:
    summary_path = step_output_dir / "summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r") as f:
        summary = json.load(f)
    return summary_to_metrics(summary)


def parse_method_arg(method_arg: str) -> tuple[str, str]:
    if "=" not in method_arg:
        raise ValueError(
            f"Invalid --method {method_arg!r}. Expected format: --method name=/path/to/generated/videos"
        )
    name, root = method_arg.split("=", 1)
    name = name.strip()
    root = root.strip()
    if not name or not root:
        raise ValueError(f"Invalid --method {method_arg!r}. Name/root cannot be empty.")
    return name, root


def run_eval_videos(
    eval_script: Path,
    method_name: str,
    generated_root: str,
    output_dir: Path,
    max_frames: int,
    args: argparse.Namespace,
    step_output_dir: Path,
) -> dict:
    step_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_script),
        "--generated-root",
        generated_root,
        "--output-dir",
        str(step_output_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--max-frames",
        str(max_frames),
        "--skip-first-n-frames",
        str(args.skip_first_n_frames),
    ]

    if args.compute_droid:
        cmd.extend(
            [
                "--compute-droid",
                "--droid-weights",
                args.droid_weights,
                "--droid-calib",
                args.droid_calib,
                "--droid-stride",
                str(args.droid_stride),
                "--droid-max-frames",
                str(args.droid_max_frames),
                "--droid-resize",
                str(args.droid_resize),
                "--droid-buffer",
                str(args.droid_buffer),
                "--droid-filter-thresh",
                str(args.droid_filter_thresh),
            ]
        )
        if args.droid_upsample:
            cmd.append("--droid-upsample")

    if args.compute_spatial_distance:
        cmd.extend(
            [
                "--compute-spatial-distance",
                "--sd-frame-stride",
                str(args.sd_frame_stride),
                "--sd-chunk-size",
                str(args.sd_chunk_size),
                "--sd-confidence-threshold",
                str(args.sd_confidence_threshold),
                "--sd-max-points-per-frame",
                str(args.sd_max_points_per_frame),
                "--sd-max-merged-points",
                str(args.sd_max_merged_points),
                "--sd-seed",
                str(args.sd_seed),
                "--sd-vggt-repo-dir",
                args.sd_vggt_repo_dir,
                "--sd-model-path",
                args.sd_model_path,
            ]
        )
        if args.sd_device is not None:
            cmd.extend(["--sd-device", args.sd_device])
        if args.sd_max_frames is not None:
            cmd.extend(["--sd-max-frames", str(args.sd_max_frames)])
        if args.sd_vggt_batch_size is not None:
            cmd.extend(["--sd-vggt-batch-size", str(args.sd_vggt_batch_size)])

    if args.eval_metadata:
        cmd.extend(["--eval-metadata", args.eval_metadata, "--split", args.split])
        if args.reference_root:
            cmd.extend(["--reference-root", args.reference_root])
    else:
        cmd.extend(["--reference-root", args.reference_root])

    if args.verbose:
        print("\n[run]", " ".join(cmd))

    subprocess.run(cmd, check=True)

    summary_path = step_output_dir / "summary.json"
    with summary_path.open("r") as f:
        summary = json.load(f)
    return summary_to_metrics(summary)


def value_for_metric(plot_metric: str, metrics: dict, invert_score: bool) -> float | None:
    if plot_metric == "memory_error":
        droid_score = metrics["droid_score"]
        if droid_score is None:
            return None
        return -droid_score if invert_score else droid_score
    value = metrics.get(plot_metric)
    if value is None:
        return None
    return float(value)


def plot_series(
    rows: list[dict],
    methods: list[tuple[str, str]],
    plot_metric: str,
    output_dir: Path,
    plot_title: str,
):
    plt.figure(figsize=(8, 5))
    for method_name, _ in methods:
        xs = [r["frame"] for r in rows if r["method"] == method_name and r[plot_metric] is not None]
        ys = [r[plot_metric] for r in rows if r["method"] == method_name and r[plot_metric] is not None]
        if xs:
            plt.plot(xs, ys, marker="o", label=method_name)

    plt.title(plot_title)
    plt.xlabel("Timestep / Frame Index")
    y_label = {
        "memory_error": "Memory Error (DROID reprojection)",
        "droid_score": "DROID Reprojection Score",
        "spatial_distance_mean": "Spatial Distance (mean)",
        "spatial_distance_max": "Spatial Distance (max)",
        "ssim": "SSIM",
        "psnr": "PSNR",
        "lpips": "LPIPS",
    }[plot_metric]
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = output_dir / f"memory_consistency_vs_time_{plot_metric}.png"
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create Memory Consistency vs Time plot by repeatedly running eval_videos.py "
            "with increasing --max-frames (e.g., 10, 20, 30, ...)."
        )
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method mapping in format: name=/path/to/generated/root . Repeat for multiple lines.",
    )
    parser.add_argument(
        "--reference-root",
        type=str,
        default=None,
        help="Reference root (required unless --eval-metadata is set).",
    )
    parser.add_argument(
        "--eval-metadata",
        type=str,
        default=None,
        help="Optional eval metadata JSON (if using metadata mode from eval_videos.py).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="memory",
        choices=["memory", "random", "all"],
        help="Split used when --eval-metadata is provided.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-first-n-frames", type=int, default=0)

    parser.add_argument("--start-frame", type=int, default=10)
    parser.add_argument("--end-frame", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)

    parser.add_argument(
        "--eval-script",
        type=str,
        default=str(Path(__file__).resolve().parent / "eval_videos.py"),
        help="Path to eval_videos.py.",
    )
    parser.add_argument("--invert-score", action="store_true")
    parser.add_argument("--plot-title", type=str, default="Memory Consistency vs Time")
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="memory_error",
        choices=PLOT_METRIC_CHOICES,
        help=(
            "Metric to plot on y-axis. memory_error uses droid_score and applies "
            "--invert-score when enabled."
        ),
    )
    parser.add_argument(
        "--plot-metrics",
        type=str,
        default=None,
        help="Comma-separated metrics to plot in one run (e.g. ssim,psnr,lpips). Overrides --plot-metric.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing raw/<method>/frames_XXXX/summary.json files and skip re-evaluation for those steps.",
    )
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--compute-droid", action="store_true")
    parser.add_argument("--compute-spatial-distance", action="store_true")

    # DROID options mirrored from eval_videos.py
    parser.add_argument(
        "--droid-weights",
        type=str,
        default="/u/cz5047/videogen/data/models/droid_models/droid.pth",
    )
    parser.add_argument("--droid-calib", type=str, default="500,500,256,256")
    parser.add_argument("--droid-stride", type=int, default=2)
    parser.add_argument("--droid-max-frames", type=int, default=200)
    parser.add_argument("--droid-resize", type=int, default=256)
    parser.add_argument("--droid-buffer", type=int, default=512)
    parser.add_argument("--droid-filter-thresh", type=float, default=0.01)
    parser.add_argument("--droid-upsample", action="store_true")
    # Spatial distance options mirrored from eval_videos.py
    parser.add_argument("--sd-device", type=str, default=None)
    parser.add_argument("--sd-frame-stride", type=int, default=1)
    parser.add_argument("--sd-max-frames", type=int, default=None)
    parser.add_argument("--sd-vggt-batch-size", type=int, default=None)
    parser.add_argument("--sd-confidence-threshold", type=float, default=0.2)
    parser.add_argument("--sd-max-points-per-frame", type=int, default=200000)
    parser.add_argument("--sd-max-merged-points", type=int, default=200000)
    parser.add_argument("--sd-chunk-size", type=int, default=4096)
    parser.add_argument("--sd-seed", type=int, default=0)
    parser.add_argument(
        "--sd-vggt-repo-dir",
        type=str,
        default="/u/cz5047/videogen/world-eval-latest/vggt",
    )
    parser.add_argument(
        "--sd-model-path",
        type=str,
        default="/u/cz5047/videogen/world-eval-latest/vggt/model.pt",
    )

    args = parser.parse_args()

    if args.eval_metadata is None and args.reference_root is None:
        parser.error("--reference-root is required unless --eval-metadata is set")
    if args.start_frame <= 0 or args.end_frame <= 0 or args.stride <= 0:
        parser.error("--start-frame, --end-frame, and --stride must be positive")
    if args.start_frame > args.end_frame:
        parser.error("--start-frame must be <= --end-frame")
    if not args.reuse_existing and not args.compute_droid and not args.compute_spatial_distance:
        parser.error("Enable at least one metric: --compute-droid and/or --compute-spatial-distance")
    if args.plot_metric in {"memory_error", "droid_score"} and not (
        args.compute_droid or args.reuse_existing
    ):
        parser.error("--plot-metric requires --compute-droid")
    if args.plot_metric.startswith("spatial_distance") and not (
        args.compute_spatial_distance or args.reuse_existing
    ):
        parser.error("--plot-metric requires --compute-spatial-distance")

    eval_script = Path(args.eval_script).resolve()
    if not eval_script.is_file():
        raise FileNotFoundError(f"Could not find eval script: {eval_script}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = [parse_method_arg(m) for m in args.method]
    frame_steps = list(range(args.start_frame, args.end_frame + 1, args.stride))
    if args.plot_metrics:
        selected_plot_metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    else:
        selected_plot_metrics = [args.plot_metric]
    invalid_metrics = [m for m in selected_plot_metrics if m not in PLOT_METRIC_CHOICES]
    if invalid_metrics:
        parser.error(f"Invalid metrics in --plot-metrics: {invalid_metrics}")
    for m in selected_plot_metrics:
        if m in {"memory_error", "droid_score"} and not args.compute_droid:
            if not args.reuse_existing:
                parser.error(f"{m} requires --compute-droid")
        if m.startswith("spatial_distance") and not args.compute_spatial_distance:
            if not args.reuse_existing:
                parser.error(f"{m} requires --compute-spatial-distance")

    series_rows = []
    for method_name, method_root in methods:
        for frame_idx in frame_steps:
            step_output_dir = output_dir / "raw" / method_name / f"frames_{frame_idx:04d}"
            metrics = None
            if args.reuse_existing:
                metrics = load_existing_metrics(step_output_dir=step_output_dir)
            if metrics is None:
                if args.reuse_existing and not (args.compute_droid or args.compute_spatial_distance):
                    raise RuntimeError(
                        "Missing cached summary for replot-only mode at "
                        f"{step_output_dir / 'summary.json'}"
                    )
                metrics = run_eval_videos(
                    eval_script=eval_script,
                    method_name=method_name,
                    generated_root=method_root,
                    output_dir=output_dir,
                    max_frames=frame_idx,
                    args=args,
                    step_output_dir=step_output_dir,
                )
            droid_score = metrics["droid_score"]
            sd_mean = metrics["spatial_distance_mean"]
            sd_max = metrics["spatial_distance_max"]
            ssim = metrics["ssim"]
            psnr = metrics["psnr"]
            lpips = metrics["lpips"]

            if (
                droid_score is None
                and sd_mean is None
                and sd_max is None
                and ssim is None
                and psnr is None
                and lpips is None
            ):
                continue
            series_rows.append(
                {
                    "method": method_name,
                    "frame": frame_idx,
                    "droid_score": droid_score,
                    "spatial_distance_mean": sd_mean,
                    "spatial_distance_max": sd_max,
                    "ssim": ssim,
                    "psnr": psnr,
                    "lpips": lpips,
                    "memory_error": value_for_metric("memory_error", metrics, args.invert_score),
                }
            )
            print(
                f"[{method_name}] frames<= {frame_idx:4d} | "
                f"droid={droid_score} | sd_mean={sd_mean} | sd_max={sd_max} | "
                f"ssim={ssim} | psnr={psnr} | lpips={lpips}"
            )

    if not series_rows:
        raise RuntimeError("No valid scores collected. Check paths and eval settings.")

    csv_path = output_dir / "memory_consistency_vs_time.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "frame",
                "droid_score",
                "spatial_distance_mean",
                "spatial_distance_max",
                "ssim",
                "psnr",
                "lpips",
                "memory_error",
            ],
        )
        writer.writeheader()
        writer.writerows(series_rows)

    generated_plot_paths = []
    for metric in selected_plot_metrics:
        png_path = plot_series(
            rows=series_rows,
            methods=methods,
            plot_metric=metric,
            output_dir=output_dir,
            plot_title=args.plot_title,
        )
        generated_plot_paths.append(str(png_path))

    summary_path = output_dir / "memory_consistency_vs_time_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "frame_steps": frame_steps,
                "methods": [m[0] for m in methods],
                "csv_path": str(csv_path),
                "plot_paths": generated_plot_paths,
                "invert_score": args.invert_score,
                "plot_metrics": selected_plot_metrics,
                "compute_droid": args.compute_droid,
                "compute_spatial_distance": args.compute_spatial_distance,
                "reuse_existing": args.reuse_existing,
            },
            f,
            indent=2,
        )

    print("\nDone.")
    print(f"CSV: {csv_path}")
    for p in generated_plot_paths:
        print(f"Plot: {p}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
