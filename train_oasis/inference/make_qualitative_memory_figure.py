#!/usr/bin/env python3
"""
Build a thesis-style figure from qualitative_high_low.json (output of
correlate_memory_perceptual_metrics.py).

Modes:
  clip_grid — multiple clips; one frame each (middle / third / index).
  temporal_strip — one high- and one low-scoring clip; a row of frames every
    K frames (default K=10) so temporal behavior is visible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torchvision


def load_video_rgb(path: Path) -> np.ndarray:
    """THWC float in [0, 1]."""
    video, _, _ = torchvision.io.read_video(str(path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"No frames in {path}")
    return video.float().numpy() / 255.0


def load_frame_rgb(path: Path, frame_index: str | int) -> np.ndarray:
    """HxWx3 float in [0,1] for a single frame."""
    v = load_video_rgb(path)
    t = v.shape[0]
    if frame_index == "middle":
        idx = t // 2
    elif frame_index == "third":
        idx = min(t - 1, max(0, t // 3))
    else:
        idx = int(frame_index)
        idx = max(0, min(t - 1, idx))
    frame = v[idx]
    if frame.shape[-1] == 3:
        return frame
    return frame


def subsample_stride_indices(num_frames: int, stride: int, max_panels: int) -> list[int]:
    """0, stride, 2*stride, ... capped by max_panels (even subsample if needed)."""
    if num_frames <= 0:
        return [0]
    raw = list(range(0, num_frames, max(1, stride)))
    if len(raw) > max_panels:
        pick = np.linspace(0, len(raw) - 1, max_panels, dtype=int)
        raw = [raw[int(i)] for i in pick]
    return raw


def _short_label(video_rel: str) -> str:
    p = Path(video_rel)
    return f"{p.parent.name}/{p.stem}" if p.parent.name else p.stem


def _item_path(item: dict[str, Any]) -> Path:
    path_str = item.get("generated_path") or ""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(
            f"Video not found: {path}\n"
            "Regenerate qualitative_high_low.json with correct --generated-root."
        )
    return path


def _title_metrics(item: dict[str, Any], video_rel: str) -> str:
    lab = _short_label(video_rel)
    d = item.get("droid_score")
    sd = item.get("spatial_distance_mean")
    psnr_v = item.get("psnr")
    tail = []
    if d is not None:
        tail.append(f"DROID {d:.2f}")
    if sd is not None:
        tail.append(f"SD {sd:.3f}")
    if psnr_v is not None:
        tail.append(f"PSNR {psnr_v:.1f}")
    return lab + ("\n" + ", ".join(tail) if tail else "")


def _row_axis_labels(signal: str) -> tuple[str, str]:
    """Short two-line labels for vertical text beside temporal strips (avoids overlap)."""
    if signal == "droid":
        return ("Higher\nDROID", "Lower\nDROID")
    return ("Higher spatial\nconsistency", "Lower spatial\nconsistency")


def run_clip_grid(args: argparse.Namespace, block: dict[str, Any]) -> None:
    high: list[dict] = block["high_consistency"]
    low: list[dict] = block["low_consistency"]
    n = min(args.ncols, len(high), len(low))
    if n < 1:
        raise SystemExit("Need at least one high and one low clip.")

    row_labels = _row_axis_labels(args.signal)

    fig_w = 2.5 * n + 0.9
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 5.0))
    axes = np.asarray(axes).reshape(2, n)

    plt.subplots_adjust(left=0.12, right=0.99, top=0.82, bottom=0.05, wspace=0.08, hspace=0.22)
    for row, (items, row_title) in enumerate(zip((high[:n], low[:n]), row_labels)):
        fig.text(
            0.03,
            0.70 - row * 0.42,
            row_title,
            fontsize=9,
            fontweight="bold",
            va="center",
            rotation=90,
            linespacing=0.95,
        )
        for col in range(n):
            ax = axes[row, col]
            item = items[col]
            path = _item_path(item)
            img = load_frame_rgb(path, args.frame)
            ax.imshow(np.clip(img, 0.0, 1.0))
            ax.axis("off")
            ax.set_title(_title_metrics(item, item.get("video_rel", path.name)), fontsize=8)

    note = block.get("note", "")
    if args.frame == "middle":
        frame_desc = "middle frame"
    elif args.frame == "third":
        frame_desc = "early frame (T/3)"
    else:
        frame_desc = f"frame {args.frame}"
    fig.suptitle(
        "Representative clips by memory score (" + frame_desc + ")\n"
        + (f"({note})" if note else ""),
        fontsize=10,
    )
    _save_fig(fig, args)


def run_temporal_strip(args: argparse.Namespace, block: dict[str, Any]) -> None:
    high: list[dict] = block["high_consistency"]
    low: list[dict] = block["low_consistency"]
    hi = args.high_rank
    lo = args.low_rank
    if hi >= len(high) or lo >= len(low):
        raise SystemExit(f"Need high_rank < {len(high)}, low_rank < {len(low)}.")

    item_hi = high[hi]
    item_lo = low[lo]
    path_hi = _item_path(item_hi)
    path_lo = _item_path(item_lo)

    v_hi = load_video_rgb(path_hi)
    v_lo = load_video_rgb(path_lo)
    t_hi, t_lo = v_hi.shape[0], v_lo.shape[0]
    idx_hi = subsample_stride_indices(t_hi, args.frame_stride, args.max_panels)
    idx_lo = subsample_stride_indices(t_lo, args.frame_stride, args.max_panels)
    ncols = max(len(idx_hi), len(idx_lo))

    # Figure size so each subplot cell matches frame W:H (no stretching with aspect="equal").
    h_hi, w_hi = int(v_hi.shape[1]), int(v_hi.shape[2])
    h_lo, w_lo = int(v_lo.shape[1]), int(v_lo.shape[2])
    H = max(h_hi, h_lo)
    W = max(w_hi, w_lo)
    if H <= 0 or W <= 0:
        raise SystemExit("Invalid frame dimensions in video.")
    row_h_in = float(args.panel_row_height_inches)
    panel_w_in = row_h_in * (W / H)
    fig_w_raw = ncols * panel_w_in + 0.42
    max_w = float(args.max_figure_width_inches)
    if fig_w_raw > max_w:
        scale = max_w / fig_w_raw
        row_h_in *= scale
        panel_w_in = row_h_in * (W / H)
        fig_w = max_w
    else:
        fig_w = fig_w_raw
    fig_h = 2 * row_h_in + 0.88
    fig, axes = plt.subplots(2, ncols, figsize=(fig_w, fig_h))
    axes = np.asarray(axes).reshape(2, ncols)

    row_labels = _row_axis_labels(args.signal)
    items_row = (item_hi, item_lo)
    videos = (v_hi, v_lo)
    indices_row = (idx_hi, idx_lo)

    # Tight layout: small hspace/wspace, no xlabels (saves vertical gap between rows).
    # Leave room for row labels drawn in axes coords on column 0 (outside subplot).
    plt.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.86,
        bottom=0.07,
        wspace=0.02,
        hspace=0.06,
    )
    for row in range(2):
        v = videos[row]
        idxs = indices_row[row]
        item = items_row[row]
        for col in range(ncols):
            ax = axes[row, col]
            if col < len(idxs):
                fi = idxs[col]
                img = np.clip(v[fi], 0.0, 1.0)
                ax.imshow(img, aspect="equal")
                ax.text(
                    0.03,
                    0.03,
                    f"t={fi}",
                    transform=ax.transAxes,
                    fontsize=5,
                    color="white",
                    va="bottom",
                    ha="left",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.55},
                )
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(left=False, bottom=False)
        # Same vertical extent as the frames in this row (all axes in row share y span).
        label_fs = 5.5 if args.signal == "spatial_consistency" else 6
        axes[row, 0].text(
            -0.12,
            0.5,
            row_labels[row],
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fs,
            fontweight="bold",
            linespacing=0.95,
            clip_on=False,
        )

    note = block.get("note", "")
    cap = (
        f"Every {args.frame_stride} frames (≤{args.max_panels} cols); "
        f"T={t_hi} / {t_lo}. "
        + (note if note else "")
    )
    fig.suptitle("High vs low memory score (temporal strips)\n" + cap, fontsize=8, y=0.995)
    fig.text(
        0.5,
        0.01,
        "Top: "
        + _title_metrics(item_hi, item_hi.get("video_rel", "")).replace("\n", " ")
        + "  |  Bottom: "
        + _title_metrics(item_lo, item_lo.get("video_rel", "")).replace("\n", " "),
        ha="center",
        fontsize=6.5,
    )
    _save_fig(fig, args)


def _save_fig(fig: plt.Figure, args: argparse.Namespace) -> None:
    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pad = args.save_pad_inches
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure from qualitative_high_low.json")
    parser.add_argument("--json", type=Path, required=True, help="qualitative_high_low.json")
    parser.add_argument(
        "--signal",
        type=str,
        default="droid",
        choices=["droid", "spatial_consistency"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="temporal_strip",
        choices=["clip_grid", "temporal_strip"],
        help="clip_grid: many clips, one frame each. temporal_strip: two clips, frames every --frame-stride.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="clip_grid only: clips per row.",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default="middle",
        help='clip_grid only: "middle", "third", or integer frame index.',
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=10,
        help="temporal_strip: show frames 0, stride, 2*stride, ...",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=20,
        help="temporal_strip: max columns per row (subsample if stride yields more).",
    )
    parser.add_argument(
        "--high-rank",
        type=int,
        default=0,
        help="temporal_strip: which entry in high_consistency (0 = best in list).",
    )
    parser.add_argument(
        "--low-rank",
        type=int,
        default=0,
        help="temporal_strip: which entry in low_consistency (0 = worst in list).",
    )
    parser.add_argument(
        "--panel-row-height-inches",
        type=float,
        default=0.58,
        help="temporal_strip: target height (in) of one frame row; width follows frame aspect ratio.",
    )
    parser.add_argument(
        "--max-figure-width-inches",
        type=float,
        default=24.0,
        help="temporal_strip: shrink panel height if total width would exceed this.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--save-pad-inches",
        type=float,
        default=0.02,
        help="Padding for bbox_inches=tight (smaller = tighter crop).",
    )
    parser.add_argument("--output", type=Path, default=Path("memory_qualitative_figure.png"))

    args = parser.parse_args()

    with open(args.json.expanduser().resolve()) as f:
        qual: dict[str, Any] = json.load(f)

    block = qual.get(args.signal)
    if not isinstance(block, dict) or "high_consistency" not in block:
        raise SystemExit(f"No usable block {args.signal!r} in JSON (keys: {list(qual.keys())})")

    if args.mode == "clip_grid":
        run_clip_grid(args, block)
    else:
        run_temporal_strip(args, block)


if __name__ == "__main__":
    main()
