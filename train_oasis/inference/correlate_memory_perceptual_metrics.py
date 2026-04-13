#!/usr/bin/env python3
"""
Analyze correlation between memory-oriented metrics (DROID, spatial distance)
and frame-wise perceptual metrics (PSNR, SSIM, LPIPS) from per_video_metrics.csv
produced by eval_videos.py.

Also writes a qualitative manifest: top/bottom clips by a chosen memory score
for thesis figures (high vs low memory consistency).

Example (after running eval_videos.py):

    python train_oasis/inference/correlate_memory_perceptual_metrics.py \\
      --csv /path/to/output/per_video_metrics.csv \\
      --generated-root /path/to/generated/videos \\
      --output-dir /path/to/correlation_outputs \\
      --method spearman \\
      --qualitative-k 6

Install SciPy for two-sided p-values (train-oasis requirements.txt lists scipy).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _parse_float(cell: str) -> float | None:
    if cell is None:
        return None
    s = str(cell).strip()
    if s == "" or s.lower() in ("none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for raw in reader:
            row = dict(raw)
            for key in ("ssim", "psnr", "lpips", "droid_score", "spatial_distance_mean", "spatial_distance_max"):
                if key in row:
                    row[key] = _parse_float(row.get(key, ""))
            if "num_frames" in row and row["num_frames"] not in (None, ""):
                try:
                    row["num_frames"] = int(float(row["num_frames"]))
                except ValueError:
                    row["num_frames"] = None
            rows.append(row)
        return rows


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties (0 .. n-1 scale)."""
    n = len(values)
    sort_idx = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_vals = values[sort_idx]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = 0.5 * (i + j)
        ranks[sort_idx[i : j + 1]] = avg
        i = j + 1
    return ranks


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 0.0:
        return float("nan")
    return float((x * y).sum() / denom)


def _pearson_p_two_sided(r: float, n: int) -> float | None:
    """Two-sided p-value for Pearson r; requires SciPy for the t CDF."""
    if scipy_stats is None or n < 3:
        return None
    df = n - 2
    if abs(r) >= 1.0 - 1e-15:
        return 0.0
    t_stat = r * math.sqrt(df / max(1e-30, 1.0 - r * r))
    p = 2.0 * float(scipy_stats.t.sf(abs(t_stat), df))
    return min(1.0, max(0.0, p))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return None, None, n
    xs, ys = x[m], y[m]
    if scipy_stats is not None:
        r, p = scipy_stats.spearmanr(xs, ys)
        if isinstance(r, np.ndarray):
            r = float(r.ravel()[0])
        if isinstance(p, np.ndarray):
            p = float(p.ravel()[0])
        if math.isnan(r):
            return None, None, n
        return float(r), float(p) if p is not None and not math.isnan(p) else None, n
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    r = _pearson_r(rx, ry)
    if math.isnan(r):
        return None, None, n
    return float(r), None, n


def safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, int]:
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return None, None, n
    xs, ys = x[m], y[m]
    if scipy_stats is not None:
        r, p = scipy_stats.pearsonr(xs, ys)
        if math.isnan(r):
            return None, None, n
        return float(r), float(p) if p is not None and not math.isnan(p) else None, n
    r = _pearson_r(xs, ys)
    if math.isnan(r):
        return None, None, n
    return float(r), _pearson_p_two_sided(r, n), n


def build_arrays(
    rows: list[dict[str, Any]],
    memory_values: np.ndarray,
    perceptual_keys: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[int]]:
    """Return memory vector, dict of perceptual vectors, and list of row indices used."""
    idxs = []
    mem_list = []
    per_lists: dict[str, list[float]] = {k: [] for k in perceptual_keys}

    for i, row in enumerate(rows):
        m = float(memory_values[i])
        if not math.isfinite(m):
            continue
        ok = True
        per_row: dict[str, float] = {}
        for k in perceptual_keys:
            v = row.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ok = False
                break
            per_row[k] = float(v)
        if not ok:
            continue
        idxs.append(i)
        mem_list.append(float(m))
        for k in perceptual_keys:
            per_lists[k].append(per_row[k])

    mem_arr = np.asarray(mem_list, dtype=np.float64)
    per_arr = {k: np.asarray(per_lists[k], dtype=np.float64) for k in perceptual_keys}
    return mem_arr, per_arr, idxs


def correlation_block(
    memory: np.ndarray,
    perceptual: dict[str, np.ndarray],
    method: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": int(len(memory)), "method": method}
    for pk, y in perceptual.items():
        if method == "spearman":
            r, p, n = safe_spearman(memory, y)
        else:
            r, p, n = safe_pearson(memory, y)
        out[pk] = {"r": r, "p": p, "n": n}
    return out


def latex_table_rows(
    label: str,
    block: dict[str, Any],
    perceptual_keys: tuple[str, ...],
    digits: int = 3,
) -> str:
    cells = []
    for pk in perceptual_keys:
        cell = block.get(pk, {})
        r = cell.get("r")
        if r is None:
            cells.append("---")
        else:
            cells.append(f"{r:.{digits}f}")
    return f"{label} & " + " & ".join(cells) + r" \\"


def resolve_gen_path(generated_root: Path | None, video_rel: str) -> str | None:
    if generated_root is None:
        return None
    p = generated_root / video_rel
    if p.is_file():
        return str(p.resolve())
    return None

import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(memory, perceptual, name_x, name_y, save_path):
    plt.figure()
    plt.scatter(memory, perceptual, alpha=0.7)
    
    # optional: best-fit line
    m, b = np.polyfit(memory, perceptual, 1)
    plt.plot(memory, m*memory + b)

    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(f"{name_x} vs {name_y}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlate memory metrics (DROID, spatial distance) with PSNR/SSIM/LPIPS."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="per_video_metrics.csv from eval_videos.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write JSON / TeX / qualitative manifest (default: CSV parent).",
    )
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=None,
        help="Root of generated videos (resolves `video` column for qualitative manifest).",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="spearman",
        choices=["spearman", "pearson", "both"],
    )
    parser.add_argument(
        "--negate-spatial-distance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use -spatial_distance_mean as consistency (higher = closer to GT in 3D). Default: true.",
    )
    parser.add_argument(
        "--droid-higher-is-better",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank/select qualitative samples with higher DROID as better. Default: true.",
    )
    parser.add_argument(
        "--qualitative-k",
        type=int,
        default=6,
        help="Write top-k and bottom-k clips per memory signal (0 disables).",
    )
    parser.add_argument(
        "--qualitative-metric",
        type=str,
        default="both",
        choices=["droid", "spatial", "both"],
        help="Which memory signal to use for qualitative high/low lists.",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    out_dir = (args.output_dir or csv_path.parent).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(csv_path)
    n_raw = len(rows)
    perceptual_keys = ("psnr", "ssim", "lpips")

    # Per-row derived memory scores (same length as rows)
    spatial_src = np.array(
        [
            float(r["spatial_distance_mean"])
            if r.get("spatial_distance_mean") is not None
            else math.nan
            for r in rows
        ],
        dtype=np.float64,
    )
    if args.negate_spatial_distance:
        spatial_consistency = -spatial_src
    else:
        spatial_consistency = spatial_src

    droid_src = np.array(
        [
            float(r["droid_score"]) if r.get("droid_score") is not None else math.nan
            for r in rows
        ],
        dtype=np.float64,
    )

    results: dict[str, Any] = {
        "csv": str(csv_path),
        "num_rows_csv": n_raw,
        "negate_spatial_distance": args.negate_spatial_distance,
        "droid_higher_is_better": args.droid_higher_is_better,
        "perceptual_keys": list(perceptual_keys),
    }

    methods = ["spearman", "pearson"] if args.method == "both" else [args.method]

    for mem_name, label_tex in (
        ("droid", "DROID reprojection"),
        ("spatial_consistency", r"Spatial ($-$mean dist.)" if args.negate_spatial_distance else "Spatial mean dist."),
    ):
        if mem_name == "droid" and not np.any(np.isfinite(droid_src)):
            results["droid"] = {"skipped": True, "reason": "no droid_score values"}
            continue
        if mem_name == "spatial_consistency" and not np.any(np.isfinite(spatial_src)):
            results["spatial_consistency"] = {"skipped": True, "reason": "no spatial_distance_mean values"}
            continue

        mem_vals = droid_src if mem_name == "droid" else spatial_consistency
        memory, perceptual, used_idx = build_arrays(rows, mem_vals, perceptual_keys)
        block: dict[str, Any] = {
            "label": label_tex,
            "memory_column": "droid_score" if mem_name == "droid" else "spatial_distance_mean",
            "memory_transform": None if mem_name == "droid" else ("negated_mean" if args.negate_spatial_distance else "mean"),
            "n_used": int(len(memory)),
            "row_indices": used_idx,
        }
        for m in methods:
            block[m] = correlation_block(memory, perceptual, m)
        results[mem_name] = block

    json_path = out_dir / "memory_perceptual_correlation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # LaTeX table (Spearman by default; if both, prefer spearman in .tex body)
    tex_method = "spearman" if args.method in ("spearman", "both") else "pearson"
    tex_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Metric & PSNR & SSIM & LPIPS " + r"\\",
        r"\midrule",
    ]
    for mem_name, label_tex in (
        ("droid", "DROID reprojection"),
        ("spatial_consistency", r"Spatial ($-$mean dist.)" if args.negate_spatial_distance else "Spatial mean dist."),
    ):
        b = results.get(mem_name)
        if not isinstance(b, dict) or "skipped" in b:
            continue
        blk = b.get(tex_method)
        if blk is None:
            continue
        tex_lines.append(latex_table_rows(label_tex, blk, perceptual_keys))
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])

    tex_path = out_dir / "correlation_table.tex"
    with open(tex_path, "w") as f:
        f.write("% Correlation table (" + tex_method + r"). Build full table env in thesis." + "\n")
        f.write("\n".join(tex_lines) + "\n")

    # Qualitative manifest
    qual_path = out_dir / "qualitative_high_low.json"
    if args.qualitative_k > 0:
        k = args.qualitative_k
        gen_root = args.generated_root.expanduser().resolve() if args.generated_root else None

        def row_payload(i: int) -> dict[str, Any]:
            r = rows[i]
            rel = r.get("video", "")
            payload = {
                "video_rel": rel,
                "generated_path": resolve_gen_path(gen_root, rel) if gen_root else None,
                "reference": r.get("reference"),
                "psnr": r.get("psnr"),
                "ssim": r.get("ssim"),
                "lpips": r.get("lpips"),
                "droid_score": r.get("droid_score"),
                "spatial_distance_mean": r.get("spatial_distance_mean"),
            }
            return payload

        qual: dict[str, Any] = {"k": k, "generated_root": str(gen_root) if gen_root else None}

        def pick_order(mem_key: str) -> tuple[np.ndarray, list[int], str]:
            scores = []
            idxs = []
            if mem_key == "droid":
                for i, r in enumerate(rows):
                    v = r.get("droid_score")
                    if v is None:
                        continue
                    scores.append(float(v))
                    idxs.append(i)
                arr = np.asarray(scores, dtype=np.float64)
                note = "top = highest droid_score" if args.droid_higher_is_better else "top = lowest droid_score"
            else:
                for i in range(len(rows)):
                    v = spatial_consistency[i]
                    if not math.isfinite(v):
                        continue
                    scores.append(float(v))
                    idxs.append(i)
                arr = np.asarray(scores, dtype=np.float64)
                note = "top = highest spatial consistency (-mean dist)" if args.negate_spatial_distance else "top = highest raw spatial_distance_mean"
            return arr, idxs, note

        if args.qualitative_metric in ("droid", "both"):
            arr, idxs, note = pick_order("droid")
            if len(arr) >= 2 * k:
                if args.droid_higher_is_better:
                    order = np.argsort(-arr)
                else:
                    order = np.argsort(arr)
                top_idx = [idxs[int(j)] for j in order[:k]]
                bot_idx = [idxs[int(j)] for j in order[-k:][::-1]]
                qual["droid"] = {
                    "note": note,
                    "high_consistency": [row_payload(i) for i in top_idx],
                    "low_consistency": [row_payload(i) for i in bot_idx],
                }
            else:
                qual["droid"] = {"skipped": True, "reason": f"need >= {2 * k} clips with droid_score"}

        if args.qualitative_metric in ("spatial", "both"):
            arr, idxs, note = pick_order("spatial")
            if len(arr) >= 2 * k:
                if args.negate_spatial_distance:
                    order = np.argsort(-arr)
                else:
                    order = np.argsort(arr)
                top_idx = [idxs[int(j)] for j in order[:k]]
                bot_idx = [idxs[int(j)] for j in order[-k:][::-1]]
                qual["spatial_consistency"] = {
                    "note": note,
                    "high_consistency": [row_payload(i) for i in top_idx],
                    "low_consistency": [row_payload(i) for i in bot_idx],
                }
            else:
                qual["spatial_consistency"] = {
                    "skipped": True,
                    "reason": f"need >= {2 * k} clips with spatial_distance_mean",
                }

        with open(qual_path, "w") as f:
            json.dump(qual, f, indent=2)
    else:
        qual_path = None

    # Console summary
    print(f"Loaded {n_raw} rows from {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {tex_path}")
    if qual_path:
        print(f"Wrote {qual_path}")

    for mem_name in ("droid", "spatial_consistency"):
        b = results.get(mem_name)
        if not isinstance(b, dict) or "skipped" in b:
            continue
        print(f"\n=== {mem_name} (n={b['n_used']}) ===")
        for m in methods:
            blk = b[m]
            print(f"  [{m}]")
            for pk in perceptual_keys:
                c = blk[pk]
                r, p, n = c["r"], c["p"], c["n"]
                ps = f"p={p:.4g}" if p is not None else "p=NA"
                print(f"    vs {pk}: r={r}, {ps}, n={n}")

    print("\nLaTeX fragment (copy into table):")
    print(tex_path.read_text())
    if scipy_stats is None:
        print(
            "\nNote: SciPy not found; correlation coefficients are still valid, "
            "but p-values are omitted. Install scipy for p-values."
        )


    scatter_plot(memory, perceptual["psnr"], "DROID Score", "PSNR", "droid_vs_psnr.png")
    scatter_plot(memory, perceptual["ssim"], "DROID Score", "SSIM", "droid_vs_ssim.png")
    scatter_plot(memory, perceptual["lpips"], "DROID Score", "LPIPS", "droid_vs_lpips.png")

if __name__ == "__main__":
    main()
