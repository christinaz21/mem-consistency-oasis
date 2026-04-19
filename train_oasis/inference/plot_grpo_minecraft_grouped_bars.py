"""
GRPO Minecraft fine-tuning plots (table: tab:grpo_res).

1) Absolute grouped bars: baseline vs GRPO runs (DROID vs spatial-distance reward × 500/1000 fine-tuning steps).
2) Normalized vs baseline: % change oriented so positive = better on every metric
   (heatmap + grouped bars over the four fine-tuned checkpoints).
   Raw numbers stay in the paper table; these figures only show relative improvement.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# Column order matches the paper table (index 0 = baseline).
CATEGORY_AXIS_LABELS = [
    "Baseline",
    "DROID reward\n(500 fine-tuning steps)",
    "DROID reward\n(1000 fine-tuning steps)",
    "Spatial-distance reward\n(500 fine-tuning steps)",
    "Spatial-distance reward\n(1000 fine-tuning steps)",
]

# Single-line labels for the faceted figure legend (same order as axis columns).
CATEGORY_LEGEND_LABELS = [
    "Baseline (no GRPO)",
    "DROID reward, 500 fine-tuning steps",
    "DROID reward, 1000 fine-tuning steps",
    "Spatial-distance reward, 500 fine-tuning steps",
    "Spatial-distance reward, 1000 fine-tuning steps",
]

# Fine-tuned columns only (for % vs baseline heatmap / grouped plot); same strings as axis.
CONDITIONS_FT_LABELS = CATEGORY_AXIS_LABELS[1:]

# Table values (Minecraft GRPO)
GRPO_METRICS: list[dict] = [
    {
        "key": "droid",
        "title": "DROID",
        "ylabel": "DROID (lower is better)",
        "values": [3.513, 3.233, 3.321, 3.345, 3.234],
        "higher_is_better": False,
    },
    {
        "key": "s_mean",
        "title": "Spatial mean",
        "ylabel": "S-Mean (lower is better)",
        "values": [0.0680, 0.0767, 0.0780, 0.0759, 0.0793],
        "higher_is_better": False,
    },
    {
        "key": "s_max",
        "title": "Spatial max",
        "ylabel": "S-Max (lower is better)",
        "values": [0.1864, 0.1987, 0.1908, 0.1872, 0.1926],
        "higher_is_better": False,
    },
    {
        "key": "psnr",
        "title": "PSNR",
        "ylabel": "PSNR (higher is better)",
        "values": [12.15, 12.78, 12.79, 12.79, 12.80],
        "higher_is_better": True,
    },
    {
        "key": "ssim",
        "title": "SSIM",
        "ylabel": "SSIM (higher is better)",
        "values": [0.398, 0.426, 0.426, 0.426, 0.427],
        "higher_is_better": True,
    },
    {
        "key": "lpips",
        "title": "LPIPS",
        "ylabel": "LPIPS (lower is better)",
        "values": [0.665, 0.615, 0.614, 0.614, 0.614],
        "higher_is_better": False,
    },
]

# Full `GRPO_METRICS` keeps spatial columns for the paper table; figures only use:
FIGURE_METRIC_ORDER = ("droid", "psnr", "ssim", "lpips")


def figure_metrics() -> list[dict]:
    by_key = {m["key"]: m for m in GRPO_METRICS}
    return [by_key[k] for k in FIGURE_METRIC_ORDER]


# Baseline neutral; Droid ramp blue; SD ramp coral (matches BoN-style figures)
BAR_COLORS = ["#94a3b8", "#7c9ef0", "#4F6AF0", "#f0a090", "#E87461"]

# Distinct colors for metrics in the grouped % plot (DROID, PSNR, SSIM, LPIPS)
METRIC_COLORS = ["#4F6AF0", "#ca8a04", "#db2777", "#E87461"]

# Slightly compact figure sizes so tick/legend text reads larger relative to the data panel.
FACETED_FIGSIZE = (9.0, 6.9)
SINGLE_METRIC_FIGSIZE = (6.4, 3.85)
HEATMAP_FIGSIZE = (8.8, 5.35)
GROUPED_PCT_FIGSIZE = (10.5, 5.35)

FS_XTICK_MULTILINE = 10.0
FS_YTICK = 10.5
FS_AXIS_LABEL = 11.5
FS_SUBPLOT_TITLE = 12.5
FS_SUPTITLE = 14.0
FS_LEGEND = 9.5
FS_CAPTION = 10.0
FS_HEATMAP_CELL = 11.0


def pct_improvement_vs_baseline(
    value: float, baseline: float, *, higher_is_better: bool
) -> float:
    """
    Percent change vs baseline such that **positive always means better**.

    - Lower-is-better: +% when value drops (reduction from baseline).
    - Higher-is-better: +% when value rises (gain over baseline).
    """
    if abs(baseline) < 1e-15:
        return 0.0
    if higher_is_better:
        return 100.0 * (value - baseline) / abs(baseline)
    return 100.0 * (baseline - value) / abs(baseline)


def build_pct_improvement_matrix() -> tuple[np.ndarray, list[str]]:
    """Shape (n_metrics, 4): rows = metrics, cols = four GRPO fine-tuned conditions (see CATEGORY_AXIS_LABELS[1:])."""
    rows = []
    titles = []
    for spec in figure_metrics():
        titles.append(spec["title"])
        baseline = spec["values"][0]
        row = [
            pct_improvement_vs_baseline(
                spec["values"][k],
                baseline,
                higher_is_better=spec["higher_is_better"],
            )
            for k in range(1, 5)
        ]
        rows.append(row)
    return np.array(rows, dtype=float), titles


def _best_indices(values: list[float], higher_is_better: bool) -> list[int]:
    """All indices tied for best (e.g. LPIPS ties at 0.614)."""
    arr = np.array(values, dtype=float)
    if higher_is_better:
        thr = float(np.max(arr))
    else:
        thr = float(np.min(arr))
    return [i for i in range(len(arr)) if np.isclose(arr[i], thr, rtol=0.0, atol=1e-8)]


def _style_bar_axes(ax):
    ax.set_facecolor("#edf2f8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#94a3b8")
        ax.spines[side].set_linewidth(0.9)
    ax.grid(True, axis="y", color="#a8bbd4", linewidth=1.05, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#475569", axis="y", labelsize=FS_YTICK)


def _draw_metric_bars(ax, spec: dict, *, show_xlabels: bool) -> None:
    values = spec["values"]
    higher = spec["higher_is_better"]
    x = np.arange(len(CATEGORY_AXIS_LABELS))
    bests = set(_best_indices(values, higher))

    for i, (v, c) in enumerate(zip(values, BAR_COLORS)):
        edge = "#1e293b" if i in bests else "#ffffff"
        ew = 1.8 if i in bests else 0.9
        ax.bar(
            x[i],
            v,
            width=0.72,
            color=c,
            edgecolor=edge,
            linewidth=ew,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        CATEGORY_AXIS_LABELS, rotation=0, ha="center", fontsize=FS_XTICK_MULTILINE
    )
    if not show_xlabels:
        ax.set_xticklabels([])
    ax.set_ylabel(spec["ylabel"], fontsize=FS_AXIS_LABEL)
    ax.set_title(
        spec["title"],
        fontsize=FS_SUBPLOT_TITLE,
        fontweight="600",
        color="#0f172a",
        pad=6,
    )
    _style_bar_axes(ax)


def _grpo_rc():
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }


def save_faceted_grid(output_path: Path) -> None:
    with plt.rc_context(_grpo_rc()):
        fig, axes = plt.subplots(2, 2, figsize=FACETED_FIGSIZE, layout="constrained")
        fig.patch.set_facecolor("#ffffff")
        for ax, spec in zip(axes.flat, figure_metrics()):
            _draw_metric_bars(ax, spec, show_xlabels=False)
        # x labels only on bottom row
        for ax in axes[1, :]:
            ax.set_xticklabels(
                CATEGORY_AXIS_LABELS,
                rotation=0,
                ha="center",
                fontsize=FS_XTICK_MULTILINE,
            )
        fig.suptitle(
            "GRPO fine-tuning (Minecraft): DROID, PSNR, SSIM, LPIPS",
            fontsize=FS_SUPTITLE,
            fontweight="600",
            color="#0f172a",
            y=1.02,
        )
        fig.supxlabel(
            "Condition (spatial mean/max in table)",
            fontsize=FS_AXIS_LABEL,
            fontweight="500",
            color="#64748b",
            y=0.01,
        )

        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=BAR_COLORS[i], edgecolor="none")
            for i in range(len(CATEGORY_LEGEND_LABELS))
        ]
        fig.legend(
            handles,
            CATEGORY_LEGEND_LABELS,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.16),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=True,
            fancybox=True,
            facecolor="#ffffff",
            edgecolor="#d8dee9",
            fontsize=FS_LEGEND,
            columnspacing=1.05,
        )

        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor="#ffffff",
        )
        plt.close(fig)


def save_single_metric(spec: dict, output_path: Path) -> None:
    with plt.rc_context(_grpo_rc()):
        fig, ax = plt.subplots(figsize=SINGLE_METRIC_FIGSIZE, layout="constrained")
        _draw_metric_bars(ax, spec, show_xlabels=True)
        fig.suptitle(
            f"GRPO (Minecraft): {spec['title']}",
            fontsize=FS_SUPTITLE - 0.5,
            fontweight="600",
            color="#0f172a",
            y=1.02,
        )
        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="#ffffff",
        )
        plt.close(fig)


def save_improvement_heatmap(output_path: Path) -> None:
    P, row_labels = build_pct_improvement_matrix()
    pmax = float(np.max(np.abs(P)))
    if pmax < 1e-9:
        pmax = 1.0
    norm = TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax)

    with plt.rc_context(_grpo_rc()):
        fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE, layout="constrained")
        fig.patch.set_facecolor("#ffffff")
        im = ax.imshow(P, cmap="RdYlGn", norm=norm, aspect="auto")

        ax.set_xticks(np.arange(P.shape[1]))
        ax.set_xticklabels(
            CONDITIONS_FT_LABELS,
            rotation=0,
            ha="center",
            fontsize=FS_XTICK_MULTILINE,
        )
        ax.set_yticks(np.arange(P.shape[0]))
        ax.set_yticklabels(row_labels, fontsize=FS_YTICK)
        ax.set_xlabel(
            "GRPO condition (500 / 1000 = fine-tuning optimizer steps)",
            fontsize=FS_AXIS_LABEL,
            color="#334155",
        )
        ax.set_ylabel("Metric", fontsize=FS_AXIS_LABEL, color="#334155")

        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                val = P[i, j]
                tcol = "#f8fafc" if abs(val) > 0.55 * pmax else "#0f172a"
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=FS_HEATMAP_CELL,
                    color=tcol,
                    fontweight="500",
                )

        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=FS_YTICK)
        cbar.set_label(
            "% vs baseline (↑ better on all metrics)", fontsize=FS_AXIS_LABEL
        )

        fig.suptitle(
            "GRPO (Minecraft): improvement vs baseline",
            fontsize=FS_SUPTITLE,
            fontweight="600",
            color="#0f172a",
            y=1.02,
        )
        fig.text(
            0.5,
            -0.02,
            "Positive % = better for that metric. Spatial mean/max are in the full table.",
            ha="center",
            fontsize=FS_CAPTION,
            color="#64748b",
            transform=fig.transFigure,
        )

        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.22,
            facecolor="#ffffff",
        )
        plt.close(fig)


def save_improvement_grouped_bars(output_path: Path) -> None:
    """Four groups (fine-tuned GRPO conditions); within each group one bar per metric (% vs baseline)."""
    specs = figure_metrics()
    n_cond = len(CONDITIONS_FT_LABELS)
    n_met = len(specs)
    x = np.arange(n_cond, dtype=float)
    bar_w = 0.11
    offsets = (np.arange(n_met) - (n_met - 1) / 2.0) * bar_w

    with plt.rc_context(_grpo_rc()):
        fig, ax = plt.subplots(figsize=GROUPED_PCT_FIGSIZE, layout="constrained")
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#edf2f8")
        ax.axhline(0.0, color="#64748b", linewidth=1.05, zorder=1)

        for j, spec in enumerate(specs):
            baseline = spec["values"][0]
            heights = [
                pct_improvement_vs_baseline(
                    spec["values"][k + 1],
                    baseline,
                    higher_is_better=spec["higher_is_better"],
                )
                for k in range(n_cond)
            ]
            ax.bar(
                x + offsets[j],
                heights,
                width=bar_w * 0.92,
                label=spec["title"],
                color=METRIC_COLORS[j],
                edgecolor="#ffffff",
                linewidth=0.6,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            CONDITIONS_FT_LABELS,
            rotation=0,
            ha="center",
            fontsize=FS_XTICK_MULTILINE,
        )
        ax.set_xlabel(
            "GRPO condition (500 / 1000 = fine-tuning optimizer steps)",
            fontsize=FS_AXIS_LABEL,
            color="#334155",
        )
        ax.set_ylabel(
            "% vs baseline (↑ better on all metrics)", fontsize=FS_AXIS_LABEL
        )
        ax.grid(True, axis="y", color="#a8bbd4", linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#94a3b8")
        ax.spines["bottom"].set_color("#94a3b8")
        ax.tick_params(colors="#475569", axis="y", labelsize=FS_YTICK)

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fancybox=True,
            facecolor="#ffffff",
            edgecolor="#d8dee9",
            fontsize=FS_LEGEND,
        )

        fig.suptitle(
            "GRPO (Minecraft): % improvement vs baseline by condition",
            fontsize=FS_SUPTITLE,
            fontweight="600",
            color="#0f172a",
            y=1.03,
        )
        fig.text(
            0.5,
            -0.04,
            "Absolute values and spatial mean/max: paper table. Bars are % vs baseline only.",
            ha="center",
            fontsize=FS_CAPTION,
            color="#64748b",
            transform=fig.transFigure,
        )

        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.2,
            facecolor="#ffffff",
        )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="GRPO Minecraft grouped bar charts (per table).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--also-separate",
        action="store_true",
        help="Also write one PNG per metric (six files, absolute scale).",
    )
    p.add_argument(
        "--skip-absolute",
        action="store_true",
        help="Do not write absolute-scale faceted / separate bar figures.",
    )
    p.add_argument(
        "--skip-normalized",
        action="store_true",
        help="Do not write %% vs baseline heatmap / grouped bar figures.",
    )
    args = p.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_absolute:
        save_faceted_grid(out / "grpo_minecraft_grouped_bars_faceted.png")
        print(f"Saved: {out / 'grpo_minecraft_grouped_bars_faceted.png'}")
        if args.also_separate:
            for spec in figure_metrics():
                fn = f"grpo_minecraft_grouped_bars_{spec['key']}.png"
                save_single_metric(spec, out / fn)
                print(f"Saved: {out / fn}")

    if not args.skip_normalized:
        save_improvement_heatmap(out / "grpo_minecraft_improvement_pct_heatmap.png")
        save_improvement_grouped_bars(out / "grpo_minecraft_improvement_pct_grouped.png")
        print(f"Saved: {out / 'grpo_minecraft_improvement_pct_heatmap.png'}")
        print(f"Saved: {out / 'grpo_minecraft_improvement_pct_grouped.png'}")


if __name__ == "__main__":
    main()
