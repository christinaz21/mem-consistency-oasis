import argparse
from pathlib import Path

import matplotlib.pyplot as plt


N_VALUES = [1, 2, 4, 8, 16]

# Lower is better for droid/lpips/spatial, higher is better for psnr/ssim.
DATA = {
    "minecraft_droid_bon": {
        "droid": [3.24848, 3.14568, 3.05415, 3.08010, 2.93317],
        "psnr": [12.1469, 12.1137, 12.3133, 12.1944, 12.1278],
        "ssim": [0.397328, 0.400197, 0.411501, 0.411862, 0.410238],
        "lpips": [0.66813, 0.66017, 0.648004, 0.647025, 0.645938],
    },
    "cosmos_droid_bon": {
        "droid": [0.908372, 0.933473, 0.930459, 0.958272, 0.833185],
        "psnr": [11.29589, 11.34069, 11.31710, 11.32690, 11.32244],
        "ssim": [0.484458, 0.487136, 0.482985, 0.483032, 0.484716],
        "lpips": [0.624021, 0.621844, 0.621343, 0.620764, 0.621379],
    },
    "minecraft_spatial_bon": {
        "droid": [3.51317, 3.56132, 3.62514, 3.56361, 3.55650],
        "spatial_mean": [0.0680058, 0.0445144, 0.0354217, 0.0255137, 0.0216406],
        "spatial_max": [0.186428, 0.145477, 0.102886, 0.066676, 0.049559],
        "psnr": [12.1466, 12.1425, 12.2947, 12.2393, 12.2584],
        "ssim": [0.397950, 0.399726, 0.404329, 0.402442, 0.399471],
        "lpips": [0.66456, 0.65633, 0.64747, 0.65295, 0.65607],
    },
    "cosmos_spatial_bon": {
        "droid": [0.956887, 1.046385, 1.025316, 1.065048, 1.005427],
        "spatial_mean": [0.181527, 0.080481, 0.042033, 0.032332, 0.023465],
        "spatial_max": [0.387153, 0.190668, 0.106575, 0.089399, 0.064324],
        "psnr": [11.2959, 11.3229, 11.2908, 11.3537, 11.3261],
        "ssim": [0.484458, 0.485435, 0.480414, 0.483087, 0.484514],
        "lpips": [0.624021, 0.622043, 0.623762, 0.623693, 0.623225],
    },
}

# Keys shared by minecraft_droid_bon / cosmos_droid_bon / minecraft_spatial_bon / cosmos_spatial_bon
BON_METRIC_YLABELS = {
    "droid": "DROID score (lower is better)",
    "psnr": "PSNR (higher is better)",
    "ssim": "SSIM (higher is better)",
    "lpips": "LPIPS (lower is better)",
}

BON_METRIC_HEADLINE = {
    "droid": "DROID score",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "lpips": "LPIPS",
}


def _bon_rc_context() -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "600",
        "axes.titlecolor": "#0f172a",
        "axes.edgecolor": "#94a3b8",
        "text.color": "#334155",
        "axes.labelcolor": "#334155",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    }


def save_single_plot(
    x,
    y_a,
    y_b,
    label_a: str,
    label_b: str,
    title: str,
    y_label: str,
    output_path: Path,
):
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, y_a, marker="o", linewidth=2, label=label_a)
    plt.plot(x, y_b, marker="s", linewidth=2, label=label_b)
    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])
    plt.xlabel("n (Best-of-N)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_droid_and_sd_bon_same_plot(
    x,
    droid_score_droid_bon: list[float],
    droid_score_sd_bon: list[float],
    dataset_title: str,
    output_path: Path,
):
    """
    Both lines use the same y-axis: DROID score (lower is better).
    - DROID BoN: best sample chosen by DROID reward.
    - SD BoN: DROID score evaluated on the sample chosen by spatial-distance BoN
      (from your tables' DROID column under spatial BoN runs).
    """
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(
        x,
        droid_score_droid_bon,
        marker="o",
        linewidth=2,
        color="C0",
        label="DROID BoN",
    )
    plt.plot(
        x,
        droid_score_sd_bon,
        marker="s",
        linewidth=2,
        color="C1",
        label="SD BoN (DROID score on SD-selected videos)",
    )
    plt.xlabel("n (Best-of-N)")
    plt.ylabel("DROID score (lower is better)")
    plt.title(f"{dataset_title}: DROID BoN vs SD BoN")
    plt.xscale("log", base=2)
    plt.xticks(x, [str(v) for v in x])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _bon_panel_style(ax, *, grid_color: str | None = None, grid_linewidth: float | None = None):
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color("#94a3b8")
    gc = grid_color if grid_color is not None else "#e8edf2"
    glw = grid_linewidth if grid_linewidth is not None else 1.05
    ax.grid(
        True,
        axis="y",
        color=gc,
        linestyle="-",
        linewidth=glw,
        zorder=0,
    )
    ax.set_axisbelow(True)
    ax.tick_params(
        axis="both",
        colors="#475569",
        width=0.85,
        length=4,
        labelsize=10,
    )
    ax.margins(x=0.06, y=0.02)


def _bon_shrink_y_axis(ax, *, pad_frac: float = 0.038) -> None:
    """Tight y-limits from plotted lines with a small pad (less tall white space)."""
    lines = ax.get_lines()
    if not lines:
        return
    ys: list[float] = []
    for line in lines:
        ys.extend(float(v) for v in line.get_ydata())
    lo, hi = min(ys), max(ys)
    span = hi - lo
    if span <= 0:
        pad = max(abs(hi) * 0.012, 1e-6)
    else:
        pad = max(span * pad_frac, 1e-9)
    ax.set_ylim(lo - pad, hi + pad)


def _bon_line_kw(color: str, linestyle: str | tuple, marker: str, lw: float, ms: float):
    return dict(
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        marker=marker,
        markersize=ms,
        markeredgewidth=1.05,
        markeredgecolor="#ffffff",
        clip_on=False,
        zorder=3,
        antialiased=True,
        solid_capstyle="round",
        solid_joinstyle="round",
    )


def save_bon_minecraft_cosmos_combined_metric(
    x,
    metric: str,
    output_path: Path,
    *,
    pdf_title: str | None = None,
):
    """
    Side-by-side Minecraft | Cosmos, DROID BoN vs SD BoN, y-axis from `metric`
    (droid | psnr | ssim | lpips). Independent y-range per panel column.
    """
    if metric not in BON_METRIC_YLABELS:
        raise ValueError(f"Unknown metric {metric!r}, expected one of {set(BON_METRIC_YLABELS)}")

    c_bon = "#4F6AF0"
    c_sd = "#E87461"
    lw, ms = 2.35, 6.5

    with plt.rc_context(_bon_rc_context()):
        fig, (ax_mc, ax_co) = plt.subplots(
            1,
            2,
            # Short height: less vertical pixels per unit of y (flatter curves visually).
            figsize=(10.4, 2.45),
            constrained_layout=True,
        )
        fig.patch.set_facecolor("#ffffff")
        fig.patch.set_alpha(1.0)

        kw0 = _bon_line_kw(c_bon, "-", "o", lw, ms)
        kw1 = _bon_line_kw(c_sd, (0, (4.5, 3)), "s", lw, ms)

        y_mc_db = DATA["minecraft_droid_bon"][metric]
        y_mc_sd = DATA["minecraft_spatial_bon"][metric]
        y_co_db = DATA["cosmos_droid_bon"][metric]
        y_co_sd = DATA["cosmos_spatial_bon"][metric]

        (l0,) = ax_mc.plot(x, y_mc_db, label="DROID BoN", **kw0)
        (l1,) = ax_mc.plot(x, y_mc_sd, label="SD BoN", **kw1)
        ax_mc.set_title("Minecraft", pad=10)
        _bon_panel_style(ax_mc)

        ax_co.plot(x, y_co_db, **kw0)
        ax_co.plot(x, y_co_sd, **kw1)
        ax_co.set_title("Cosmos", pad=10)
        _bon_panel_style(ax_co)

        for ax in (ax_mc, ax_co):
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x])
            _bon_shrink_y_axis(ax)

        fig.supylabel(
            BON_METRIC_YLABELS[metric],
            fontsize=11.5,
            fontweight="500",
            color="#1e293b",
        )
        fig.supxlabel(
            "Best-of-N (n)",
            fontsize=11.5,
            fontweight="500",
            color="#1e293b",
        )

        fig.legend(
            handles=[l0, l1],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=True,
            fancybox=True,
            shadow=False,
            facecolor="#ffffff",
            edgecolor="#d8dee9",
            framealpha=1.0,
            fontsize=10.75,
            columnspacing=2.4,
            borderpad=0.65,
            labelspacing=0.45,
            handlelength=2.6,
            handletextpad=0.55,
        )

        save_kw: dict = dict(dpi=240, bbox_inches="tight", pad_inches=0.14)
        if pdf_title is not None:
            save_kw["metadata"] = {"Title": pdf_title}
        fig.savefig(output_path, **save_kw)
        plt.close(fig)


def save_droid_sd_bon_minecraft_and_cosmos_combined(x, output_path: Path):
    """Same layout as other BoN plots; y-axis is DROID score."""
    save_bon_minecraft_cosmos_combined_metric(
        x,
        "droid",
        output_path,
        pdf_title="Best-of-N: DROID vs SD selection",
    )


def save_bon_minecraft_cosmos_grid_2x2(x, output_path: Path):
    """2×2 grid of Minecraft|Cosmos panels: DROID, PSNR, SSIM, LPIPS."""
    if not hasattr(plt.Figure, "subfigures"):
        raise RuntimeError(
            "Matplotlib 3.5+ is required for the 2×2 grid figure (Figure.subfigures). "
            "Upgrade matplotlib or use the single-metric PNG outputs only."
        )
    order = ["droid", "psnr", "ssim", "lpips"]
    c_bon = "#4F6AF0"
    c_sd = "#E87461"
    lw, ms = 2.35, 6.5
    kw0 = _bon_line_kw(c_bon, "-", "o", lw, ms)
    kw1 = _bon_line_kw(c_sd, (0, (4.5, 3)), "s", lw, ms)

    with plt.rc_context(_bon_rc_context()):
        fig = plt.figure(figsize=(14.6, 6.75), layout="constrained")
        # White page; gutters stay white; each metric cell is a lightly tinted panel.
        fig.patch.set_facecolor("#ffffff")

        subfigs = fig.subfigures(
            2,
            2,
            hspace=0.16,
            # Tighter gap between left column (DROID, SSIM) and right (PSNR, LPIPS).
            wspace=0.055,
        )

        handles: tuple | None = None
        for i, mkey in enumerate(order):
            subfig = subfigs.flat[i]
            subfig.set_facecolor("#f1f5fa")
            _p = subfig.patch
            _p.set_edgecolor("#c5d0de")
            _p.set_linewidth(1.15)
            _p.set_alpha(1.0)

            subfig.suptitle(
                BON_METRIC_HEADLINE[mkey],
                fontsize=12.8,
                fontweight="600",
                color="#0f172a",
                y=1.085,
            )
            ax_mc, ax_co = subfig.subplots(1, 2)
            (l0,) = ax_mc.plot(
                x,
                DATA["minecraft_droid_bon"][mkey],
                label="DROID BoN",
                **kw0,
            )
            (l1,) = ax_mc.plot(
                x,
                DATA["minecraft_spatial_bon"][mkey],
                label="SD BoN",
                **kw1,
            )
            ax_co.plot(x, DATA["cosmos_droid_bon"][mkey], **kw0)
            ax_co.plot(x, DATA["cosmos_spatial_bon"][mkey], **kw1)

            ax_mc.set_title("Minecraft", pad=8)
            ax_co.set_title("Cosmos", pad=8)
            # Slightly darker y-grid on tinted axes so lines stay visible.
            _bon_panel_style(ax_mc, grid_color="#a8bbd4", grid_linewidth=1.12)
            _bon_panel_style(ax_co, grid_color="#a8bbd4", grid_linewidth=1.12)
            # Soft tint inside the axes (white figure / gutters; tinted chart areas).
            _chart_bg = "#edf2f8"
            ax_mc.set_facecolor(_chart_bg)
            ax_co.set_facecolor(_chart_bg)
            for ax in (ax_mc, ax_co):
                ax.set_xscale("log", base=2)
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in x])
            _bon_shrink_y_axis(ax_mc)
            _bon_shrink_y_axis(ax_co)

            row = i // 2
            # Every cell needs its own y label (previously only the left column had one).
            subfig.supylabel(
                BON_METRIC_YLABELS[mkey],
                fontsize=10.75,
                fontweight="500",
                color="#1e293b",
            )
            if row == 1:
                subfig.supxlabel(
                    "Best-of-N (n)",
                    fontsize=11.25,
                    fontweight="500",
                    color="#1e293b",
                )
            if handles is None:
                handles = (l0, l1)

        assert handles is not None
        fig.legend(
            handles=list(handles),
            labels=["DROID BoN", "SD BoN"],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.075),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=True,
            fancybox=True,
            facecolor="#ffffff",
            edgecolor="#d8dee9",
            fontsize=10.75,
            columnspacing=2.6,
            handlelength=2.6,
            handletextpad=0.55,
        )

        fig.savefig(
            output_path,
            dpi=240,
            bbox_inches="tight",
            pad_inches=0.26,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            metadata={"Title": "Best-of-N: DROID / PSNR / SSIM / LPIPS"},
        )
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot Best-of-N reward curves from hardcoded datapoints.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--also-plot-quality-metrics",
        action="store_true",
        help="Also generate PSNR/SSIM/LPIPS plots in addition to DROID+Spatial.",
    )
    parser.add_argument(
        "--also-cross-dataset",
        action="store_true",
        help="Also save Minecraft-vs-Cosmos comparison plots (droid-only, sd-only).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per dataset: DROID score for DROID-BoN vs DROID score on videos from SD-BoN (same y-axis).
    save_droid_and_sd_bon_same_plot(
        x=N_VALUES,
        droid_score_droid_bon=DATA["minecraft_droid_bon"]["droid"],
        droid_score_sd_bon=DATA["minecraft_spatial_bon"]["droid"],
        dataset_title="Minecraft",
        output_path=output_dir / "best_of_n_minecraft_droid_vs_sd_bon.png",
    )
    save_droid_and_sd_bon_same_plot(
        x=N_VALUES,
        droid_score_droid_bon=DATA["cosmos_droid_bon"]["droid"],
        droid_score_sd_bon=DATA["cosmos_spatial_bon"]["droid"],
        dataset_title="Cosmos",
        output_path=output_dir / "best_of_n_cosmos_droid_vs_sd_bon.png",
    )
    save_droid_sd_bon_minecraft_and_cosmos_combined(
        x=N_VALUES,
        output_path=output_dir / "best_of_n_minecraft_and_cosmos_droid_vs_sd_bon.png",
    )
    save_bon_minecraft_cosmos_combined_metric(
        x=N_VALUES,
        metric="psnr",
        output_path=output_dir / "best_of_n_minecraft_and_cosmos_bon_psnr.png",
        pdf_title="Best-of-N: PSNR (DROID vs SD selection)",
    )
    save_bon_minecraft_cosmos_combined_metric(
        x=N_VALUES,
        metric="ssim",
        output_path=output_dir / "best_of_n_minecraft_and_cosmos_bon_ssim.png",
        pdf_title="Best-of-N: SSIM (DROID vs SD selection)",
    )
    save_bon_minecraft_cosmos_combined_metric(
        x=N_VALUES,
        metric="lpips",
        output_path=output_dir / "best_of_n_minecraft_and_cosmos_bon_lpips.png",
        pdf_title="Best-of-N: LPIPS (DROID vs SD selection)",
    )
    save_bon_minecraft_cosmos_grid_2x2(
        x=N_VALUES,
        output_path=output_dir / "best_of_n_minecraft_and_cosmos_bon_grid_2x2.png",
    )

    if args.also_cross_dataset:
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_droid_bon"]["droid"],
            y_b=DATA["cosmos_droid_bon"]["droid"],
            label_a="Minecraft DROID-BoN",
            label_b="Cosmos DROID-BoN",
            title="DROID Reward vs Best-of-N",
            y_label="DROID Score (lower is better)",
            output_path=output_dir / "best_of_n_droid.png",
        )
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_spatial_bon"]["spatial_mean"],
            y_b=DATA["cosmos_spatial_bon"]["spatial_mean"],
            label_a="Minecraft Spatial-BoN",
            label_b="Cosmos Spatial-BoN",
            title="Spatial Distance Mean vs Best-of-N",
            y_label="Spatial Distance Mean (lower is better)",
            output_path=output_dir / "best_of_n_spatial_mean.png",
        )
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_spatial_bon"]["spatial_max"],
            y_b=DATA["cosmos_spatial_bon"]["spatial_max"],
            label_a="Minecraft Spatial-BoN",
            label_b="Cosmos Spatial-BoN",
            title="Spatial Distance Max vs Best-of-N",
            y_label="Spatial Distance Max (lower is better)",
            output_path=output_dir / "best_of_n_spatial_max.png",
        )

    if args.also_plot_quality_metrics:
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_droid_bon"]["psnr"],
            y_b=DATA["cosmos_droid_bon"]["psnr"],
            label_a="Minecraft DROID-BoN",
            label_b="Cosmos DROID-BoN",
            title="PSNR vs Best-of-N",
            y_label="PSNR (higher is better)",
            output_path=output_dir / "best_of_n_psnr.png",
        )
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_droid_bon"]["ssim"],
            y_b=DATA["cosmos_droid_bon"]["ssim"],
            label_a="Minecraft DROID-BoN",
            label_b="Cosmos DROID-BoN",
            title="SSIM vs Best-of-N",
            y_label="SSIM (higher is better)",
            output_path=output_dir / "best_of_n_ssim.png",
        )
        save_single_plot(
            x=N_VALUES,
            y_a=DATA["minecraft_droid_bon"]["lpips"],
            y_b=DATA["cosmos_droid_bon"]["lpips"],
            label_a="Minecraft DROID-BoN",
            label_b="Cosmos DROID-BoN",
            title="LPIPS vs Best-of-N",
            y_label="LPIPS (lower is better)",
            output_path=output_dir / "best_of_n_lpips.png",
        )

    print(f"Saved plots to: {output_dir}")
    print(
        "  BoN (MC|Cosmos, DROID vs SD lines): "
        "best_of_n_minecraft_and_cosmos_droid_vs_sd_bon.png, "
        "best_of_n_minecraft_and_cosmos_bon_psnr.png, "
        "best_of_n_minecraft_and_cosmos_bon_ssim.png, "
        "best_of_n_minecraft_and_cosmos_bon_lpips.png, "
        "best_of_n_minecraft_and_cosmos_bon_grid_2x2.png"
    )


if __name__ == "__main__":
    main()
