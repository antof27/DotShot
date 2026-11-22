"""Publication-grade 300 DPI plotting using the Okabe-Ito colorblind-safe palette."""

import argparse
import os
from typing import List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.eval.evaluate import load_stage_runs, summarize_stage


OKABE_ITO = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Bluish Green
    "#CC79A7",  # Reddish Purple
    "#D55E00",  # Vermilion
    "#56B4E9",  # Sky Blue
    "#000000",  # Black
    "#F0E442",  # Yellow
]

LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def plot_stage_curves(
    stage_dir: str,
    out_path: str,
    title: str,
    metrics: List[str] | None = None,
) -> None:
    grouped = load_stage_runs(stage_dir)
    if not grouped:
        raise RuntimeError(f"No runs found in {stage_dir}")

    if metrics is None:
        metrics = ["eval/return_mean", "eval/elemental_accuracy"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 4.2), dpi=300)
    if len(metrics) == 1:
        axes = [axes]

    metric_labels = {
        "eval/return_mean": "Episodic Return",
        "eval/win_rate": "Win Rate",
        "eval/core_hp_ratio": "Core Survival Ratio",
        "eval/elemental_accuracy": "Elemental Accuracy",
        "eval/kills": "Enemies Eliminated",
    }

    for ax, metric in zip(axes, metrics):
        for idx, (algo, trajs) in enumerate(sorted(grouped.items())):
            color = OKABE_ITO[idx % len(OKABE_ITO)]
            ls = LINESTYLES[idx % len(LINESTYLES)]

            # Collect common step grid via linear interpolation
            max_step = max(max(r["step"] for r in tr) for tr in trajs)
            grid = np.linspace(0, max_step, num=40)
            interp_curves = []
            for tr in trajs:
                xs = np.array([r["step"] for r in tr], dtype=np.float64)
                ys = np.array([r.get(metric, 0.0) for r in tr], dtype=np.float64)
                if len(xs) == 1:
                    interp_curves.append(np.full_like(grid, ys[0]))
                else:
                    interp_curves.append(np.interp(grid, xs, ys))

            mat = np.stack(interp_curves, axis=0)
            mean_curve = mat.mean(axis=0)
            std_curve = mat.std(axis=0)

            ax.plot(grid, mean_curve, label=algo.upper(), color=color, linestyle=ls, linewidth=2.0)
            if mat.shape[0] > 1:
                ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.18)

        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="best")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-grade learning curves from DotShot runs.")
    parser.add_argument("--stage-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--title", type=str, default="DotShot Benchmark Comparison")
    parser.add_argument("--metrics", nargs="+", default=["eval/return_mean", "eval/elemental_accuracy"])
    args = parser.parse_args()

    plot_stage_curves(args.stage_dir, args.out, args.title, args.metrics)
    _, md_table = summarize_stage(args.stage_dir)
    print(f"Saved plot to {args.out}\n\n{md_table}")


if __name__ == "__main__":
    main()
