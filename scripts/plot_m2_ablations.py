"""Generate 2-panel Milestone 2 ablation figure (Credit Assignment Alpha + LOS Occlusion)."""

import os
import pathlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from benchmarks.eval.evaluate import load_stage_runs, summarize_stage


def plot_ablations(out_path: str = "docs/assets/figures/m2_ablations_alpha_los.png") -> None:
    alpha_summary, alpha_md = summarize_stage("runs/ablation_alpha", n_boot=2000)
    los_grouped = load_stage_runs("runs/ablation_los")
    _, los_md = summarize_stage("runs/ablation_los", n_boot=2000)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=300)

    # Panel 1: Alpha IQM Return + 95% CI & Core HP %
    alphas_sorted = sorted(alpha_summary.keys())
    x_vals = [float(k.replace("alpha_", "").replace("_", ".")) for k in alphas_sorted]
    iqms = [alpha_summary[k]["eval/return_mean"]["final"]["iqm"] for k in alphas_sorted]
    lo_err = [iqms[i] - alpha_summary[k]["eval/return_mean"]["final"]["ci_low"] for i, k in enumerate(alphas_sorted)]
    hi_err = [alpha_summary[k]["eval/return_mean"]["final"]["ci_high"] - iqms[i] for i, k in enumerate(alphas_sorted)]
    core_hps = [alpha_summary[k]["eval/core_hp_ratio"]["final"]["iqm"] * 100.0 for k in alphas_sorted]

    ax1.errorbar(x_vals, iqms, yerr=[lo_err, hi_err], fmt="-o", color="#0072B2", linewidth=2.2, capsize=4, label="Return IQM [95% CI]")
    ax1.set_xlabel(r"Team Reward Mixing Weight $\alpha$")
    ax1.set_ylabel("Final Return IQM", color="#0072B2")
    ax1.tick_params(axis="y", labelcolor="#0072B2")
    ax1.set_xticks(x_vals)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.spines["top"].set_visible(False)

    ax1_r = ax1.twinx()
    ax1_r.plot(x_vals, core_hps, "--s", color="#009E73", linewidth=2.0, label="Core Survival (%)")
    ax1_r.set_ylabel("Core Survival (%)", color="#009E73")
    ax1_r.tick_params(axis="y", labelcolor="#009E73")
    ax1_r.spines["top"].set_visible(False)
    ax1.set_title(r"Credit Assignment Mixing ($\alpha \in \{0.0 \dots 1.0\}$, 3 Seeds)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")

    # Panel 2: LOS Partial Observability learning curves
    palette = {"los_on": ("#D55E00", "-", "Partial Obs (LOS Occluded)"), "los_off": ("#0072B2", "--", "Full Arena Vision (LOS Off)")}
    for key in sorted(los_grouped.keys()):
        trajs = los_grouped[key]
        color, ls, label = palette.get(key, ("#000000", "-", key))
        max_step = max(max(r["step"] for r in tr) for tr in trajs)
        grid = np.linspace(0, max_step, num=30)
        interp = [np.interp(grid, [r["step"] for r in tr], [r["eval/return_mean"] for r in tr]) for tr in trajs]
        mat = np.stack(interp, axis=0)
        mean_c, std_c = mat.mean(axis=0), mat.std(axis=0)
        ax2.plot(grid, mean_c, label=label, color=color, linestyle=ls, linewidth=2.2)
        ax2.fill_between(grid, mean_c - std_c, mean_c + std_c, color=color, alpha=0.18)

    ax2.set_title("Line-of-Sight Occlusion Ablation (Stage 3 MAPPO, 3 Seeds)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episodic Return")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc="best")

    fig.suptitle("DotShot Stage 3 Controlled Ablations (50k Steps, RTX 3060 CUDA)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved ablation figure to {out_path}\n")
    print("### Alpha Ablation Table\n" + alpha_md + "\n")
    print("### LOS Ablation Table\n" + los_md + "\n")


if __name__ == "__main__":
    plot_ablations()
