"""300 DPI Optuna BOHB report: best-so-far curves with COMPLETE/PRUNED markers and importances."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

_REPO = Path(__file__).resolve().parents[1]
SUMMARY_PATH = _REPO / "docs" / "results" / "m3_bohb_summary.json"
OUT_PATH = _REPO / "docs" / "assets" / "figures" / "m3_bohb_tuning.png"

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":"]


def _style(ax) -> None:
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot(summary_path: Path = SUMMARY_PATH, out_path: Path = OUT_PATH) -> None:
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found; run bohb_tune.py first.")
    with open(summary_path, "r", encoding="utf-8") as f:
        studies = json.load(f).get("studies", {})

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 10})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.0, 5.6), dpi=300, gridspec_kw={"width_ratios": [1.0, 1.3]}
    )

    handles: list = []
    for idx, (name, data) in enumerate(studies.items()):
        color = OKABE_ITO[idx % len(OKABE_ITO)]
        style = LINESTYLES[idx % len(LINESTYLES)]
        complete = [
            (t["number"], t["value"])
            for t in data.get("trials", [])
            if t["state"] == "COMPLETE" and t["value"] is not None
        ]
        pruned = [
            (t["number"], t["value"])
            for t in data.get("trials", [])
            if t["state"] == "PRUNED" and t["value"] is not None
        ]
        handles.append(Line2D([], [], color=color, linestyle=style, linewidth=2.0, label=name))
        if complete:
            xs = np.array([x for x, _ in complete], dtype=float)
            ys = np.array([y for _, y in complete], dtype=float)
            ax1.plot(xs, np.maximum.accumulate(ys), color=color, linestyle=style, linewidth=2.0)
            ax1.scatter(xs, ys, marker="o", s=28, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        if pruned:
            px = np.array([x for x, _ in pruned], dtype=float)
            py = np.array([y for _, y in pruned], dtype=float)
            ax1.scatter(
                px,
                py,
                marker="X",
                s=52,
                facecolors="none",
                edgecolors=color,
                linewidths=1.4,
                zorder=3,
            )

    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Eval Return (best value reported)")
    ax1.set_title("Best-so-far Objective")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax1)
    handles += [
        Line2D([], [], marker="o", linestyle="", color="#4D4D4D", label="COMPLETE"),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#4D4D4D",
            label="PRUNED",
        ),
    ]

    ranked = [(name, data["param_importances"]) for name, data in studies.items() if data.get("param_importances")]
    if ranked:
        names = sorted(
            {param for _, importances in ranked for param in importances},
            key=lambda param: max(importances.get(param, 0.0) for _, importances in ranked),
        )
        positions = np.arange(len(names))
        height = 0.8 / len(ranked)
        for idx, (name, importances) in enumerate(ranked):
            color = OKABE_ITO[list(studies).index(name) % len(OKABE_ITO)]
            total = sum(importances.values()) or 1.0
            values = [importances.get(param, 0.0) / total for param in names]
            ax2.barh(
                positions + (idx - (len(ranked) - 1) / 2) * height,
                values,
                height=height,
                color=color,
                label=name,
            )
        ax2.set_yticks(positions)
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel("Normalized Importance")
        ax2.set_title("Hyperparameter Importances")
    else:
        ax2.text(
            0.5,
            0.5,
            "Importances need at least 2 COMPLETE trials",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )
        ax2.set_title("Hyperparameter Importances")
    _style(ax2)

    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=8,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot()
