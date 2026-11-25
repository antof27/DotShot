"""Figures for the report and handbook (docs/assets/figures/).

Each figure is built only if its inputs exist, so the script can run while
experiments are still in progress. Colours follow the learner, never its rank;
scratch and warm-started runs of one learner share a colour and differ in line
style or marker.

    python scripts/plot_report_figures.py            # every figure whose data exists
    python scripts/plot_report_figures.py --only tuning learnability
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = ROOT / "docs" / "assets" / "figures"
RES = ROOT / "docs" / "results"
RUNS = ROOT / "runs"

# Reference categorical palette (light mode), fixed learner -> slot mapping.
COLOR = {"PPO": "#2a78d6", "MAPPO": "#2a78d6", "HAPPO": "#eb6834", "HATRPO": "#1baf7a",
         "HAA2C": "#eda100", "MAT": "#e87ba4", "PPO transfer": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SCRIPT = "#52514e"
LEARNERS = ["MAPPO", "HAPPO", "HATRPO", "HAA2C", "MAT"]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "lines.linewidth": 1.8, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def learner_name(key: str) -> str:
    for name in LEARNERS:
        if name.lower() in key.lower():
            return name
    return "PPO"


def _rows(path: pathlib.Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _curve(run: pathlib.Path, repeat: int = 8, key: str = "eval/return_mean"):
    rows = _rows(run / "metrics.jsonl")
    pts = [(r["step"] * repeat, r[key]) for r in rows if key in r]
    return np.array(pts) if pts else None


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.png")


# --------------------------------------------------------------------------- development history


def fig_learnability() -> None:
    """Stage 3 HARL-MAPPO: what made the stage learnable (single-seed development runs)."""
    runs = [
        ("Scratch, 6M-tick schedule", RUNS / "scale_k8/stage3_harl_mappo_v22/seed_0", "-", MUTED),
        ("Scratch, GRU actor", RUNS / "round6/gru/stage3_harl_mappo_v22/seed_0", ":", MUTED),
        ("Scratch, guard potential", RUNS / "round6/guard/stage3_harl_mappo_v22/seed_0", "--", MUTED),
        ("Scratch, 24M-tick schedule", RUNS / "round6/long/stage3_harl_mappo_v22/seed_0", "-", COLOR["MAPPO"]),
        ("Cloned script + RL", RUNS / "round7/bc_a/stage3_harl_mappo_v22/seed_0", "-", COLOR["HAPPO"]),
    ]
    if not all((r / "metrics.jsonl").exists() for _, r, _, _ in runs):
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for label, run, ls, color in runs:
        c = _curve(run)
        ax.plot(c[:, 0] / 1e6, c[:, 1], ls=ls, color=color, label=label,
                marker="o" if "Cloned" in label else None, ms=3.5)
    ax.axhline(87.59, color=SCRIPT, ls=(0, (4, 3)), lw=1.2)
    ax.text(24.3, 87.59, "scripted\nreference", va="center", ha="left", color=MUTED, fontsize=7.5)
    ax.set_xlabel("Simulator ticks (millions)")
    ax.set_ylabel("Validation return")
    ax.set_xlim(0, 24)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Stage 3, HARL-MAPPO, one seed: budget and warm start decide learnability", loc="left")
    _save(fig, "dev_stage3_learnability")


def fig_action_repeat() -> None:
    path = RES / "dev" / "action_repeat.json"
    if not path.exists():
        return
    rows = [r for r in json.loads(path.read_text())["rows"] if r["learner"].startswith("SB3 PPO, Stage 2")]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    labels = [f"k = {r['k']}\n{r['ticks'] / 1e6:.0f}M ticks" for r in rows]
    vals = [100 * r["win_rate"] for r in rows]
    bars = ax.bar(labels, vals, color=COLOR["PPO"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", color=INK, fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Test win rate (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Stage 2 PPO by decision interval", loc="left")
    _save(fig, "dev_action_repeat")


# --------------------------------------------------------------------------- hyperparameter search


def fig_tuning() -> None:
    path = RES / "optuna_summary.json"
    if not path.exists():
        return
    studies = json.loads(path.read_text())["studies"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, stage in zip(axes, (3, 4)):
        for name in LEARNERS:
            key = f"s{stage}_{'mat' if name == 'MAT' else 'harl_' + name.lower()}"
            s = studies.get(key)
            if not s:
                continue
            for t in s["trials"]:
                if t["state"] != "COMPLETE":
                    continue
                bc = t["params"].get("warm_start") == "bc"
                ax.scatter(t["number"], t["value"], s=22, color=COLOR[name], marker="o" if bc else "x",
                           edgecolor="white" if bc else None, linewidth=0.6 if bc else 1.2, zorder=3)
            best = [max([u["value"] for u in s["trials"][: i + 1] if u["state"] == "COMPLETE"] or [np.nan])
                    for i in range(len(s["trials"]))]
            ax.plot(range(len(best)), best, color=COLOR[name], lw=1.2, label=name)
        ax.set_title(f"Stage {stage}", loc="left")
        ax.set_xlabel("Trial")
    axes[0].set_ylabel("Objective (mean final validation return)")
    handles = [plt.Line2D([], [], color=COLOR[n], lw=1.5, label=n) for n in LEARNERS]
    handles += [plt.Line2D([], [], color=MUTED, marker="o", ls="", label="cloned start"),
                plt.Line2D([], [], color=MUTED, marker="x", ls="", label="scratch")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Hyperparameter search: completed trials and best-so-far per learner", x=0.01, ha="left", fontsize=9.5)
    fig.subplots_adjust(top=0.84, wspace=0.08)
    _save(fig, "tuning_stage34")

    # parameter importances (fANOVA) of the multi-agent studies
    names = [k for k in studies if studies[k].get("param_importances") and not k.startswith(("s1", "s2"))]
    if names:
        params = sorted({p for k in names for p in studies[k]["param_importances"]})
        mat = np.array([[studies[k]["param_importances"].get(p, np.nan) for p in params] for k in names])
        fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(names) + 1.2))
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(params)), params, rotation=40, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(params)):
                if not np.isnan(mat[i, j]) and mat[i, j] >= 0.1:
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mat[i, j] > 0.55 else INK)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="fANOVA importance")
        ax.set_title("Which hyperparameters mattered (share of objective variance)", loc="left")
        _save(fig, "tuning_importances")


# --------------------------------------------------------------------------- final runs


def _band(run_dirs, key="eval/return_mean", repeat=8, scale=1.0):
    """Mean and 95% bootstrap band over seeds, at the evaluation steps every seed reached."""
    curves = [dict((r["step"], r[key]) for r in _rows(d / "metrics.jsonl") if key in r) for d in run_dirs]
    steps = sorted(set.intersection(*(set(c) for c in curves))) if curves else []
    rng = np.random.default_rng(0)
    xs, mean, lo, hi = [], [], [], []
    for s in steps:
        v = np.array([c[s] for c in curves]) * scale
        boots = rng.choice(v, size=(1000, len(v))).mean(1)
        xs.append(s * repeat / 1e6)
        mean.append(v.mean())
        lo.append(np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5))
    return np.array(xs), np.array(mean), np.array(lo), np.array(hi)


def _seeds(cfg_dir: pathlib.Path):
    return sorted(d for d in cfg_dir.glob("seed_*") if (d / "metrics.jsonl").exists()
                  and "test/return_mean" in (d / "metrics.jsonl").read_text(encoding="utf-8"))


def fig_final_single() -> None:
    s1 = _seeds(RUNS / "final/stage1/s1_sb3_ppo_scratch")
    s2 = {"scratch": _seeds(RUNS / "final/stage2/s2_sb3_ppo_scratch"),
          "transfer": _seeds(RUNS / "final/stage2/s2_sb3_ppo_transfer")}
    if not s1 or not all(s2.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    panels = [
        (axes[0], {"PPO": s1}, "eval/return_mean", 1.0, "Stage 1: validation return"),
        (axes[1], {"PPO scratch": s2["scratch"], "PPO transfer": s2["transfer"]}, "eval/return_mean", 1.0,
         "Stage 2: validation return"),
        (axes[2], {"PPO scratch": s2["scratch"], "PPO transfer": s2["transfer"]}, "eval/elemental_accuracy", 100.0,
         "Stage 2: right element (%)"),
    ]
    for ax, groups, key, scale, title in panels:
        for label, dirs in groups.items():
            color = COLOR["PPO transfer"] if "transfer" in label else COLOR["PPO"]
            xs, m, lo, hi = _band(dirs, key, scale=scale)
            ax.plot(xs, m, color=color, label=f"{label.replace('PPO ', '')} (n={len(dirs)})")
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Ticks (millions)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "final_stage12_curves")


FIGURES = {"learnability": fig_learnability, "action_repeat": fig_action_repeat, "tuning": fig_tuning,
           "final_single": fig_final_single}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
