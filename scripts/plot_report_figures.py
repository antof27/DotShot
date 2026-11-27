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


def _marl_groups(stage: int):
    """{(learner, kind): [seed dirs]} for the final Stage 3-4 runs; kind is 'bc' or 'scratch'."""
    out = {}
    for cfg in sorted((RUNS / f"final/stage{stage}").glob(f"s{stage}_*")):
        seeds = _seeds(cfg)
        if seeds:
            out[(learner_name(cfg.name), "scratch" if "scratch" in cfg.name else "bc")] = seeds
    return out


def _baseline(stage: int, name: str):
    p = RUNS / f"baselines_v22k8/stage{stage}/summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["results"][name]["metrics"]


def fig_final_marl(stage: int) -> None:
    groups = _marl_groups(stage)
    if not groups:
        return
    script = _baseline(stage, "heuristic_calibrated")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    for (name, kind), dirs in groups.items():
        ax = axes[0] if kind == "bc" else axes[1]
        xs, m, lo, hi = _band(dirs)
        ax.plot(xs, m, color=COLOR[name], ls="-" if kind == "bc" else "--", label=f"{name} (n={len(dirs)})")
        ax.fill_between(xs, lo, hi, color=COLOR[name], alpha=0.15, lw=0)
    for ax, title in zip(axes, ("Started from the cloned script", "From scratch")):
        if script:
            ax.axhline(script["eval/return_mean"], color=SCRIPT, ls=(0, (4, 3)), lw=1.1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Ticks (millions)")
        ax.legend(frameon=False, loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("Validation return")
    if script:
        axes[1].text(axes[1].get_xlim()[1], script["eval/return_mean"], " script", va="center", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(wspace=0.22)
    _save(fig, f"final_stage{stage}_curves")

    # test return and win rate per configuration, mean and 95% CI over seeds
    summary = RES / f"final_stage{stage}.json"
    if not summary.exists():
        return
    data = json.loads(summary.read_text())
    rows = sorted(data["learners"].items(), key=lambda kv: kv[1]["return_mean"]["mean"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.34 * len(rows) + 1.0), sharey=True)
    for i, (key, row) in enumerate(rows):
        name, kind = learner_name(key), ("scratch" if "scratch" in key else "cloned start")
        for ax, metric, scale in ((axes[0], "return_mean", 1), (axes[1], "win_rate", 100)):
            st = row[metric]
            ax.errorbar(st["mean"] * scale, i, xerr=[[(st["mean"] - st["mean_ci_low"]) * scale],
                                                    [(st["mean_ci_high"] - st["mean"]) * scale]],
                        fmt="o" if kind != "scratch" else "s", color=COLOR[name], ms=5, capsize=2, lw=1.4,
                        mfc=COLOR[name] if kind != "scratch" else "white")
    labels = [f"{learner_name(k)}, {'scratch' if 'scratch' in k else 'cloned start'}" for k, _ in rows]
    axes[0].set_yticks(range(len(rows)), labels)
    for ax, metric, scale, xl in ((axes[0], "return_mean", 1, "Test return"), (axes[1], "win_rate", 100, "Test win rate (%)")):
        for bname, ls in (("heuristic_calibrated", (0, (4, 3))), ("noop", ":")):
            b = data["baselines"].get(bname)
            if b:
                ax.axvline(b[metric]["mean"] * scale, color=SCRIPT, ls=ls, lw=1.0)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    axes[0].text(data["baselines"]["heuristic_calibrated"]["return_mean"]["mean"], len(rows) - 0.4, "script",
                 ha="center", color=MUTED, fontsize=7.5)
    axes[0].text(data["baselines"]["noop"]["return_mean"]["mean"], len(rows) - 0.4, "noop", ha="center",
                 color=MUTED, fontsize=7.5)
    fig.subplots_adjust(wspace=0.06)
    _save(fig, f"final_stage{stage}_scores")


COMPONENT_GROUPS = {
    "combat": ["hits_correct", "precision_hits", "kills", "boss_kills", "shield_breaks", "combo_kills",
               "team_kills", "team_boss_kills", "team_shield_breaks", "team_combo_kills", "team_wave_cleared"],
    "shaping": ["aim_shaping", "fire_element_credit"],
    "support": ["shield_absorbed", "heal_utility", "mark_utility", "team_core_healed"],
    "survival and victory": ["survival", "team_core_alive", "team_win_bonus"],
    "penalties": ["hits_wrong", "missed_bullets", "ally_hits", "ally_kills", "damage_taken", "proximity_danger",
                  "wall_bump", "death", "team_core_damage", "team_core_destroyed", "team_agent_died"],
}
GROUP_COLOR = {"combat": "#2a78d6", "shaping": "#eda100", "support": "#1baf7a", "survival and victory": "#6250d6",
               "penalties": "#e34948"}


def fig_components(stage: int) -> None:
    groups = _marl_groups(stage)
    if not groups:
        return
    rows = []
    for (name, kind), dirs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        tests = [[json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if "test/return_mean" in l][-1]
                 for d in dirs]
        comp = {g: np.mean([sum(t.get(f"test/rc_{k}", 0.0) for k in keys) for t in tests])
                for g, keys in COMPONENT_GROUPS.items()}
        rows.append((f"{name}, {'scratch' if kind == 'scratch' else 'cloned'}", comp))
    script = _baseline(stage, "heuristic_calibrated")
    if script:
        rows.append(("scripted reference", {g: sum(script.get(f"eval/rc_{k}", 0.0) for k in keys)
                                            for g, keys in COMPONENT_GROUPS.items()}))
    fig, ax = plt.subplots(figsize=(7.4, 0.36 * len(rows) + 1.1))
    used = [g for g in COMPONENT_GROUPS if any(abs(c[g]) > 0.05 for _, c in rows)]
    for i, (label, comp) in enumerate(rows):
        pos, neg = 0.0, 0.0
        for g in used:
            v = comp[g]
            left = pos if v >= 0 else neg + v
            ax.barh(i, abs(v), left=left, color=GROUP_COLOR[g], height=0.62, edgecolor="white", linewidth=1.0,
                    label=g if i == 0 else None)
            if v >= 0:
                pos += v
            else:
                neg += v
        ax.text(pos + 1, i, f"{sum(comp.values()):.1f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Mean test return by reward group (net total at the right)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=7.5)
    _save(fig, f"final_stage{stage}_components")


def fig_behaviour(stage: int) -> None:
    beh = RES / "behaviour"
    files = sorted(beh.glob(f"s{stage}_*.json"))
    if not files:
        return
    data = {f.stem: json.loads(f.read_text()) for f in files}
    order = sorted(data, key=lambda k: (k != f"s{stage}_script", "scratch" in k, k))

    def label(k):
        return "scripted reference" if k.endswith("_script") else \
            f"{learner_name(k)}, {'scratch' if 'scratch' in k else 'cloned'}"

    # occupancy: all agents of a team summed; arena drawn to scale with its obstacles and the Core
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Circle, Rectangle

    from dotshot.core.config import get_stage_config

    cfg = get_stage_config(stage)
    n = len(order)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.85 * ncols, 1.45 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, k in zip(axes, order):
        h = np.array(data[k]["heatmap"]).sum(0)
        ax.imshow(h / h.sum(), cmap="Blues", origin="upper", extent=(0, cfg.width, cfg.height, 0),
                  norm=PowerNorm(0.45, vmin=0, vmax=(h / h.sum()).max()))
        for o in cfg.obstacles:
            ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, fill=False, ec=MUTED, lw=0.7))
        ax.add_patch(Circle((cfg.width / 2, cfg.height / 2), 38, fc="#e34948", ec="white", lw=0.6))
        ax.set_xlim(0, cfg.width), ax.set_ylim(cfg.height, 0)
        ax.set_title(label(k), fontsize=7.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(GRID)
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    _save(fig, f"behaviour_stage{stage}_heatmaps")

    # Core HP over the episode, and weapon choice given the target element (cloned vs scratch)
    cloned = [k for k in order if not k.endswith("_script") and "scratch" not in k]
    scratch = [k for k in order if "scratch" in k]
    picks = [x for x in (cloned[:1] + scratch[:1]) if x]
    fig, axes = plt.subplots(1, 1 + len(picks), figsize=(7.6, 2.6),
                             gridspec_kw={"width_ratios": [1.35] + [1] * len(picks)})
    for k in order:
        d = data[k]
        m = np.array(d["core_hp_curve"]["mean"]) * 100
        xs = np.arange(1, len(m) + 1) / len(m) * 100
        name = learner_name(k) if not k.endswith("_script") else None
        axes[0].plot(xs, m, color=COLOR.get(name, SCRIPT), ls="--" if ("scratch" in k or name is None) else "-",
                     lw=1.6 if name else 1.2, label=label(k))
    axes[0].set_xlabel("Episode progress (%)")
    axes[0].set_ylabel("Core HP (%), 16 episodes")
    axes[0].legend(handles=[plt.Line2D([], [], color=SCRIPT, ls="--", lw=1.2, label="scripted reference"),
                            plt.Line2D([], [], color=MUTED, lw=1.6, label="cloned start (learner colour)"),
                            plt.Line2D([], [], color=MUTED, ls="--", lw=1.6, label="scratch (learner colour)")],
                   frameon=False, fontsize=6.5, loc="lower left")
    for ax, pick in zip(axes[1:], picks):
        conf = np.array(data[pick]["confusion"]).sum(0)
        shares = conf / np.maximum(conf.sum(1, keepdims=True), 1) * 100
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(4), ["hold", "W", "G", "F"])
        ax.set_yticks(range(3), ["Water", "Grass", "Fire"])
        ax.set_xlabel("Weapon chosen")
        ax.grid(False)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, f"{shares[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if shares[i, j] > 55 else INK)
        ax.set_title(label(pick), loc="left", fontsize=8)
    axes[1].set_ylabel("Enemy on the aim ray")
    fig.subplots_adjust(wspace=0.55)
    _save(fig, f"behaviour_stage{stage}_core_weapons")


def fig_roles(stage: int = 4) -> None:
    """Stage 4: per-role position maps (script vs the best cloned learner) and per-role statistics."""
    beh = RES / "behaviour"
    files = {f.stem: json.loads(f.read_text()) for f in sorted(beh.glob(f"s{stage}_*.json"))}
    if f"s{stage}_script" not in files or len(files) < 2:
        return
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Circle, Rectangle

    from dotshot.core.config import get_stage_config

    cfg = get_stage_config(stage)
    final = json.loads((RES / f"final_stage{stage}.json").read_text())["learners"]
    best = max((k for k in final if "scratch" not in k), key=lambda k: final[k]["return_mean"]["mean"])
    pair = [(f"s{stage}_script", "scripted reference"), (best, f"{learner_name(best)}, cloned")]
    roles = files[f"s{stage}_script"]["roles"]
    fig, axes = plt.subplots(2, len(roles), figsize=(6.0, 2.9))
    for r, (key, title) in enumerate(pair):
        heat = np.array(files[key]["heatmap"])
        for c, role in enumerate(roles):
            ax = axes[r, c]
            h = heat[c] / heat[c].sum()
            ax.imshow(h, cmap="Blues", origin="upper", extent=(0, cfg.width, cfg.height, 0),
                      norm=PowerNorm(0.45, vmin=0, vmax=h.max()))
            for o in cfg.obstacles:
                ax.add_patch(Rectangle((o.x, o.y), o.w, o.h, fill=False, ec=MUTED, lw=0.6))
            ax.add_patch(Circle((cfg.width / 2, cfg.height / 2), 38, fc="#e34948", ec="white", lw=0.6))
            ax.set_xlim(0, cfg.width), ax.set_ylim(cfg.height, 0)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True), s.set_color(GRID)
            if r == 0:
                ax.set_title(role, fontsize=8.5)
            if c == 0:
                ax.set_ylabel(title, fontsize=7.5)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    _save(fig, f"behaviour_stage{stage}_roles_maps")

    # per-role distance to the Core and ability uses per episode, every analysed policy
    keys = [f"s{stage}_script"] + sorted(k for k in files if k != f"s{stage}_script")
    labels = ["script" if k.endswith("_script") else
              f"{learner_name(k)}{' (scratch)' if 'scratch' in k else ''}" for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    # roles are a different entity from learners, so they take the palette's later slots
    role_color = {"Vanguard": "#6250d6", "Striker": "#e34948", "Support": "#008300"}
    for ax, metric, rset in ((axes[0], "dist", roles), (axes[1], "abil", [r for r in roles if r != "Striker"])):
        width = 0.8 / len(rset)
        for j, role in enumerate(rset):
            idx = roles.index(role)
            vals = [(files[k]["core_distance"]["mean"][idx] or 0) if metric == "dist"
                    else files[k]["abilities_per_episode"][idx] for k in keys]
            xs = np.arange(len(keys)) + (j - (len(rset) - 1) / 2) * width
            ax.bar(xs, vals, width=width * 0.9, color=role_color[role], label=role)
    for ax, yl in zip(axes, ("Mean distance to the Core (px)", "Ability uses per episode")):
        ax.set_xticks(range(len(keys)), labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylabel(yl)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(wspace=0.3)
    _save(fig, f"behaviour_stage{stage}_roles_stats")


FIGURES = {"roles": fig_roles, "learnability": fig_learnability, "action_repeat": fig_action_repeat, "tuning": fig_tuning,
           "final_single": fig_final_single,
           "final_marl": lambda: [fig_final_marl(s) for s in (3, 4)],
           "components": lambda: [fig_components(s) for s in (3, 4)],
           "behaviour": lambda: [fig_behaviour(s) for s in (3, 4)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = parser.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
