"""Tables, JSON summaries and learning-curve figures for the final benchmark.

Learner rows: one final-policy `test` score per seed, summarised with mean,
median, IQM (n >= 4) and 95% bootstrap CIs over seeds. Baseline rows: the scripted
policies' per-episode `test` scores (`run_marl_baselines.py`, or the single-agent
`random` / `heuristic` runners). Curves plot the `val` return against simulator ticks
(decisions x action_repeat), mean and 95% bootstrap CI over seeds.

Usage:
    python scripts/report_final.py --stage 3 --runs runs/final/stage3 \
        --baselines runs/baselines_v22k8/stage3 --out docs/results/final_stage3.json \
        --figure docs/assets/figures/final_stage3_curves.png
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.eval.evaluate import compute_rliable_stats  # noqa: E402

METRICS = ("return_mean", "win_rate", "kills_mean", "elemental_accuracy", "core_hp_ratio",
           "episode_length", "hold_frac", "move_frac", "shots_per_100", "deaths")
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#000000", "#F0E442"]


def _rows(path: pathlib.Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _get(row: dict, prefix: str, metric: str):
    if metric == "kills_mean" and f"{prefix}/kills_mean" not in row:
        metric = "kills"  # single-agent runner
    return row.get(f"{prefix}/{metric}")


def learner_rows(runs_dir: pathlib.Path) -> dict:
    out = {}
    for cfg_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        finals, curves = [], []
        for mpath in sorted(cfg_dir.glob("seed_*/metrics.jsonl")):
            rows = _rows(mpath)
            test = [r for r in rows if "test/return_mean" in r]
            if not test:
                continue  # unfinished seed
            finals.append(test[-1])
            curves.append([(r["step"], r["eval/return_mean"]) for r in rows if "eval/return_mean" in r])
        if finals:
            out[cfg_dir.name] = {"finals": finals, "curves": curves,
                                 "repeat": _repeat(cfg_dir)}
    return out


def _repeat(cfg_dir: pathlib.Path) -> int:
    import yaml

    for cfg_path in cfg_dir.glob("seed_*/config.yaml"):
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return int((cfg.get("env_overrides") or {}).get("action_repeat", 8))
    return 8


def summarize(values) -> dict:
    vals = [v for v in values if v is not None]
    return compute_rliable_stats(vals) if vals else {"n": 0}


def baseline_rows(base_dir: pathlib.Path) -> dict:
    out = {}
    if not base_dir.exists():
        return out
    for p in sorted(base_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        eps = json.loads(p.read_text(encoding="utf-8"))["episodes"]
        row = {"n_episodes": len(eps)}
        for m in METRICS:
            key = {"return_mean": "return", "win_rate": "win", "kills_mean": "kills"}.get(m, m)
            vals = [e.get(key) for e in eps if e.get(key) is not None]
            if vals:
                row[m] = summarize(vals)
        out[p.stem] = row
    return out


def fmt(stats: dict, scale: float = 1.0, digits: int = 1, ci: bool = False) -> str:
    if not stats or not stats.get("n"):
        return "–"
    s = f"{stats['mean'] * scale:.{digits}f}"
    if ci and stats["n"] > 1:
        s += f" [{stats['mean_ci_low'] * scale:.{digits}f}, {stats['mean_ci_high'] * scale:.{digits}f}]"
    return s


def markdown(learners: dict, baselines: dict) -> str:
    lines = ["| Policy | n | Return mean [95% CI] | Median | IQM | Win (%) | Kills | Elem. (%) | Core HP (%) "
             "| Length | Hold | Move | Deaths |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, row in list(learners.items()) + [(f"*{k}*", v) for k, v in baselines.items()]:
        r = row["return_mean"]
        n = f"{r['n']} seeds" if not name.startswith("*") else f"{r['n']} ep."
        iqm = "–" if r.get("iqm") is None else f"{r['iqm']:.2f}"
        lines.append(
            f"| {name} | {n} | {fmt(r, digits=2, ci=True)} | {r['median']:.2f} | {iqm} "
            f"| {fmt(row.get('win_rate'), 100)} | {fmt(row.get('kills_mean'))} | {fmt(row.get('elemental_accuracy'), 100)} "
            f"| {fmt(row.get('core_hp_ratio'), 100)} | {fmt(row.get('episode_length'), digits=0)} "
            f"| {fmt(row.get('hold_frac'), digits=2)} | {fmt(row.get('move_frac'), digits=2)} | {fmt(row.get('deaths'))} |")
    return "\n".join(lines)


def plot(learners: dict, baselines: dict, title: str, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    for i, (name, data) in enumerate(learners.items()):
        steps = sorted({s for c in data["curves"] for s, _ in c})
        means, lo, hi, xs = [], [], [], []
        for s in steps:
            vals = [dict(c).get(s) for c in data["curves"]]
            vals = [v for v in vals if v is not None]
            if len(vals) < len(data["curves"]):
                continue  # only points every seed reached
            st = compute_rliable_stats(vals, n_boot=500)
            xs.append(s * data["repeat"] / 1e6)
            means.append(st["mean"])
            lo.append(st.get("mean_ci_low", st["mean"]))
            hi.append(st.get("mean_ci_high", st["mean"]))
        color = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(xs, means, color=color, lw=1.8, label=f"{name} (n={len(data['curves'])})")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
    for name, style in (("heuristic_calibrated", "--"), ("noop", ":")):
        if name in baselines:
            ax.axhline(baselines[name]["return_mean"]["mean"], color="0.35", ls=style, lw=1.1, label=name)
    ax.set_xlabel("Simulator ticks (millions)")
    ax.set_ylabel("Validation return (mean, 95% CI over seeds)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--baselines", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()

    raw = learner_rows(pathlib.Path(args.runs))
    learners = {name: {m: summarize([_get(f, "test", m) for f in d["finals"]]) for m in METRICS}
                for name, d in raw.items()}
    baselines = baseline_rows(pathlib.Path(args.baselines)) if args.baselines else {}
    md = markdown(learners, baselines)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(
        {"stage": args.stage, "learners": learners, "baselines": baselines, "markdown": md}, indent=2), encoding="utf-8")
    print(md)
    if args.figure:
        plot(raw, baselines, f"DotShot v2.2 Stage {args.stage}", pathlib.Path(args.figure))


if __name__ == "__main__":
    main()
