"""Multi-seed rliable evaluator (IQM, Mean, Median, 95% bootstrap CIs) and Markdown summary export."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_iqm(values: np.ndarray) -> Optional[float]:
    """Interquartile Mean (middle 50% mean), following Agarwal et al. (2021).

    Returns None for n < 4, where the middle 50% is not defined; callers must not
    label a plain mean as IQM.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(arr)
    if n < 4:
        return None
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(arr[lo:hi]))


def _bootstrap_ci(arr: np.ndarray, stat, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.array([stat(arr[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_rliable_stats(
    values: List[float] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Mean, median, IQM (n >= 4 only) and 95% percentile-bootstrap CIs with the real n.

    ``ci_low``/``ci_high`` bracket the IQM when n >= 4 and the mean otherwise
    (``ci_of`` says which); ``mean_ci_low``/``mean_ci_high`` always bracket the mean.
    With a single task, rliable's stratified bootstrap reduces to this resampling of runs.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = len(arr)
    if n == 0:
        return {"n": 0}
    out: Dict[str, Any] = {
        "n": n,
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "iqm": compute_iqm(arr),
    }
    if n == 1:
        v = float(arr[0])
        out.update(mean_ci_low=v, mean_ci_high=v, ci_low=v, ci_high=v, ci_of="mean")
        return out
    out["mean_ci_low"], out["mean_ci_high"] = _bootstrap_ci(arr, np.mean, n_boot, seed)
    if n >= 4:
        out["ci_low"], out["ci_high"] = _bootstrap_ci(arr, compute_iqm, n_boot, seed)
        out["ci_of"] = "iqm"
    else:
        out["ci_low"], out["ci_high"] = out["mean_ci_low"], out["mean_ci_high"]
        out["ci_of"] = "mean"
    return out


def load_stage_runs(stage_dir: str) -> Dict[str, List[List[Dict[str, Any]]]]:
    """Load metrics.jsonl histories grouped by algo -> list of seed trajectories."""
    grouped: Dict[str, List[List[Dict[str, Any]]]] = {}
    stage_path = Path(stage_dir)
    if not stage_path.exists():
        return grouped

    for algo_dir in sorted(p for p in stage_path.iterdir() if p.is_dir()):
        algo_name = algo_dir.name
        seed_trajs: List[List[Dict[str, Any]]] = []
        for metrics_file in sorted(algo_dir.glob("**/metrics.jsonl")):
            rows = []
            with open(metrics_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            if rows:
                seed_trajs.append(rows)
        if seed_trajs:
            grouped[algo_name] = seed_trajs
    return grouped


FINAL_METRICS = (
    "return_mean",
    "win_rate",
    "core_hp_ratio",
    "elemental_accuracy",
    "kills_mean",
    "synergy_combos",
    "episode_length",
)


def final_value(traj: List[Dict[str, Any]], metric: str) -> Optional[float]:
    """Held-out ``test/`` score of the final policy, else its last ``eval/`` score.

    Rows without the key are skipped, so a missing metric is reported as missing
    instead of a sentinel. ``eval/kills`` (single-agent runner) aliases ``kills_mean``.
    """
    for prefix in ("test", "eval"):
        names = [f"{prefix}/{metric}"]
        if metric == "kills_mean":
            names.append(f"{prefix}/kills")
        for row in reversed(traj):
            for name in names:
                if name in row:
                    return float(row[name])
    return None


def summarize_stage(stage_dir: str, n_boot: int = 2000) -> Tuple[Dict[str, Any], str]:
    """Aggregate final-policy metrics per algorithm (one value per seed) and render Markdown."""
    grouped = load_stage_runs(stage_dir)
    summary: Dict[str, Any] = {}
    for algo, trajs in grouped.items():
        algo_summary: Dict[str, Any] = {"num_seeds": len(trajs)}
        for metric in FINAL_METRICS:
            vals = [v for v in (final_value(tr, metric) for tr in trajs) if v is not None]
            if vals:
                algo_summary[metric] = compute_rliable_stats(vals, n_boot=n_boot)
        summary[algo] = algo_summary

    def fmt(stats: Optional[Dict[str, Any]], scale: float = 1.0, digits: int = 1) -> str:
        return "n/a" if not stats else f"{stats['mean'] * scale:.{digits}f}"

    lines = [
        "| Algorithm | Seeds | Final return mean [95% CI] | IQM | Win (%) | Core HP (%) | Elem. acc. (%) | Kills / ep | Ep. length |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for algo, data in summary.items():
        ret = data.get("return_mean")
        if not ret:
            continue
        iqm = "n/a (n<4)" if ret["iqm"] is None else f"{ret['iqm']:.2f}"
        lines.append(
            f"| `{algo}` | {data['num_seeds']} "
            f"| {ret['mean']:.2f} [{ret['mean_ci_low']:.2f}, {ret['mean_ci_high']:.2f}] | {iqm} "
            f"| {fmt(data.get('win_rate'), 100)} | {fmt(data.get('core_hp_ratio'), 100)} "
            f"| {fmt(data.get('elemental_accuracy'), 100)} | {fmt(data.get('kills_mean'))} "
            f"| {fmt(data.get('episode_length'), 1, 0)} |"
        )
    return summary, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed DotShot benchmark runs with rliable stats.")
    parser.add_argument("--stage-dir", type=str, required=True, help="Directory e.g. runs/stage3")
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to save eval_summary.json")
    parser.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for 95% CI")
    args = parser.parse_args()

    summary, md_table = summarize_stage(args.stage_dir, n_boot=args.n_boot)
    print("\n" + md_table + "\n")
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
