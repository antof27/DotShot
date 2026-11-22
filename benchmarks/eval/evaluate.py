"""Multi-seed rliable evaluator (IQM, Mean, Median, 95% bootstrap CIs) and Markdown summary export."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


def compute_iqm(values: np.ndarray) -> float:
    """Interquartile Mean (middle 50% mean), following Agarwal et al. (2021)."""
    arr = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = len(arr)
    if n == 0:
        return 0.0
    if n < 4:
        return float(np.mean(arr))
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    return float(np.mean(arr[lo:hi]))


def compute_rliable_stats(
    values: List[float] | np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Compute Mean, Median, IQM, and 95% bootstrap CI around IQM."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "iqm": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if len(arr) == 1:
        v = float(arr[0])
        return {"mean": v, "std": 0.0, "median": v, "iqm": v, "ci_low": v, "ci_high": v}

    rng = np.random.default_rng(seed)
    boot_iqms = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for b in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_iqms[b] = compute_iqm(sample)

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "iqm": compute_iqm(arr),
        "ci_low": float(np.percentile(boot_iqms, 2.5)),
        "ci_high": float(np.percentile(boot_iqms, 97.5)),
    }


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


def summarize_stage(stage_dir: str, n_boot: int = 2000) -> Tuple[Dict[str, Any], str]:
    """Aggregate final-checkpoint metrics per algorithm and render a Markdown table."""
    grouped = load_stage_runs(stage_dir)
    summary: Dict[str, Any] = {}
    metrics_keys = [
        "eval/return_mean",
        "eval/win_rate",
        "eval/core_hp_ratio",
        "eval/elemental_accuracy",
        "eval/kills",
        "eval/synergy_combos",
    ]

    for algo, trajs in grouped.items():
        algo_summary: Dict[str, Any] = {"num_seeds": len(trajs)}
        for mk in metrics_keys:
            final_vals = [float(tr[-1].get(mk, 0.0)) for tr in trajs]
            best_vals = [float(max(r.get(mk, -1e9) for r in tr)) for tr in trajs]
            algo_summary[mk] = {
                "final": compute_rliable_stats(final_vals, n_boot=n_boot),
                "best": compute_rliable_stats(best_vals, n_boot=n_boot),
            }
        summary[algo] = algo_summary

    lines = [
        "| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for algo, data in summary.items():
        f_ret = data["eval/return_mean"]["final"]
        b_ret = data["eval/return_mean"]["best"]
        win = data["eval/win_rate"]["final"]["iqm"] * 100.0
        core = data["eval/core_hp_ratio"]["final"]["iqm"] * 100.0
        elem = data["eval/elemental_accuracy"]["final"]["iqm"] * 100.0
        kills = data["eval/kills"]["final"]["iqm"]
        lines.append(
            f"| `{algo.upper()}` | {data['num_seeds']} "
            f"| {f_ret['iqm']:.2f} [{f_ret['ci_low']:.2f}, {f_ret['ci_high']:.2f}] "
            f"| {b_ret['iqm']:.2f} [{b_ret['ci_low']:.2f}, {b_ret['ci_high']:.2f}] "
            f"| {win:.1f}% | {core:.1f}% | {elem:.1f}% | {kills:.1f} |"
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
