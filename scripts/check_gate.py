"""Phase gate check: learner (one final test score per seed) vs scripted baselines (per episode).

A learner "beats" a baseline when the lower end of its 95% bootstrap CI of the mean
lies above the upper end of the baseline's CI.

Usage:
    python scripts/check_gate.py --runs runs/phase1/stage3_harl_mappo_gate \
        --baselines runs/baselines_phase1/stage3 --min-kills 3
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.eval.evaluate import compute_rliable_stats, final_value  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="Directory with seed_*/metrics.jsonl")
    parser.add_argument("--baselines", required=True, help="Directory with <policy>.json episode files")
    parser.add_argument("--against", default="noop,charge")
    parser.add_argument("--min-kills", type=float, default=3.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trajs = []
    for path in sorted(pathlib.Path(args.runs).glob("seed_*/metrics.jsonl")):
        trajs.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])
    learner = {m: compute_rliable_stats([final_value(t, m) for t in trajs])
               for m in ("return_mean", "kills_mean", "episode_length", "win_rate", "core_hp_ratio")}
    report = {"runs": args.runs, "n_seeds": len(trajs), "learner": learner, "baselines": {}, "checks": {}}
    lo = learner["return_mean"]["mean_ci_low"]
    for name in args.against.split(","):
        data = json.loads((pathlib.Path(args.baselines) / f"{name}.json").read_text(encoding="utf-8"))
        eps = data["episodes"]
        ret = compute_rliable_stats([e["return"] for e in eps])
        length = compute_rliable_stats([e["episode_length"] for e in eps])
        report["baselines"][name] = {"return": ret, "episode_length": length}
        report["checks"][f"beats_{name}"] = bool(lo > ret["mean_ci_high"])
    charge_len = report["baselines"].get("charge", {}).get("episode_length", {}).get("mean", 0.0)
    report["checks"]["kills>=min"] = bool(learner["kills_mean"]["mean"] >= args.min_kills)
    report["checks"]["longer_than_charge"] = bool(learner["episode_length"]["mean"] > charge_len)
    report["passed"] = all(report["checks"].values())

    r = learner["return_mean"]
    print(f"learner n={len(trajs)}: return {r['mean']:.2f} [{r['mean_ci_low']:.2f}, {r['mean_ci_high']:.2f}], "
          f"kills {learner['kills_mean']['mean']:.2f}, length {learner['episode_length']['mean']:.0f}, "
          f"win {learner['win_rate']['mean']*100:.1f}%")
    for name, b in report["baselines"].items():
        print(f"  {name}: return {b['return']['mean']:.2f} [{b['return']['mean_ci_low']:.2f}, "
              f"{b['return']['mean_ci_high']:.2f}] (n={b['return']['n']}), length {b['episode_length']['mean']:.0f}")
    print("checks:", report["checks"], "PASSED" if report["passed"] else "FAILED")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
