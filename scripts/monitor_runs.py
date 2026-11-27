"""Print the learning curves and behaviour audit of every run under a directory.

Flags policies that look degenerate rather than competent, so reward hacking is
caught while training is still running:
  IDLE     the agents barely move (move_frac < 0.15)
  NOFIRE   the agents rarely fire (hold_frac > 0.8 or shots_per_100 < 1)
  SUICIDE  more than 3 deaths per agent per 1000 ticks
  ELEM<33  wrong elements more often than chance

Usage:
    python scripts/monitor_runs.py runs/ab_repeat [--last 4]
"""

import argparse
import json
import pathlib


def flags(r: dict, prefix: str, n_agents: int) -> str:
    out = []
    if r.get(f"{prefix}/move_frac", 1.0) < 0.15:
        out.append("IDLE")
    if r.get(f"{prefix}/hold_frac", 0.0) > 0.8 or r.get(f"{prefix}/shots_per_100", 99.0) < 1.0:
        out.append("NOFIRE")
    length = max(1.0, r.get(f"{prefix}/episode_length", 1.0))
    if r.get(f"{prefix}/deaths", 0.0) / n_agents / length * 1000.0 > 3.0:
        out.append("SUICIDE")
    hits = r.get(f"{prefix}/elemental_accuracy")
    if hits is not None and r.get(f"{prefix}/kills_mean", r.get(f"{prefix}/kills", 0.0)) > 0.5 and hits < 0.33:
        out.append("ELEM<33")
    return " ".join(out)


def row(r: dict, prefix: str, n_agents: int) -> str:
    g = lambda k, d=0.0: r.get(f"{prefix}/{k}", d)  # noqa: E731
    kills = g("kills_mean", g("kills"))
    return (f"{r['step']:>9} {prefix:5s} ret {g('return_mean'):7.2f} win {g('win_rate')*100:5.1f} "
            f"kills {kills:5.1f} elem {g('elemental_accuracy')*100:5.1f} core {g('core_hp_ratio')*100:5.1f} "
            f"len {g('episode_length'):6.0f} | hold {g('hold_frac'):.2f} move {g('move_frac'):.2f} "
            f"shots/100 {g('shots_per_100'):5.1f} deaths {g('deaths'):4.1f}  {flags(r, prefix, n_agents)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--last", type=int, default=4)
    args = parser.parse_args()
    for path in sorted(pathlib.Path(args.root).glob("**/metrics.jsonl")):
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        stage = next((r.get("stage") for r in rows if "stage" in r), None)
        n_agents = {3: 2, 4: 3}.get(stage, 1)
        print(f"== {path.parent.relative_to(args.root)}")
        evals = [r for r in rows if "eval/return_mean" in r]
        for r in evals[-args.last:]:
            print("  " + row(r, "eval", n_agents))
        for r in rows:
            for prefix in ("test", "final_val"):
                if f"{prefix}/return_mean" in r:
                    print("  " + row(r, prefix, n_agents))


if __name__ == "__main__":
    main()
