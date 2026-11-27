"""One-line status per Optuna study: trial states, best value, and the best trial's key settings.

Usage:
    python scripts/optuna_status.py [--db-dir runs/optuna]
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import optuna  # noqa: E402

from benchmarks.official.bohb_tune import STUDIES, best_per_category, load_study  # noqa: E402

KEYS = ("warm_start", "use_recurrent_policy", "lr", "entropy_coef", "hidden_size", "kl_threshold", "net_arch_size")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", type=pathlib.Path, default=ROOT / "runs" / "optuna")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    for name in STUDIES:
        study = load_study(name, args.db_dir)
        if study is None:
            continue
        states = [t.state.name[0] for t in study.trials]  # C(omplete) P(runed) R(unning) F(ail)
        counts = {s: states.count(s) for s in "CPRF"}
        line = f"{name:16s} C{counts['C']:>2} P{counts['P']:>2} R{counts['R']} F{counts['F']}"
        for tag, t in sorted(best_per_category(study).items()):
            short = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in t.params.items() if k in KEYS}
            line += f" | best {tag}: {t.value:7.2f} (#{t.number}) {short}"
        print(line)


if __name__ == "__main__":
    main()
