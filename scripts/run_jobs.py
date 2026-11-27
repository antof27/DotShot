"""Resumable multi-seed launcher for SB3, HARL and MAT runs.

Each (config, seed) pair trains in its own process. A run counts as complete
once its ``metrics.jsonl`` holds the final held-out ``test/return_mean`` row;
complete runs are skipped, incomplete ones are deleted and restarted.

Usage:
    python scripts/run_jobs.py --config configs/official/stage3_harl_mappo.yaml \
        --seeds 0-4 --out runs/official --workers 5 --device cpu
"""

import argparse
import json
import pathlib
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def is_complete(run_dir: pathlib.Path) -> bool:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        return False
    with open(path, "r", encoding="utf-8") as f:
        return any("test/return_mean" in line for line in f)


def run_one(cfg: dict, seed: int, device: str, run_dir: str) -> dict:
    """Dispatch one training run to the matching runner (executes in a worker process)."""
    algo = str(cfg["algo"]).lower()
    if algo.startswith("harl_"):
        from benchmarks.official.harl_runner import run_harl_experiment

        return run_harl_experiment(cfg, seed=seed, device=device, run_dir_override=run_dir)
    if algo in ("mat", "mat_dec", "official_mat"):
        from benchmarks.official.mat_runner import run_mat_experiment

        return run_mat_experiment({**cfg, "algo": "mat" if algo == "official_mat" else algo},
                                  seed=seed, device=device, run_dir_override=run_dir)
    from benchmarks.runners.run_single_agent import run_experiment

    return run_experiment(cfg, seed=seed, device=device, run_dir_override=run_dir)


def _worker(job: tuple) -> str:
    cfg, seed, device, run_dir = job
    start = time.time()
    try:
        result = run_one(cfg, seed, device, run_dir)
        return f"done {run_dir} final_return={result['final_return']:.3f} ({(time.time() - start) / 60:.1f} min)"
    except Exception:  # noqa: BLE001 - report and keep the other jobs running
        return f"FAILED {run_dir}\n{traceback.format_exc()}"


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", action="append", required=True, help="YAML config (repeatable)")
    parser.add_argument("--seeds", default="0-4")
    parser.add_argument("--out", default="runs/official", help="Root; runs go to <out>/<config stem>/seed_<k>")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--set", action="append", default=[], help="Override, e.g. --set total_steps=2000000")
    args = parser.parse_args()

    jobs = []
    for cfg_path in args.config:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for item in args.set:
            key, value = item.split("=", 1)
            cfg[key] = yaml.safe_load(value)
        cfg["config_path"] = pathlib.Path(cfg_path).as_posix()
        for seed in parse_seeds(args.seeds):
            # "{seed}" in string values pairs runs across stages, e.g. Stage 2 seed k
            # warm-starts from Stage 1 seed k's checkpoint.
            seed_cfg = {k: v.format(seed=seed) if isinstance(v, str) and "{seed}" in v else v for k, v in cfg.items()}
            run_dir = pathlib.Path(args.out) / pathlib.Path(cfg_path).stem / f"seed_{seed}"
            if is_complete(run_dir):
                print(f"skip (complete) {run_dir}", flush=True)
                continue
            if run_dir.exists():
                shutil.rmtree(run_dir)  # partial run from an interrupted launch
            jobs.append((seed_cfg, seed, args.device, str(run_dir)))

    print(f"{len(jobs)} job(s), {args.workers} worker(s), device={args.device}", flush=True)
    if not jobs:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_worker, job) for job in jobs]):
            print(fut.result(), flush=True)
    summary = {
        str(p.parent.name) + "/" + p.name: is_complete(p)
        for p in sorted({pathlib.Path(j[3]) for j in jobs})
    }
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
