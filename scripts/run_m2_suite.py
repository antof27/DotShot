"""Reproducible Milestone 2 MARL sweep and ablation orchestrator (Windows dummy VecEnv + parallel workers)."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _load_cfg(path: str, steps: int, eval_freq: int, n_eval: int) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["vec_env"] = "dummy"
    cfg["total_steps"] = steps
    cfg["eval_freq"] = eval_freq
    cfg["n_eval_episodes"] = n_eval
    return cfg


def _worker_job(job: Tuple[str, Dict, int, str, Optional[str]]) -> str:
    label, cfg, seed, device, run_dir = job
    # Import inside spawned worker so Windows multiprocessing stays clean
    from benchmarks.marl.train_marl import run_marl_experiment

    t0 = time.perf_counter()
    metrics = run_marl_experiment(cfg, seed=seed, device_str=device, run_dir_override=run_dir)
    dt = time.perf_counter() - t0
    ret = metrics.get("eval/return_mean", float("nan"))
    core = metrics.get("eval/core_hp_ratio", 0.0) * 100.0
    elem = metrics.get("eval/elemental_accuracy", 0.0) * 100.0
    return f"[{label} seed={seed}] done in {dt:.1f}s | Ret={ret:.2f} | Core={core:.1f}% | ElemAcc={elem:.1f}%"


def _build_stage3_jobs(device: str, steps: int = 60_000) -> List[Tuple]:
    jobs = []
    for algo in ["ippo", "qmix", "mappo", "happo", "mat"]:
        cfg = _load_cfg(f"configs/stage3_{algo}.yaml", steps=steps, eval_freq=20_000, n_eval=6)
        for seed in range(5):
            jobs.append((f"S3-{algo.upper()}", cfg, seed, device, None))
    return jobs


def _build_stage4_jobs(device: str, steps: int = 60_000) -> List[Tuple]:
    jobs = []
    for algo in ["ippo", "mappo", "happo", "romat"]:
        cfg = _load_cfg(f"configs/stage4_{algo}.yaml", steps=steps, eval_freq=20_000, n_eval=6)
        for seed in range(5):
            jobs.append((f"S4-{algo.upper()}", cfg, seed, device, None))
    return jobs


def _build_ablation_jobs(device: str, steps: int = 50_000) -> List[Tuple]:
    jobs = []
    base_cfg = _load_cfg("configs/stage3_mappo.yaml", steps=steps, eval_freq=25_000, n_eval=6)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        label = f"alpha_{alpha:.2f}".replace(".", "_")
        for seed in range(3):
            cfg = {**base_cfg, "reward_alpha": alpha, "exp_name": label}
            run_dir = os.path.join("runs", "ablation_alpha", label, f"seed_{seed}")
            jobs.append((f"ABL-A={alpha:.2f}", cfg, seed, device, run_dir))

    for los_on, label in [(True, "los_on"), (False, "los_off")]:
        for seed in range(3):
            cfg = {**base_cfg, "partial_obs": los_on, "exp_name": label}
            run_dir = os.path.join("runs", "ablation_los", label, f"seed_{seed}")
            jobs.append((f"ABL-{label}", cfg, seed, device, run_dir))
    return jobs


def run_jobs(jobs: List[Tuple], workers: int) -> None:
    total = len(jobs)
    t_start = time.perf_counter()
    print(f"Dispatching {total} MARL jobs across {workers} parallel workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker_job, job) for job in jobs]
        for idx, fut in enumerate(as_completed(futures), start=1):
            summary = fut.result()
            print(f"({idx}/{total}) {summary}", flush=True)
    print(f"Completed {total} jobs in {time.perf_counter() - t_start:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone 2 5-seed MARL sweeps and ablations.")
    parser.add_argument("--part", choices=["all", "stage3", "stage4", "ablations"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    jobs: List[Tuple] = []
    if args.part in ("all", "stage3"):
        jobs.extend(_build_stage3_jobs(args.device))
    if args.part in ("all", "stage4"):
        jobs.extend(_build_stage4_jobs(args.device))
    if args.part in ("all", "ablations"):
        jobs.extend(_build_ablation_jobs(args.device))

    run_jobs(jobs, workers=args.workers)


if __name__ == "__main__":
    main()
