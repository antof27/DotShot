"""Resumable 1M-step multi-seed official SOTA benchmark suite for Milestone 3."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

STAGE3_ALGOS = ["harl_mappo", "harl_happo", "harl_hatrpo", "harl_haa2c", "official_mat"]
STAGE4_ALGOS = ["harl_mappo", "harl_happo", "harl_hatrpo", "harl_haa2c", "official_mat"]
SA_METRICS = ["eval/return_mean", "eval/win_rate", "eval/elemental_accuracy"]
MARL_METRICS = ["eval/return_mean", "eval/core_hp_ratio", "eval/elemental_accuracy"]


def _load_yaml(path: pathlib.Path) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _config(candidates: list[str], defaults: dict) -> dict:
    """Load the first existing candidate config, otherwise use the defaults."""
    for rel_path in candidates:
        path = ROOT / rel_path
        if path.is_file():
            return _load_yaml(path)
    return dict(defaults)


def _max_logged_step(run_dir: str | pathlib.Path) -> int:
    metrics_path = pathlib.Path(run_dir) / "metrics.jsonl"
    if not metrics_path.is_file():
        return -1
    max_step = -1
    with open(metrics_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                max_step = max(max_step, int(json.loads(line)["step"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return max_step


def _is_done(run_dir: str | pathlib.Path, target_steps: int) -> bool:
    """A run is resumable-complete only with a final checkpoint and full metrics."""
    run_path = pathlib.Path(run_dir)
    has_final = (
        (run_path / "checkpoints" / "final").is_dir()
        or (run_path / "checkpoints" / "final.zip").is_file()
    )
    return has_final and _max_logged_step(run_path) >= target_steps


def _metrics_done(run_dir: str | pathlib.Path, target_steps: int) -> bool:
    """Baselines have no checkpoint, so their logged metrics are the completion mark."""
    return _max_logged_step(run_dir) >= target_steps


def _run_single(
    cfg: dict,
    seed: int,
    device: str,
    steps: int,
    run_dir: pathlib.Path,
    init_model_path: str | None = None,
) -> dict:
    from benchmarks.runners.run_single_agent import run_experiment

    return run_experiment(
        cfg,
        seed=seed,
        device=device,
        steps_override=steps,
        run_dir_override=str(run_dir),
        init_model_path=init_model_path,
    )


def _best_stage1_checkpoint(stage1_dir: pathlib.Path) -> pathlib.Path:
    best_return = -float("inf")
    best_path: pathlib.Path | None = None
    for metrics_file in sorted((stage1_dir / "ppo").glob("seed_*/metrics.jsonl")):
        returns = []
        with open(metrics_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    returns.append(float(json.loads(line).get("eval/return_mean", float("-inf"))))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        if not returns:
            continue
        checkpoint = metrics_file.parent / "checkpoints" / "best.zip"
        if not checkpoint.is_file():
            checkpoint = metrics_file.parent / "checkpoints" / "final.zip"
        if not checkpoint.is_file():
            continue
        peak = max(returns)
        if peak > best_return:
            best_return = peak
            best_path = checkpoint
    if best_path is None:
        raise FileNotFoundError(
            f"No completed Stage 1 PPO checkpoint with eval/return_mean under {stage1_dir / 'ppo'}"
        )
    print(f"Best Stage 1 PPO checkpoint: {best_path} (eval/return_mean={best_return:.2f})", flush=True)
    return best_path


def run_stage12(args: argparse.Namespace) -> None:
    out_root = pathlib.Path(args.out_root)
    stage1_dir = out_root / "stage1"
    stage2_dir = out_root / "stage2"

    ppo1 = _config(
        ["configs/official/stage1_ppo.yaml", "configs/stage1_ppo.yaml"],
        {"stage": 1, "algo": "ppo", "exp_name": "stage1_ppo", "total_steps": 120_000},
    )
    ppo1.update({"stage": 1, "algo": "ppo"})

    for seed in args.seeds:
        run_dir = stage1_dir / "ppo" / f"seed_{seed}"
        if _is_done(run_dir, args.steps):
            print(f"[stage1 ppo seed={seed}] complete, skipping.", flush=True)
        else:
            _run_single(ppo1, seed, args.device, args.steps, run_dir)
        for algo in ("heuristic", "random"):
            baseline_cfg = _config(
                [f"configs/official/stage1_{algo}.yaml", f"configs/stage1_{algo}.yaml"],
                {"stage": 1, "algo": algo, "exp_name": f"stage1_{algo}"},
            )
            baseline_cfg.update({"stage": 1, "algo": algo, "n_eval_episodes": 16})
            baseline_dir = stage1_dir / algo / f"seed_{seed}"
            if _metrics_done(baseline_dir, args.steps):
                print(f"[stage1 {algo} seed={seed}] complete, skipping.", flush=True)
            else:
                _run_single(baseline_cfg, seed, args.device, args.steps, baseline_dir)

    _evaluate_stage(args.out_root, 1)
    stage1_best = _best_stage1_checkpoint(stage1_dir)

    ppo2 = _config(
        ["configs/official/stage2_ppo.yaml", "configs/stage2_ppo.yaml"],
        {"stage": 2, "algo": "ppo", "exp_name": "stage2_ppo", "total_steps": 150_000},
    )
    ppo2.update({"stage": 2, "algo": "ppo"})
    transfer_cfg = {**ppo2, "exp_name": "stage2_ppo_transfer"}
    transfer_official = ROOT / "configs/official/stage2_ppo_transfer.yaml"
    if transfer_official.is_file():
        transfer_cfg.update(_load_yaml(transfer_official))
    transfer_cfg.update({"stage": 2, "algo": "ppo", "exp_name": "stage2_ppo_transfer"})

    for seed in args.seeds:
        ppo_dir = stage2_dir / "ppo" / f"seed_{seed}"
        if _is_done(ppo_dir, args.steps):
            print(f"[stage2 ppo seed={seed}] complete, skipping.", flush=True)
        else:
            _run_single(ppo2, seed, args.device, args.steps, ppo_dir)

        transfer_dir = stage2_dir / "ppo_transfer" / f"seed_{seed}"
        if _is_done(transfer_dir, args.steps):
            print(f"[stage2 ppo_transfer seed={seed}] complete, skipping.", flush=True)
        else:
            _run_single(
                transfer_cfg,
                seed,
                args.device,
                args.steps,
                transfer_dir,
                init_model_path=str(stage1_best),
            )

        for algo in ("heuristic", "random"):
            baseline_cfg = _config(
                [f"configs/official/stage2_{algo}.yaml", f"configs/stage2_{algo}.yaml"],
                {"stage": 2, "algo": algo, "exp_name": f"stage2_{algo}"},
            )
            baseline_cfg.update({"stage": 2, "algo": algo, "n_eval_episodes": 16})
            baseline_dir = stage2_dir / algo / f"seed_{seed}"
            if _metrics_done(baseline_dir, args.steps):
                print(f"[stage2 {algo} seed={seed}] complete, skipping.", flush=True)
            else:
                _run_single(baseline_cfg, seed, args.device, args.steps, baseline_dir)

    _evaluate_stage(args.out_root, 2)


def _stage_config(stage: int, algo: str, steps: int) -> dict:
    reward_alpha = 0.65 if stage == 3 else 0.70
    if algo == "official_mat":
        defaults = {
            "stage": stage,
            "algo": "official_mat",
            "exp_name": f"stage{stage}_official_mat",
            "total_steps": steps,
            "episode_length": 200,
            "n_rollout_threads": 32,
            "n_eval_episodes": 10,
            "partial_obs": True,
            "reward_alpha": reward_alpha,
            "max_steps": None,
            "n_block": 1,
            "n_embd": 64,
            "n_head": 1,
            "hidden_size": 64,
            "lr": 5e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "ppo_epoch": 5,
            "entropy_coef": 0.01,
        }
        cfg = {**defaults, **_config([f"configs/official/stage{stage}_{algo}.yaml"], {})}
        cfg.update(
            {
                "stage": stage,
                "algo": "mat",
                "algorithm_name": "mat",
                "exp_name": f"stage{stage}_official_mat",
                "total_steps": steps,
                "eval_freq": max(25_000, steps // 10),
            }
        )
        return cfg

    base_algo = algo[len("harl_") :]
    defaults = {
        "stage": stage,
        "algo": algo,
        "exp_name": f"stage{stage}_{algo}",
        "total_steps": steps,
        "episode_length": 200,
        "n_rollout_threads": 32,
        "n_eval_episodes": 10,
        "partial_obs": True,
        "reward_alpha": reward_alpha,
        "max_steps": None,
        "hidden_sizes": [128, 128],
        "use_recurrent_policy": False,
        "state_type": "FP",
        "share_param": algo == "harl_mappo",
        "use_valuenorm": True,
        "use_linear_lr_decay": False,
        "lr": 5e-4,
        "critic_lr": 5e-4,
        "ppo_epoch": 5,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    }
    cfg = {
        **defaults,
        **_config(
            [
                f"configs/official/stage{stage}_{algo}.yaml",
                f"configs/stage{stage}_{base_algo}.yaml",
            ],
            {},
        ),
    }
    cfg.update(
        {
            "stage": stage,
            "algo": algo,
            "exp_name": f"stage{stage}_{algo}",
            "total_steps": steps,
            "n_rollout_threads": max(2, int(cfg.get("n_rollout_threads", 32))),
            "eval_freq": max(25_000, steps // 10),
        }
    )
    return cfg


def _worker_job(job: tuple[str, dict, int, str, str]) -> str:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    import torch

    torch.set_num_threads(1)

    algo, cfg, seed, device, run_dir = job
    if algo == "official_mat":
        from benchmarks.official.mat_runner import run_mat_experiment as train

        runner_label = "official_mat"
    else:
        from benchmarks.official.harl_runner import run_harl_experiment as train

        runner_label = algo

    started = time.perf_counter()
    result = train(cfg, seed=seed, device=device, run_dir_override=run_dir)
    elapsed = time.perf_counter() - started
    final = result.get("final_metrics", {})
    return (
        f"[{runner_label} seed={seed}] done in {elapsed:.1f}s | "
        f"Ret={final.get('eval/return_mean', float('nan')):.2f} | "
        f"Core={final.get('eval/core_hp_ratio', 0.0) * 100.0:.1f}% | "
        f"Win={final.get('eval/win_rate', 0.0) * 100.0:.1f}%"
    )


def run_stage(args: argparse.Namespace, stage: int) -> None:
    algos = STAGE3_ALGOS if stage == 3 else STAGE4_ALGOS
    stage_dir = pathlib.Path(args.out_root) / f"stage{stage}"
    jobs: list[tuple[str, dict, int, str, str]] = []

    for algo in algos:
        cfg = _stage_config(stage, algo, args.steps)
        for seed in args.seeds:
            run_dir = stage_dir / algo / f"seed_{seed}"
            if _is_done(run_dir, args.steps):
                print(f"[stage{stage} {algo} seed={seed}] complete, skipping.", flush=True)
                continue
            jobs.append((algo, cfg, seed, args.device, str(run_dir)))

    if jobs:
        total = len(jobs)
        started = time.perf_counter()
        print(
            f"Dispatching {total} Stage {stage} jobs across {max(1, args.workers)} workers...",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(_worker_job, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                print(f"({index}/{total}) {future.result()}", flush=True)
        print(f"Completed Stage {stage} jobs in {time.perf_counter() - started:.1f}s.", flush=True)

    _evaluate_stage(args.out_root, stage)


def plot_stage_comparison(
    stage_dir: str,
    out_path: str,
    metrics: list[str] | None = None,
    title: str | None = None,
) -> None:
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from benchmarks.eval.plot_results import plot_stage_curves

    plot_stage_curves(stage_dir, out_path, title or "DotShot Official Benchmark", metrics)


def _evaluate_stage(out_root: str, stage: int, n_boot: int = 2000) -> None:
    stage_dir = pathlib.Path(out_root) / f"stage{stage}"
    if not stage_dir.exists():
        print(f"[eval stage{stage}] no runs under {stage_dir}, skipping.", flush=True)
        return

    from benchmarks.eval.evaluate import summarize_stage

    summary, markdown = summarize_stage(str(stage_dir), n_boot=n_boot)
    results_dir = ROOT / "docs" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"m3_stage{stage}_official_eval.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({"summary": summary, "markdown": markdown}, file, indent=2)
    print(f"[eval stage{stage}] wrote {json_path}", flush=True)

    figure_path = ROOT / "docs" / "assets" / "figures" / f"m3_stage{stage}_official_converged.png"
    plot_metrics = SA_METRICS if stage in (1, 2) else MARL_METRICS
    try:
        plot_stage_comparison(
            str(stage_dir),
            str(figure_path),
            metrics=plot_metrics,
            title=f"DotShot Stage {stage} Official SOTA Convergence",
        )
        print(f"[eval stage{stage}] wrote {figure_path}", flush=True)
    except RuntimeError as exc:
        print(f"[eval stage{stage}] plot skipped: {exc}", flush=True)

    print(markdown, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable 1M-step multi-seed official Milestone 3 SOTA suite."
    )
    parser.add_argument(
        "--part",
        choices=["stage12", "stage3", "stage4", "eval", "all"],
        default="all",
        help="Suite section to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Environment steps per run.")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers for Stages 3 and 4.")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device (cuda or cpu).")
    parser.add_argument(
        "--out-root",
        type=str,
        default="runs/m3_official",
        help="Root directory for all suite run artifacts.",
    )
    args = parser.parse_args()

    if args.part in ("stage12", "all"):
        run_stage12(args)
    if args.part in ("stage3", "all"):
        run_stage(args, 3)
    if args.part in ("stage4", "all"):
        run_stage(args, 4)
    if args.part == "eval":
        for stage in (1, 2, 3, 4):
            _evaluate_stage(args.out_root, stage)


if __name__ == "__main__":
    main()
