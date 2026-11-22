"""Unified YAML-driven MARL training CLI (IPPO, VDN, QMIX, MAPPO, HAPPO, HATRPO, MAT, PMAT, RoMAT)."""

import argparse
import os
from typing import Any, Dict
import torch
import yaml

from benchmarks.marl.algos.happo import train_happo_family
from benchmarks.marl.algos.ippo import train_ippo
from benchmarks.marl.algos.mappo import train_shared_ppo
from benchmarks.marl.algos.mat import train_mat_family
from benchmarks.marl.algos.qmix import train_qmix
from benchmarks.marl.vec_marl_env import VecMARLEnv
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import seed_everything


def run_marl_experiment(cfg: Dict[str, Any], seed: int, device_str: str, steps_override: int | None = None) -> Dict[str, float]:
    torch.set_num_threads(1)
    seed_everything(seed)

    algo = cfg["algo"].lower()
    stage = int(cfg["stage"])
    total_steps = int(steps_override if steps_override is not None else cfg["total_steps"])
    num_envs = int(cfg.get("num_envs", 8))
    vec_type = cfg.get("vec_env", "subproc")
    partial_obs = cfg.get("partial_obs", None)
    reward_alpha = cfg.get("reward_alpha", None)
    max_steps = cfg.get("max_steps", None)
    exp_name = cfg.get("exp_name", f"stage{stage}_{algo}")

    resolved_cfg = {**cfg, "seed": seed, "device": device_str, "total_steps": total_steps}
    run_dir = os.path.join("runs", f"stage{stage}", algo, exp_name, f"seed_{seed}")
    logger = ExperimentLogger(run_dir, resolved_cfg)
    device = torch.device(device_str)

    vec_env = VecMARLEnv(
        stage=stage,
        num_envs=num_envs,
        seed=seed,
        vec_type=vec_type,
        partial_obs=partial_obs,
        reward_alpha=reward_alpha,
        max_steps=max_steps,
    )

    try:
        if algo == "ippo":
            return train_ippo(vec_env, resolved_cfg, logger, device, seed)
        if algo == "mappo":
            return train_shared_ppo("mappo", vec_env, resolved_cfg, logger, device, seed)
        if algo in ("vdn", "qmix"):
            return train_qmix(algo, vec_env, resolved_cfg, logger, device, seed)
        if algo in ("happo", "hatrpo"):
            return train_happo_family(algo, vec_env, resolved_cfg, logger, device, seed)
        if algo in ("mat", "pmat", "romat"):
            return train_mat_family(algo, vec_env, resolved_cfg, logger, device, seed)
        raise ValueError(f"Unsupported MARL algorithm: {algo}")
    finally:
        vec_env.close()
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="DotShot Game v2 Modular MARL Benchmark Trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch device")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for total environment steps")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device_str = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    run_marl_experiment(cfg, seed=args.seed, device_str=device_str, steps_override=args.steps)


if __name__ == "__main__":
    main()
