"""Rebuild a trained policy from a run directory for replays and analysis.

``load_policy(run_dir)`` reads ``config.yaml`` and the ``final`` (or ``best``)
checkpoint and returns an object with the ``marl_eval`` policy interface
(``reset(env)``, ``act(env, obs) -> {agent: action}``) for HARL and MAT runs,
or a single-agent ``act(env, obs) -> action`` wrapper for SB3 runs.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


def _env_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": int(cfg.get("stage", 3)),
        "partial_obs": cfg.get("partial_obs"),
        "reward_alpha": cfg.get("reward_alpha"),
        "max_steps": cfg.get("max_steps"),
        "reward_mode": cfg.get("reward_mode"),
        "env_overrides": cfg.get("env_overrides"),
    }


class SB3Policy:
    """Deterministic SB3 policy for the single-agent env."""

    def __init__(self, model):
        self.model = model

    def reset(self, env) -> None:
        pass

    def act(self, env, obs) -> np.ndarray:
        return self.model.predict(obs, deterministic=True)[0]


def _load_harl(cfg: Dict[str, Any], ckpt: Path, device: torch.device):
    from harl.envs.env_wrappers import ShareDummyVecEnv

    from benchmarks.official.harl_dotshot_env import DotShotHARLEnv
    from benchmarks.official.harl_runner import (
        HARLEvalPolicy,
        _base_algo_name,
        _DotShotHARLRunner,
        build_harl_args,
        load_actor_weights,
    )

    algo = _base_algo_name(cfg["algo"])
    # Two placeholder envs: HARL's vec wrapper needs at least two; weights come from the checkpoint.
    build_cfg = {k: v for k, v in cfg.items() if k not in ("warm_start", "init_actor_dir", "harl_args")}
    build_cfg["n_rollout_threads"] = 2
    args = build_harl_args(build_cfg, algo)
    state_type = str(cfg.get("state_type", "EP"))
    env_kwargs = _env_kwargs(cfg)
    envs = ShareDummyVecEnv(
        [lambda r=r: DotShotHARLEnv(seed=r, state_type=state_type, **env_kwargs) for r in range(2)]
    )
    runner = _DotShotHARLRunner(algo, args, state_type, envs, num_agents=envs.n_agents, device=device)
    load_actor_weights(runner, ckpt)
    envs.close()
    return HARLEvalPolicy(runner)


def _load_mat(cfg: Dict[str, Any], ckpt: Path, device: torch.device):
    from benchmarks.official.mat_runner import MATEvalPolicy, _build_args  # first: puts upstream MAT on sys.path
    from benchmarks.official.mat_discrete import DotShotMATPolicy
    from benchmarks.official.mat_dotshot_env import DotShotMATEnv

    args = _build_args(cfg)
    args.dec_actor = args.algorithm_name == "mat_dec" or args.dec_actor
    env = DotShotMATEnv(seed=0, **_env_kwargs(cfg))
    policy = DotShotMATPolicy(
        args, env.observation_space[0], env.share_observation_space[0], env.action_space[0],
        env.n_agents, device=device,
    )
    policy.transformer.load_state_dict(torch.load(ckpt / "transformer.pt", map_location=device))
    return MATEvalPolicy(policy)


def load_policy(run_dir: str, which: str = "final", device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Return ``(policy, env_kwargs)`` for the run in ``run_dir``."""
    run = Path(run_dir)
    with open(run / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    algo = str(cfg["algo"]).lower()
    ckpt = run / "checkpoints" / which
    dev = torch.device(device)
    if algo.startswith("harl_"):
        return _load_harl(cfg, ckpt, dev), _env_kwargs(cfg)
    if algo in ("mat", "mat_dec"):
        return _load_mat(cfg, ckpt, dev), _env_kwargs(cfg)
    from stable_baselines3 import PPO

    path = ckpt if ckpt.suffix == ".zip" else run / "checkpoints" / f"{which}.zip"
    return SB3Policy(PPO.load(str(path), device=device)), {"stage": int(cfg["stage"]), "env_overrides": cfg.get("env_overrides")}
