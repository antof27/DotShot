"""Shared MARL evaluation and checkpoint utilities."""

from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from dotshot.envs.multi_agent import DotShotParallelEnv


def evaluate_marl_policy(
    stage: int,
    act_fn: Callable[[np.ndarray], np.ndarray],
    n_episodes: int,
    seed: int,
    partial_obs: Optional[bool] = None,
    reward_alpha: Optional[float] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a deterministic MARL policy over n_episodes."""
    env = DotShotParallelEnv(
        stage=stage,
        partial_obs=partial_obs,
        reward_alpha=reward_alpha,
        max_steps=max_steps,
    )
    names = list(env.possible_agents)

    returns: List[float] = []
    wins: List[float] = []
    core_hps: List[float] = []
    kills: List[float] = []
    elem_accs: List[float] = []
    hit_accs: List[float] = []
    combos: List[float] = []

    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=seed + ep * 97)
        ep_ret = 0.0
        last_info: Dict[str, Any] = {}
        while env.agents:
            obs_arr = np.stack([obs_dict[a] for a in names], axis=0)
            act_arr = act_fn(obs_arr)
            actions = {a: act_arr[i] for i, a in enumerate(env.agents)}
            obs_dict, rewards, _, _, infos = env.step(actions)
            ep_ret += float(np.mean([rewards.get(a, 0.0) for a in names]))
            if infos:
                last_info = next(iter(infos.values()))

        returns.append(ep_ret)
        wins.append(float(last_info.get("win", 0.0)))
        core_hps.append(float(last_info.get("core_hp_ratio", 0.0)))
        kills.append(float(last_info.get("kills", 0.0)))
        elem_accs.append(float(last_info.get("elemental_accuracy", 0.0)))
        hit_accs.append(float(last_info.get("hit_accuracy", 0.0)))
        combos.append(float(last_info.get("synergy_combos", 0.0)))

    env.close()
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/win_rate": float(np.mean(wins)),
        "eval/core_hp_ratio": float(np.mean(core_hps)),
        "eval/kills": float(np.mean(kills)),
        "eval/elemental_accuracy": float(np.mean(elem_accs)),
        "eval/hit_accuracy": float(np.mean(hit_accs)),
        "eval/synergy_combos": float(np.mean(combos)),
    }


def save_marl_checkpoint(
    path: str,
    algo: str,
    stage: int,
    num_agents: int,
    obs_dim: int,
    state_dim: int,
    hidden_dim: int,
    models: Dict[str, torch.nn.Module],
) -> None:
    payload = {
        "algo": algo,
        "stage": stage,
        "num_agents": num_agents,
        "obs_dim": obs_dim,
        "state_dim": state_dim,
        "hidden_dim": hidden_dim,
        "models": {k: v.state_dict() for k, v in models.items()},
    }
    torch.save(payload, path)
