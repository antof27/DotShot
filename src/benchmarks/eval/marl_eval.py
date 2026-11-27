"""Shared episode evaluator for every Stage 3-4 learner and scripted baseline.

All MARL numbers in the report come from this one function, so HARL, MAT and the
baselines use identical seeds, episode counts and metric definitions.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from dotshot.envs.multi_agent import DotShotParallelEnv

# Disjoint episode-seed sets. Training envs are seeded with small integers
# (seed + rank * 1000), so neither set overlaps them.
SEED_BASES = {"val": 1_000_003, "test": 2_000_003}

EPISODE_KEYS = (
    "win",
    "core_hp_ratio",
    "elemental_accuracy",
    "kills",
    "boss_kills",
    "shield_breaks",
    "synergy_combos",
    "episode_length",
    "hit_accuracy",
    "hold_frac",
    "move_frac",
    "shots_per_100",
    "deaths",
    "core_dist",
)


def episode_seeds(split: str, n_episodes: int) -> List[int]:
    base = SEED_BASES[split]
    return [base + 7919 * i for i in range(n_episodes)]


def run_episode(env: DotShotParallelEnv, policy: Any, seed: int) -> Dict[str, Any]:
    """Play one episode; ``policy`` has ``reset(env)`` and ``act(env, obs) -> {agent: action}``."""
    obs, _ = env.reset(seed=seed)
    policy.reset(env)
    episode_return = 0.0
    info: Dict[str, Any] = {}
    while True:
        obs, rewards, terms, truncs, infos = env.step(policy.act(env, obs))
        episode_return += float(np.mean([rewards.get(a, 0.0) for a in env.possible_agents]))
        info = infos[env.possible_agents[0]]
        if any(terms.values()) or any(truncs.values()):
            break
    record = {"seed": seed, "return": episode_return}
    record.update({k: float(info.get(k, 0.0)) for k in EPISODE_KEYS})
    record["reward_components"] = dict(info.get("reward_components", {}))
    return record


def evaluate_marl(
    env_kwargs: Dict[str, Any],
    policy: Any,
    n_episodes: int,
    split: str = "val",
    prefix: str = "eval",
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Run exactly ``n_episodes`` complete episodes on the given seed split."""
    env = DotShotParallelEnv(**env_kwargs)
    try:
        episodes = [run_episode(env, policy, s) for s in episode_seeds(split, n_episodes)]
    finally:
        env.close()
    return summarize_episodes(episodes, prefix), episodes


def summarize_episodes(episodes: List[Dict[str, Any]], prefix: str = "eval") -> Dict[str, float]:
    returns = np.array([e["return"] for e in episodes], dtype=np.float64)
    metrics = {
        f"{prefix}/n_episodes": float(len(episodes)),
        f"{prefix}/return_mean": float(returns.mean()),
        f"{prefix}/return_std": float(returns.std()),
        f"{prefix}/win_rate": float(np.mean([e["win"] for e in episodes])),
        f"{prefix}/core_hp_ratio": float(np.mean([e["core_hp_ratio"] for e in episodes])),
        f"{prefix}/elemental_accuracy": float(np.mean([e["elemental_accuracy"] for e in episodes])),
        f"{prefix}/kills_mean": float(np.mean([e["kills"] for e in episodes])),
        f"{prefix}/boss_kills_mean": float(np.mean([e["boss_kills"] for e in episodes])),
        f"{prefix}/shield_breaks_mean": float(np.mean([e["shield_breaks"] for e in episodes])),
        f"{prefix}/synergy_combos": float(np.mean([e["synergy_combos"] for e in episodes])),
        f"{prefix}/episode_length": float(np.mean([e["episode_length"] for e in episodes])),
    }
    for key in ("hit_accuracy", "hold_frac", "move_frac", "shots_per_100", "deaths", "core_dist"):
        metrics[f"{prefix}/{key}"] = float(np.mean([e[key] for e in episodes]))
    keys = sorted({k for e in episodes for k in e["reward_components"]})
    for key in keys:
        metrics[f"{prefix}/rc_{key}"] = float(
            np.mean([e["reward_components"].get(key, 0.0) for e in episodes])
        )
    return metrics
