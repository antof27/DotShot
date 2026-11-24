"""Behaviour-cloning warm start for HARL and MAT actors from the calibrated script.

The dataset (joint observations and script actions of every live agent) is
collected once per stage on dedicated seeds, disjoint from the training, val and
test seeds, and cached under ``runs/bc/``. Cloning maximises the log-likelihood
of the script actions under the learner's own action head, so the warm-started
policy keeps its architecture and is then fine-tuned with the usual RL update.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dotshot.envs.multi_agent import DotShotParallelEnv

BC_SEED_BASE = 5_000_003
CACHE_DIR = Path("runs") / "bc"


def collect_script_dataset(stage: int, n_episodes: int = 40, env_kwargs: Dict | None = None) -> Dict[str, np.ndarray]:
    """Joint arrays: obs (N, n_agents, D), actions (N, n_agents, 4), alive (N, n_agents)."""
    from benchmarks.runners.run_marl_baselines import CalibratedPolicy

    env = DotShotParallelEnv(stage=stage, **(env_kwargs or {}))
    policy = CalibratedPolicy()
    obs_rows, act_rows, alive_rows = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=BC_SEED_BASE + 7919 * i)
        while True:
            actions = policy.act(env, obs)
            names = env.possible_agents
            obs_rows.append(np.stack([obs[a] for a in names]))
            act_rows.append(np.stack([actions[a] for a in names]))
            alive_rows.append(np.array([env.engine.agents[k].alive for k in range(len(names))]))
            obs, _, terms, truncs, _ = env.step(actions)
            if any(terms.values()) or any(truncs.values()):
                break
    env.close()
    return {
        "obs": np.asarray(obs_rows, dtype=np.float32),
        "actions": np.asarray(act_rows, dtype=np.int64),
        "alive": np.asarray(alive_rows, dtype=bool),
    }


def load_dataset(stage: int, n_episodes: int = 40) -> Dict[str, np.ndarray]:
    path = CACHE_DIR / f"stage{stage}_calibrated_{n_episodes}ep.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = collect_script_dataset(stage, n_episodes)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.replace(path)  # atomic: parallel trials may race to build the cache
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def _fit(log_prob_fn, n: int, epochs: int, batch_size: int, params, lr: float, device) -> float:
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator().manual_seed(0)
    loss_val = float("nan")
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=gen).split(batch_size):
            loss = -log_prob_fn(idx.numpy()).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = float(loss.detach().cpu())
    return loss_val


def bc_harl_actor(runner, data: Dict[str, np.ndarray], epochs: int = 3, batch_size: int = 2048, lr: float = 1e-3) -> Dict[str, float]:
    """Clone the script into actor 0 (MLP) and copy it to every other actor."""
    obs = data["obs"][data["alive"]]
    actions = data["actions"][data["alive"]]
    actor = runner.actor[0]
    policy = actor.actor
    policy.train()
    hidden = runner.rnn_hidden_size
    zeros_rnn = np.zeros((batch_size, runner.recurrent_n, hidden), dtype=np.float32)
    ones = np.ones((batch_size, 1), dtype=np.float32)

    def log_prob(idx):
        lp, _, _ = policy.evaluate_actions(obs[idx], zeros_rnn[: len(idx)], actions[idx], ones[: len(idx)])
        return lp

    loss = _fit(log_prob, len(obs), epochs, batch_size, policy.parameters(), lr, runner.device)
    for other in runner.actor[1:]:
        if other is not actor:
            other.actor.load_state_dict(policy.state_dict())
    return {"bc/nll": loss, "bc/samples": float(len(obs))}


def bc_mat(policy, data: Dict[str, np.ndarray], epochs: int = 3, batch_size: int = 1024, lr: float = 5e-4) -> Dict[str, float]:
    """Clone the script's joint actions into MAT with teacher forcing (dead agents masked)."""
    obs, actions, alive = data["obs"], data["actions"], data["alive"]
    n, n_agents = obs.shape[0], obs.shape[1]
    policy.transformer.train()
    dummy_share = np.zeros((batch_size, n_agents, policy.share_obs_dim), dtype=np.float32)

    def log_prob(idx):
        b = len(idx)
        _, lp, _ = policy.evaluate_actions(
            dummy_share[:b].reshape(-1, policy.share_obs_dim), obs[idx].reshape(-1, policy.obs_dim),
            None, None, actions[idx].reshape(-1, len(policy.nvec)), None,
        )
        mask = torch.as_tensor(alive[idx].reshape(-1, 1), dtype=lp.dtype, device=lp.device)
        return lp * mask * (mask.numel() / mask.sum().clamp(min=1.0))

    loss = _fit(log_prob, n, epochs, batch_size, policy.transformer.parameters(), lr, policy.device)
    return {"bc/nll": loss, "bc/samples": float(alive.sum())}
