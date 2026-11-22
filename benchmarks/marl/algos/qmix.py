"""Off-policy Branching-VDN and Branching-QMIX value factorization trainers."""

import copy
import time
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from benchmarks.marl.algos.common import evaluate_marl_policy, save_marl_checkpoint
from benchmarks.marl.buffer import EpisodeReplayBuffer
from benchmarks.marl.networks import (
    ACTION_BRANCHES,
    BranchingQNetwork,
    QMIXMixer,
    VDNMixer,
)


def train_qmix(
    algo: str,
    vec_env: Any,
    cfg: Dict[str, Any],
    logger: Any,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Train Branching-VDN or Branching-QMIX with Double Q-learning and target networks."""
    assert algo in ("vdn", "qmix")
    stage = int(cfg["stage"])
    total_steps = int(cfg["total_steps"])
    chunk_len = int(cfg.get("chunk_len", 32))
    buffer_capacity = int(cfg.get("buffer_capacity", 2000))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 5e-4))
    gamma = float(cfg.get("gamma", 0.99))
    tau = float(cfg.get("tau", 0.005))
    eps_start = float(cfg.get("eps_start", 1.0))
    eps_end = float(cfg.get("eps_end", 0.05))
    eps_decay_frac = float(cfg.get("eps_decay_frac", 0.5))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    eval_freq = int(cfg.get("eval_freq", max(20_000, total_steps // 10)))
    n_eval = int(cfg.get("n_eval_episodes", 12))
    partial_obs = cfg.get("partial_obs", None)
    reward_alpha = cfg.get("reward_alpha", None)
    max_steps = cfg.get("max_steps", None)

    num_envs = vec_env.num_envs
    num_agents = vec_env.num_agents
    obs_dim = vec_env.obs_dim
    state_dim = vec_env.state_dim

    q_net = BranchingQNetwork(obs_dim, hidden_dim).to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    mixer = (QMIXMixer(num_agents, state_dim) if algo == "qmix" else VDNMixer()).to(device)
    target_mixer = copy.deepcopy(mixer).to(device)

    params = list(q_net.parameters()) + list(mixer.parameters())
    optimizer = optim.Adam(params, lr=lr)
    replay = EpisodeReplayBuffer(buffer_capacity, chunk_len, num_agents, obs_dim, state_dim)

    obs_np, state_np, _ = vec_env.reset(seed=seed)

    @torch.no_grad()
    def eval_act_fn(o_np: np.ndarray) -> np.ndarray:
        o_t = torch.as_tensor(o_np, dtype=torch.float32, device=device)
        return q_net.greedy_actions(o_t).cpu().numpy()

    last_eval = 0
    best_ret = -float("inf")
    init_m = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
    init_m["train/fps"] = 0.0
    logger.log_metrics(0, init_m)

    # Per-env chunk accumulators
    env_chunks = [
        {
            "obs": np.zeros((chunk_len, num_agents, obs_dim), dtype=np.float32),
            "next_obs": np.zeros((chunk_len, num_agents, obs_dim), dtype=np.float32),
            "states": np.zeros((chunk_len, state_dim), dtype=np.float32),
            "next_states": np.zeros((chunk_len, state_dim), dtype=np.float32),
            "actions": np.zeros((chunk_len, num_agents, 4), dtype=np.int64),
            "rewards": np.zeros((chunk_len,), dtype=np.float32),
            "dones": np.zeros((chunk_len,), dtype=np.float32),
            "masks": np.zeros((chunk_len,), dtype=np.float32),
            "ptr": 0,
        }
        for _ in range(num_envs)
    ]

    global_step = 0
    start_time = time.time()
    last_metrics = init_m
    td_loss_acc, n_upd = 0.0, 0
    rng = np.random.default_rng(seed)

    while global_step < total_steps:
        frac = min(1.0, global_step / max(1.0, total_steps * eps_decay_frac))
        eps = eps_start + frac * (eps_end - eps_start)

        with torch.no_grad():
            greedy = q_net.greedy_actions(torch.as_tensor(obs_np, dtype=torch.float32, device=device)).cpu().numpy()
        actions_np = greedy.copy()
        for e in range(num_envs):
            for a in range(num_agents):
                if rng.random() < eps:
                    actions_np[e, a] = [int(rng.integers(0, b)) for b in ACTION_BRANCHES]

        next_obs_np, next_state_np, rew_np, done_np, _, _ = vec_env.step(actions_np)

        for e in range(num_envs):
            ch = env_chunks[e]
            t = ch["ptr"]
            ch["obs"][t] = obs_np[e]
            ch["next_obs"][t] = next_obs_np[e]
            ch["states"][t] = state_np[e]
            ch["next_states"][t] = next_state_np[e]
            ch["actions"][t] = actions_np[e]
            ch["rewards"][t] = float(np.mean(rew_np[e]))
            d_flag = float(done_np[e, 0])
            ch["dones"][t] = d_flag
            ch["masks"][t] = 1.0
            ch["ptr"] += 1
            if d_flag > 0.5 or ch["ptr"] >= chunk_len:
                replay.add_chunk(ch)
                ch["masks"].fill(0.0)
                ch["ptr"] = 0

        obs_np, state_np = next_obs_np, next_state_np
        global_step += num_envs

        if replay.size >= batch_size:
            batch = replay.sample(batch_size, device)
            q_chosen = q_net.q_total_for_actions(batch["obs"], batch["actions"])
            q_tot = mixer(q_chosen, batch["states"])

            with torch.no_grad():
                next_greedy = q_net.greedy_actions(batch["next_obs"])
                target_q_vals = target_q_net.q_total_for_actions(batch["next_obs"], next_greedy)
                target_q_tot = target_mixer(target_q_vals, batch["next_states"])
                targets = batch["rewards"] + gamma * (1.0 - batch["dones"]) * target_q_tot

            masks = batch["masks"]
            td_err = (q_tot - targets) * masks
            loss = (td_err ** 2).sum() / masks.sum().clamp(min=1.0)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()

            td_loss_acc += float(loss.item())
            n_upd += 1

            with torch.no_grad():
                for p, tp in zip(q_net.parameters(), target_q_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(p.data * tau)
                for p, tp in zip(mixer.parameters(), target_mixer.parameters()):
                    tp.data.mul_(1.0 - tau).add_(p.data * tau)

        if global_step - last_eval >= eval_freq or global_step >= total_steps:
            last_eval = global_step
            elapsed = max(1e-6, time.time() - start_time)
            last_metrics = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
            last_metrics.update({
                "train/fps": float(global_step / elapsed),
                "train/value_loss": td_loss_acc / max(1, n_upd),
                "train/policy_loss": 0.0,
                "train/entropy": float(eps),
            })
            td_loss_acc, n_upd = 0.0, 0
            logger.log_metrics(global_step, last_metrics)
            if last_metrics["eval/return_mean"] > best_ret:
                best_ret = last_metrics["eval/return_mean"]
                save_marl_checkpoint(
                    str(logger.ckpt_dir / "best.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim,
                    {"q_net": q_net, "mixer": mixer},
                )
            print(
                f"[{algo.upper()} step={global_step:>7d}] Ret: {last_metrics['eval/return_mean']:6.2f} "
                f"| Win: {last_metrics['eval/win_rate']*100:5.1f}% | Core: {last_metrics['eval/core_hp_ratio']*100:5.1f}% "
                f"| eps: {eps:.2f} | FPS: {last_metrics['train/fps']:.0f}"
            )

    save_marl_checkpoint(
        str(logger.ckpt_dir / "final.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim,
        {"q_net": q_net, "mixer": mixer},
    )
    return last_metrics
