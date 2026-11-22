"""CTDE MAPPO and decentralized IPPO trainers (parameter-shared factored actor + ValueNorm critic)."""

import time
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.marl.algos.common import evaluate_marl_policy, save_marl_checkpoint
from benchmarks.marl.buffer import MARLRolloutBuffer
from benchmarks.marl.networks import (
    CentralizedCritic,
    FactoredMultiDiscreteActor,
    ValueNorm,
)


def train_shared_ppo(
    algo: str,
    vec_env: Any,
    cfg: Dict[str, Any],
    logger: Any,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Train MAPPO (centralized critic on s^t) or IPPO (decentralized critic on o_i^t)."""
    assert algo in ("ippo", "mappo")
    stage = int(cfg["stage"])
    total_steps = int(cfg["total_steps"])
    n_steps = int(cfg.get("n_steps", 256))
    n_epochs = int(cfg.get("n_epochs", 6))
    minibatch_size = int(cfg.get("batch_size", 512))
    lr = float(cfg.get("lr", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_coef = float(cfg.get("clip_coef", 0.2))
    ent_coef = float(cfg.get("ent_coef", 0.015))
    vf_coef = float(cfg.get("vf_coef", 0.5))
    max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
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
    critic_dim = state_dim if algo == "mappo" else obs_dim

    actor = FactoredMultiDiscreteActor(obs_dim, hidden_dim).to(device)
    critic = CentralizedCritic(critic_dim, hidden_dim * 2 if algo == "mappo" else hidden_dim).to(device)
    value_norm = ValueNorm().to(device)

    actor_opt = optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    critic_opt = optim.Adam(critic.parameters(), lr=lr, eps=1e-5)

    buffer = MARLRolloutBuffer(n_steps, num_envs, num_agents, obs_dim, state_dim, device)

    obs_np, state_np, active_np = vec_env.reset(seed=seed)
    done_np = np.zeros((num_envs, num_agents), dtype=np.float32)

    @torch.no_grad()
    def eval_act_fn(o_np: np.ndarray) -> np.ndarray:
        o_t = torch.as_tensor(o_np, dtype=torch.float32, device=device)
        acts, _ = actor.act(o_t, deterministic=True)
        return acts.cpu().numpy()

    # Step 0 baseline eval
    last_eval = 0
    best_ret = -float("inf")
    init_m = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
    init_m["train/fps"] = 0.0
    logger.log_metrics(0, init_m)

    global_step = 0
    start_time = time.time()
    last_metrics = init_m

    while global_step < total_steps:
        buffer.reset()
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            active_t = torch.as_tensor(active_np, dtype=torch.float32, device=device)
            done_t = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                acts_t, lps_t = actor.act(obs_t, deterministic=False)
                if algo == "mappo":
                    v_norm = critic(state_t).unsqueeze(-1).expand(-1, num_agents)
                else:
                    v_norm = critic(obs_t)
                v_denorm = value_norm.denormalize(v_norm)

            next_obs_np, next_state_np, rew_np, next_done_np, next_active_np, _ = vec_env.step(acts_t.cpu().numpy())
            rew_t = torch.as_tensor(rew_np, dtype=torch.float32, device=device)

            buffer.add(obs_t, state_t, acts_t, lps_t, rew_t, v_denorm, done_t, active_t)
            obs_np, state_np, active_np, done_np = next_obs_np, next_state_np, next_active_np, next_done_np
            global_step += num_envs

        with torch.no_grad():
            if algo == "mappo":
                nv_norm = critic(torch.as_tensor(state_np, dtype=torch.float32, device=device)).unsqueeze(-1).expand(-1, num_agents)
            else:
                nv_norm = critic(torch.as_tensor(obs_np, dtype=torch.float32, device=device))
            next_values = value_norm.denormalize(nv_norm)
            buffer.compute_gae(next_values, torch.as_tensor(done_np, dtype=torch.float32, device=device), gamma, gae_lambda)

        # Normalize advantages & update ValueNorm
        adv = buffer.advantages
        adv_mean = (adv * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_var = (((adv - adv_mean) ** 2) * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_norm = (adv - adv_mean) / torch.sqrt(adv_var + 1e-8)

        value_norm.update(buffer.returns)
        norm_returns = value_norm.normalize(buffer.returns)

        b_obs = buffer.obs.reshape(-1, obs_dim)
        b_states = buffer.states.unsqueeze(2).expand(-1, -1, num_agents, -1).reshape(-1, state_dim)
        b_acts = buffer.actions.reshape(-1, 4)
        b_old_lps = buffer.log_probs.reshape(-1)
        b_adv = adv_norm.reshape(-1)
        b_ret = norm_returns.reshape(-1)
        b_mask = buffer.active_masks.reshape(-1)

        n_samples = b_obs.shape[0]
        p_loss_acc, v_loss_acc, ent_acc, n_upd = 0.0, 0.0, 0.0, 0

        for _ in range(n_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, minibatch_size):
                idx = perm[start : start + minibatch_size]
                m = b_mask[idx]
                m_sum = m.sum().clamp(min=1.0)

                new_lps, entropy, _ = actor.evaluate_actions(b_obs[idx], b_acts[idx])
                ratio = torch.exp(new_lps - b_old_lps[idx])
                surr1 = ratio * b_adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv[idx]
                policy_loss = -(torch.min(surr1, surr2) * m).sum() / m_sum
                entropy_loss = (entropy * m).sum() / m_sum

                actor_opt.zero_grad()
                (policy_loss - ent_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_opt.step()

                v_pred = critic(b_states[idx] if algo == "mappo" else b_obs[idx])
                value_loss = (((v_pred - b_ret[idx]) ** 2) * m).sum() / m_sum
                critic_opt.zero_grad()
                (vf_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_opt.step()

                p_loss_acc += float(policy_loss.item())
                v_loss_acc += float(value_loss.item())
                ent_acc += float(entropy_loss.item())
                n_upd += 1

        if global_step - last_eval >= eval_freq or global_step >= total_steps:
            last_eval = global_step
            elapsed = max(1e-6, time.time() - start_time)
            last_metrics = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
            last_metrics.update({
                "train/fps": float(global_step / elapsed),
                "train/policy_loss": p_loss_acc / max(1, n_upd),
                "train/value_loss": v_loss_acc / max(1, n_upd),
                "train/entropy": ent_acc / max(1, n_upd),
            })
            logger.log_metrics(global_step, last_metrics)
            if last_metrics["eval/return_mean"] > best_ret:
                best_ret = last_metrics["eval/return_mean"]
                save_marl_checkpoint(
                    str(logger.ckpt_dir / "best.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim,
                    {"actor": actor, "critic": critic},
                )
            print(
                f"[{algo.upper()} step={global_step:>7d}] Ret: {last_metrics['eval/return_mean']:6.2f} "
                f"| Win: {last_metrics['eval/win_rate']*100:5.1f}% | Core: {last_metrics['eval/core_hp_ratio']*100:5.1f}% "
                f"| ElemAcc: {last_metrics['eval/elemental_accuracy']*100:5.1f}% | FPS: {last_metrics['train/fps']:.0f}"
            )

    save_marl_checkpoint(
        str(logger.ckpt_dir / "final.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim,
        {"actor": actor, "critic": critic},
    )
    return last_metrics
