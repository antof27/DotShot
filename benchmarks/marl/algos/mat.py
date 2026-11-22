"""Factored Multi-Agent Transformer trainers (MAT, PMAT, and RoMAT)."""

import time
from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from benchmarks.marl.algos.common import evaluate_marl_policy, save_marl_checkpoint
from benchmarks.marl.buffer import MARLRolloutBuffer
from benchmarks.marl.networks import FactoredMATNetwork, ValueNorm


def train_mat_family(
    algo: str,
    vec_env: Any,
    cfg: Dict[str, Any],
    logger: Any,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """Train Factored MAT, PMAT (Plackett-Luce), or RoMAT (role embeddings)."""
    assert algo in ("mat", "pmat", "romat")
    stage = int(cfg["stage"])
    total_steps = int(cfg["total_steps"])
    n_steps = int(cfg.get("n_steps", 256))
    n_epochs = int(cfg.get("n_epochs", 6))
    minibatch_size = int(cfg.get("batch_size", 256))
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

    model = FactoredMATNetwork(obs_dim, num_agents, hidden_dim=hidden_dim, mode=algo).to(device)
    value_norm = ValueNorm().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    buffer = MARLRolloutBuffer(n_steps, num_envs, num_agents, obs_dim, state_dim, device)
    obs_np, state_np, active_np = vec_env.reset(seed=seed)
    done_np = np.zeros((num_envs, num_agents), dtype=np.float32)

    @torch.no_grad()
    def eval_act_fn(o_np: np.ndarray) -> np.ndarray:
        o_t = torch.as_tensor(o_np, dtype=torch.float32, device=device).unsqueeze(0)
        acts, _, _, _, _ = model.act(o_t, deterministic=True)
        return acts.squeeze(0).cpu().numpy()

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

            acts_t, lps_t, vals_norm, order_t, _ = model.act(obs_t, deterministic=False)
            vals_denorm = value_norm.denormalize(vals_norm)

            next_obs_np, next_state_np, rew_np, next_done_np, next_active_np, _ = vec_env.step(acts_t.cpu().numpy())
            rew_t = torch.as_tensor(rew_np, dtype=torch.float32, device=device)

            buffer.add(obs_t, state_t, acts_t, lps_t, rew_t, vals_denorm, done_t, active_t, orders=order_t)
            obs_np, state_np, active_np, done_np = next_obs_np, next_state_np, next_active_np, next_done_np
            global_step += num_envs

        with torch.no_grad():
            _, nv_norm = model.encode(torch.as_tensor(obs_np, dtype=torch.float32, device=device))
            next_vals = value_norm.denormalize(nv_norm)
            buffer.compute_gae(next_vals, torch.as_tensor(done_np, dtype=torch.float32, device=device), gamma, gae_lambda)

        adv = buffer.advantages
        adv_mean = (adv * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_var = (((adv - adv_mean) ** 2) * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_norm = (adv - adv_mean) / torch.sqrt(adv_var + 1e-8)

        value_norm.update(buffer.returns)
        norm_returns = value_norm.normalize(buffer.returns)

        B = n_steps * num_envs
        b_obs = buffer.obs.reshape(B, num_agents, obs_dim)
        b_acts = buffer.actions.reshape(B, num_agents, 4)
        b_orders = buffer.orders.reshape(B, num_agents)
        b_old_lps = buffer.log_probs.reshape(B, num_agents)
        b_adv = adv_norm.reshape(B, num_agents)
        b_ret = norm_returns.reshape(B, num_agents)
        b_mask = buffer.active_masks.reshape(B, num_agents)

        p_loss_acc, v_loss_acc, ent_acc, n_upd = 0.0, 0.0, 0.0, 0
        for _ in range(n_epochs):
            perm = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = perm[start : start + minibatch_size]
                m = b_mask[mb]
                m_sum = m.sum().clamp(min=1.0)

                new_lps, entropy, values = model.evaluate_actions(b_obs[mb], b_acts[mb], order=b_orders[mb])
                ratio = torch.exp(new_lps - b_old_lps[mb])
                surr1 = ratio * b_adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv[mb]
                policy_loss = -(torch.min(surr1, surr2) * m).sum() / m_sum
                entropy_loss = (entropy * m).sum() / m_sum
                value_loss = (((values - b_ret[mb]) ** 2) * m).sum() / m_sum

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

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
                    {"mat": model},
                )
            print(
                f"[{algo.upper()} step={global_step:>7d}] Ret: {last_metrics['eval/return_mean']:6.2f} "
                f"| Win: {last_metrics['eval/win_rate']*100:5.1f}% | Core: {last_metrics['eval/core_hp_ratio']*100:5.1f}% "
                f"| ElemAcc: {last_metrics['eval/elemental_accuracy']*100:5.1f}% | FPS: {last_metrics['train/fps']:.0f}"
            )

    save_marl_checkpoint(
        str(logger.ckpt_dir / "final.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim,
        {"mat": model},
    )
    return last_metrics
