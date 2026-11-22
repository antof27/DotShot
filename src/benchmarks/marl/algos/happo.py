"""HAPPO and HATRPO trainers implementing Multi-Agent Advantage Decomposition and sequential updates."""

import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, kl_divergence

from benchmarks.marl.algos.common import evaluate_marl_policy, save_marl_checkpoint
from benchmarks.marl.buffer import MARLRolloutBuffer
from benchmarks.marl.networks import (
    CentralizedCritic,
    FactoredMultiDiscreteActor,
    ValueNorm,
)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p.data))
        idx += n


def _mean_factored_kl(old_logits: List[torch.Tensor], new_logits: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    kl = torch.zeros_like(mask)
    for ol, nl in zip(old_logits, new_logits):
        d_old = Categorical(logits=ol)
        d_new = Categorical(logits=nl)
        kl = kl + kl_divergence(d_old, d_new)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0)


def _conjugate_gradient(fvp_fn, b: torch.Tensor, n_iters: int = 10, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)
    for _ in range(n_iters):
        if r_dot_r < residual_tol:
            break
        Avp = fvp_fn(p)
        alpha = r_dot_r / (torch.dot(p, Avp) + 1e-8)
        x = x + alpha * p
        r = r - alpha * Avp
        new_r_dot_r = torch.dot(r, r)
        beta = new_r_dot_r / (r_dot_r + 1e-8)
        p = r + beta * p
        r_dot_r = new_r_dot_r
    return x


def train_happo_family(
    algo: str,
    vec_env: Any,
    cfg: Dict[str, Any],
    logger: Any,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    """
    Train HAPPO or HATRPO (Kuba et al., 2022; Zhong et al., 2024):
    - Separate policy parameters per agent (no parameter sharing).
    - Random permutation i_{1:n} per update.
    - Sequential importance factor M^{i_{1:m}} accumulated across agents.
    """
    assert algo in ("happo", "hatrpo")
    stage = int(cfg["stage"])
    total_steps = int(cfg["total_steps"])
    n_steps = int(cfg.get("n_steps", 256))
    n_epochs = int(cfg.get("n_epochs", 5))
    minibatch_size = int(cfg.get("batch_size", 512))
    lr = float(cfg.get("lr", 3e-4))
    gamma = float(cfg.get("gamma", 0.99))
    gae_lambda = float(cfg.get("gae_lambda", 0.95))
    clip_coef = float(cfg.get("clip_coef", 0.2))
    kl_threshold = float(cfg.get("kl_threshold", 0.015))
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

    actors = nn.ModuleList([FactoredMultiDiscreteActor(obs_dim, hidden_dim).to(device) for _ in range(num_agents)])
    actor_opts = [optim.Adam(actors[i].parameters(), lr=lr, eps=1e-5) for i in range(num_agents)]
    critic = CentralizedCritic(state_dim, hidden_dim * 2).to(device)
    critic_opt = optim.Adam(critic.parameters(), lr=lr, eps=1e-5)
    value_norm = ValueNorm().to(device)

    buffer = MARLRolloutBuffer(n_steps, num_envs, num_agents, obs_dim, state_dim, device)
    obs_np, state_np, active_np = vec_env.reset(seed=seed)
    done_np = np.zeros((num_envs, num_agents), dtype=np.float32)

    @torch.no_grad()
    def eval_act_fn(o_np: np.ndarray) -> np.ndarray:
        o_t = torch.as_tensor(o_np, dtype=torch.float32, device=device)
        out = [actors[i].act(o_t[i : i + 1], deterministic=True)[0].squeeze(0) for i in range(num_agents)]
        return torch.stack(out, dim=0).cpu().numpy()

    last_eval = 0
    best_ret = -float("inf")
    init_m = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
    init_m["train/fps"] = 0.0
    logger.log_metrics(0, init_m)

    global_step = 0
    start_time = time.time()
    last_metrics = init_m
    rng = np.random.default_rng(seed)

    while global_step < total_steps:
        buffer.reset()
        for _ in range(n_steps):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            state_t = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            active_t = torch.as_tensor(active_np, dtype=torch.float32, device=device)
            done_t = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            with torch.no_grad():
                acts_list, lps_list = [], []
                for i in range(num_agents):
                    a_i, lp_i = actors[i].act(obs_t[:, i, :], deterministic=False)
                    acts_list.append(a_i)
                    lps_list.append(lp_i)
                acts_t = torch.stack(acts_list, dim=1)
                lps_t = torch.stack(lps_list, dim=1)
                v_denorm = value_norm.denormalize(critic(state_t)).unsqueeze(-1).expand(-1, num_agents)

            next_obs_np, next_state_np, rew_np, next_done_np, next_active_np, _ = vec_env.step(acts_t.cpu().numpy())
            rew_t = torch.as_tensor(rew_np, dtype=torch.float32, device=device)

            buffer.add(obs_t, state_t, acts_t, lps_t, rew_t, v_denorm, done_t, active_t)
            obs_np, state_np, active_np, done_np = next_obs_np, next_state_np, next_active_np, next_done_np
            global_step += num_envs

        with torch.no_grad():
            nv_denorm = value_norm.denormalize(
                critic(torch.as_tensor(state_np, dtype=torch.float32, device=device))
            ).unsqueeze(-1).expand(-1, num_agents)
            buffer.compute_gae(nv_denorm, torch.as_tensor(done_np, dtype=torch.float32, device=device), gamma, gae_lambda)

        adv = buffer.advantages
        adv_mean = (adv * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_var = (((adv - adv_mean) ** 2) * buffer.active_masks).sum() / buffer.active_masks.sum().clamp(min=1.0)
        adv_norm = (adv - adv_mean) / torch.sqrt(adv_var + 1e-8)

        value_norm.update(buffer.returns)
        norm_returns = value_norm.normalize(buffer.returns)

        # Flatten [T, E, ...] -> [B, ...]
        B = n_steps * num_envs
        b_obs = buffer.obs.reshape(B, num_agents, obs_dim)
        b_states = buffer.states.reshape(B, state_dim)
        b_acts = buffer.actions.reshape(B, num_agents, 4)
        b_old_lps = buffer.log_probs.reshape(B, num_agents)
        b_adv = adv_norm.reshape(B, num_agents)
        b_ret = norm_returns.reshape(B, num_agents).mean(dim=-1)
        b_mask = buffer.active_masks.reshape(B, num_agents)

        # Sequential per-agent update with random permutation
        perm_agents = rng.permutation(num_agents)
        factor = torch.ones(B, dtype=torch.float32, device=device)

        p_loss_acc, v_loss_acc, ent_acc, n_upd = 0.0, 0.0, 0.0, 0

        for agent_idx in perm_agents:
            obs_i = b_obs[:, agent_idx, :]
            act_i = b_acts[:, agent_idx, :]
            old_lp_i = b_old_lps[:, agent_idx]
            adv_i = b_adv[:, agent_idx]
            mask_i = b_mask[:, agent_idx]

            if algo == "happo":
                for _ in range(n_epochs):
                    idx_perm = torch.randperm(B, device=device)
                    for start in range(0, B, minibatch_size):
                        mb = idx_perm[start : start + minibatch_size]
                        m = mask_i[mb]
                        m_sum = m.sum().clamp(min=1.0)

                        new_lp, ent, _ = actors[agent_idx].evaluate_actions(obs_i[mb], act_i[mb])
                        ratio = torch.exp(new_lp - old_lp_i[mb])
                        weighted_adv = factor[mb] * adv_i[mb]
                        surr1 = ratio * weighted_adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * weighted_adv
                        p_loss = -(torch.min(surr1, surr2) * m).sum() / m_sum
                        e_loss = (ent * m).sum() / m_sum

                        actor_opts[agent_idx].zero_grad()
                        (p_loss - ent_coef * e_loss).backward()
                        nn.utils.clip_grad_norm_(actors[agent_idx].parameters(), max_grad_norm)
                        actor_opts[agent_idx].step()

                        p_loss_acc += float(p_loss.item())
                        ent_acc += float(e_loss.item())
                        n_upd += 1
            else:
                # HATRPO: Fisher-vector conjugate gradient + KL-constrained backtracking line search
                m_sum = mask_i.sum().clamp(min=1.0)
                with torch.no_grad():
                    _, _, old_logits = actors[agent_idx].evaluate_actions(obs_i, act_i)
                    old_logits_det = [ol.detach() for ol in old_logits]

                new_lp, ent, _ = actors[agent_idx].evaluate_actions(obs_i, act_i)
                ratio = torch.exp(new_lp - old_lp_i)
                gain = (ratio * factor * adv_i * mask_i).sum() / m_sum
                grads = torch.autograd.grad(gain, actors[agent_idx].parameters())
                loss_grad = torch.cat([g.contiguous().view(-1) for g in grads]).detach()

                def fvp(v: torch.Tensor) -> torch.Tensor:
                    _, _, cur_logits = actors[agent_idx].evaluate_actions(obs_i, act_i)
                    kl = _mean_factored_kl(old_logits_det, cur_logits, mask_i)
                    kl_grads = torch.autograd.grad(kl, actors[agent_idx].parameters(), create_graph=True)
                    flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grads])
                    gv = (flat_kl_grad * v).sum()
                    hvp = torch.autograd.grad(gv, actors[agent_idx].parameters())
                    return torch.cat([h.contiguous().view(-1) for h in hvp]).detach() + 0.1 * v

                step_dir = _conjugate_gradient(fvp, loss_grad, n_iters=8)
                shs = 0.5 * torch.dot(step_dir, fvp(step_dir))
                max_step = torch.sqrt(kl_threshold / (shs + 1e-8))
                full_step = step_dir * max_step

                old_params = _flat_params(actors[agent_idx]).clone()
                old_gain = float(gain.item())
                accepted = False
                for step_frac in (1.0, 0.5, 0.25, 0.125):
                    _set_flat_params(actors[agent_idx], old_params + step_frac * full_step)
                    with torch.no_grad():
                        cand_lp, cand_ent, cand_logits = actors[agent_idx].evaluate_actions(obs_i, act_i)
                        cand_gain = float(((torch.exp(cand_lp - old_lp_i) * factor * adv_i) * mask_i).sum().item() / m_sum.item())
                        cand_kl = float(_mean_factored_kl(old_logits_det, cand_logits, mask_i).item())
                    if cand_kl <= kl_threshold * 1.5 and cand_gain >= old_gain:
                        accepted = True
                        p_loss_acc += -cand_gain
                        ent_acc += float((cand_ent * mask_i).sum().item() / m_sum.item())
                        n_upd += 1
                        break
                if not accepted:
                    _set_flat_params(actors[agent_idx], old_params)

            # Accumulate sequential policy ratio M^{i_{1:m}}
            with torch.no_grad():
                post_lp, _, _ = actors[agent_idx].evaluate_actions(obs_i, act_i)
                ratio_post = torch.exp(post_lp - old_lp_i)
                # Keep dead agent steps neutral (1.0)
                ratio_masked = torch.where(mask_i > 0.5, ratio_post, torch.ones_like(ratio_post))
                factor = (factor * ratio_masked).clamp(0.1, 10.0)

        # Update shared Centralized Critic
        for _ in range(n_epochs):
            idx_perm = torch.randperm(B, device=device)
            for start in range(0, B, minibatch_size):
                mb = idx_perm[start : start + minibatch_size]
                v_pred = critic(b_states[mb])
                v_loss = ((v_pred - b_ret[mb]) ** 2).mean()
                critic_opt.zero_grad()
                (vf_coef * v_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_opt.step()
                v_loss_acc += float(v_loss.item())

        if global_step - last_eval >= eval_freq or global_step >= total_steps:
            last_eval = global_step
            elapsed = max(1e-6, time.time() - start_time)
            last_metrics = evaluate_marl_policy(stage, eval_act_fn, n_eval, seed + 10_000, partial_obs, reward_alpha, max_steps)
            last_metrics.update({
                "train/fps": float(global_step / elapsed),
                "train/policy_loss": p_loss_acc / max(1, n_upd),
                "train/value_loss": v_loss_acc / max(1, n_epochs),
                "train/entropy": ent_acc / max(1, n_upd),
            })
            logger.log_metrics(global_step, last_metrics)
            models_dict = {f"actor_{i}": actors[i] for i in range(num_agents)}
            models_dict["critic"] = critic
            if last_metrics["eval/return_mean"] > best_ret:
                best_ret = last_metrics["eval/return_mean"]
                save_marl_checkpoint(
                    str(logger.ckpt_dir / "best.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim, models_dict
                )
            print(
                f"[{algo.upper()} step={global_step:>7d}] Ret: {last_metrics['eval/return_mean']:6.2f} "
                f"| Win: {last_metrics['eval/win_rate']*100:5.1f}% | Core: {last_metrics['eval/core_hp_ratio']*100:5.1f}% "
                f"| ElemAcc: {last_metrics['eval/elemental_accuracy']*100:5.1f}% | FPS: {last_metrics['train/fps']:.0f}"
            )

    models_dict = {f"actor_{i}": actors[i] for i in range(num_agents)}
    models_dict["critic"] = critic
    save_marl_checkpoint(
        str(logger.ckpt_dir / "final.pt"), algo, stage, num_agents, obs_dim, state_dim, hidden_dim, models_dict
    )
    return last_metrics
