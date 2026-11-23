"""Thin runner that trains the unmodified Multi-Agent Transformer (MAT) on DotShot."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from benchmarks.official.mat_dotshot_env import DummyMATVecEnv, DotShotContinuousMATEnv
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import seed_everything

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MAT_ROOT = _REPO_ROOT / "third_party" / "Multi-Agent-Transformer"
if str(_MAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAT_ROOT))

_ALIASES = {
    "episode_length": ("episode_length", "n_steps"),
    "n_rollout_threads": ("n_rollout_threads", "num_envs"),
    "hidden_size": ("hidden_size", "hidden_dim"),
    "entropy_coef": ("entropy_coef", "ent_coef"),
    "value_loss_coef": ("value_loss_coef", "vf_coef"),
    "clip_param": ("clip_param", "clip_coef", "clip_range"),
    "ppo_epoch": ("ppo_epoch", "n_epochs"),
}


def _t2n(x):
    return x.detach().cpu().numpy()


def _as_float(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def _build_args(config: Dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "algorithm_name": "mat",
        "env_name": "DotShot",
        "lr": 5e-4,
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "use_policy_active_masks": True,
        "use_value_active_masks": True,
        "n_block": 1,
        "n_embd": 64,
        "n_head": 1,
        "encode_state": False,
        "dec_actor": False,
        "share_actor": False,
        "clip_param": 0.2,
        "ppo_epoch": 5,
        "num_mini_batch": 1,
        "data_chunk_length": 10,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "huber_delta": 10.0,
        "use_recurrent_policy": False,
        "use_naive_recurrent_policy": False,
        "use_max_grad_norm": True,
        "use_clipped_value_loss": True,
        "use_huber_loss": True,
        "use_valuenorm": True,
        "use_popart": False,
        "use_gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "use_proper_time_limits": False,
        "hidden_size": 64,
        "recurrent_N": 1,
        "episode_length": 200,
        "n_rollout_threads": 8,
    }
    for key in defaults:
        aliases = _ALIASES.get(key, (key,))
        for alias in aliases:
            if alias in config and config[alias] is not None:
                defaults[key] = config[alias]
                break
    for algo_key in ("algorithm_name", "algorithm", "algo"):
        if algo_key in config and config[algo_key] is not None:
            defaults["algorithm_name"] = str(config[algo_key])
            break
    return argparse.Namespace(**defaults)


def _make_envs(
    count: int,
    seed: int,
    stage: int,
    partial_obs: bool,
    reward_alpha: float,
    max_steps: int | None,
) -> DummyMATVecEnv:
    envs = [
        DotShotContinuousMATEnv(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
            seed=seed + 1000 * i,
        )
        for i in range(count)
    ]
    return DummyMATVecEnv(envs)


def _save_checkpoint(ckpt_dir: Path, policy, meta: Dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.transformer.state_dict(), ckpt_dir / "transformer.pt")
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _metric_from_info(key: str, info: Dict[str, Any], default: float) -> float:
    return _as_float(info.get(key, default))


@torch.no_grad()
def _evaluate(
    policy,
    eval_env: DummyMATVecEnv,
    n_episodes: int,
    max_eval_steps: int,
    hidden_size: int,
    n_agents: int,
) -> Dict[str, float]:
    policy.eval()
    n_threads = eval_env.n_rollout_threads
    obs, share_obs, _ = eval_env.reset()
    rnn_states = np.zeros((n_threads, n_agents, 1, hidden_size), dtype=np.float32)
    masks = np.ones((n_threads, n_agents, 1), dtype=np.float32)

    episode_returns = np.zeros(n_threads, dtype=np.float64)
    returns, wins, cores, elemental, kills, boss_kills, shield_breaks = [], [], [], [], [], [], []
    last_infos = [[{} for _ in range(n_agents)] for _ in range(n_threads)]

    steps = 0
    while len(returns) < n_episodes and steps < max_eval_steps:
        actions, _ = policy.act(
            np.concatenate(share_obs),
            np.concatenate(obs),
            np.concatenate(rnn_states),
            np.concatenate(masks),
            deterministic=True,
        )
        actions = np.array(np.split(_t2n(actions), n_threads))
        obs, share_obs, rewards, dones, last_infos, _ = eval_env.step(actions)
        episode_returns += rewards.mean(axis=(1, 2)).astype(np.float64)
        for t in range(n_threads):
            if np.all(dones[t]):
                info = last_infos[t][0]
                returns.append(float(episode_returns[t]))
                wins.append(_metric_from_info("win", info, 0.0))
                cores.append(_metric_from_info("core_hp_ratio", info, 1.0))
                elemental.append(_metric_from_info("elemental_accuracy", info, 0.0))
                kills.append(_metric_from_info("kills", info, 0.0))
                boss_kills.append(_metric_from_info("boss_kills", info, 0.0))
                shield_breaks.append(_metric_from_info("shield_breaks", info, 0.0))
                episode_returns[t] = 0.0
        steps += 1

    if not returns:
        for t in range(n_threads):
            info = last_infos[t][0] if last_infos[t] else {}
            returns.append(float(episode_returns[t]))
            wins.append(_metric_from_info("win", info, 0.0))
            cores.append(_metric_from_info("core_hp_ratio", info, 1.0))
            elemental.append(_metric_from_info("elemental_accuracy", info, 0.0))
            kills.append(_metric_from_info("kills", info, 0.0))
            boss_kills.append(_metric_from_info("boss_kills", info, 0.0))
            shield_breaks.append(_metric_from_info("shield_breaks", info, 0.0))

    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/win_rate": float(np.mean(wins)),
        "eval/core_hp_ratio": float(np.mean(cores)),
        "eval/elemental_accuracy": float(np.mean(elemental)),
        "eval/kills_mean": float(np.mean(kills)),
        "eval/boss_kills_mean": float(np.mean(boss_kills)),
        "eval/shield_breaks_mean": float(np.mean(shield_breaks)),
    }


def run_mat_experiment(
    config: Dict[str, Any],
    seed: int,
    device: str = "cuda",
    run_dir_override: str | None = None,
    optuna_trial: object | None = None,
) -> Dict[str, Any]:
    """Train upstream MAT on DotShot and return the best and final evaluation metrics."""
    torch.set_num_threads(1)
    seed_everything(seed)

    stage = int(config.get("stage", 3))
    total_steps = int(config.get("total_steps", config.get("num_env_steps", 100_000)))
    episode_length = int(config.get("episode_length", config.get("n_steps", 200)))
    n_threads = int(config.get("n_rollout_threads", config.get("num_envs", 8)))
    partial_obs = bool(config.get("partial_obs", True))
    reward_alpha = float(config.get("reward_alpha", 0.5))
    max_steps = config.get("max_steps", None)
    exp_name = str(config.get("exp_name", f"stage{stage}_mat"))
    eval_freq = max(1, int(config.get("eval_freq", config.get("eval_interval", total_steps))))
    n_eval_episodes = max(1, int(config.get("n_eval_episodes", config.get("eval_episodes", 4))))
    n_eval_threads = int(
        config.get("n_eval_rollout_threads", min(n_threads, n_eval_episodes, 4))
    )
    n_eval_threads = max(1, min(n_eval_threads, n_eval_episodes))

    args = _build_args(config)
    args.episode_length = max(1, episode_length)
    args.n_rollout_threads = max(1, n_threads)
    args.num_env_steps = total_steps
    if args.algorithm_name not in ("mat", "mat_dec"):
        raise ValueError(f"MAT runner supports 'mat' or 'mat_dec', got {args.algorithm_name!r}")

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    resolved_cfg = {
        **config,
        "seed": seed,
        "device": device,
        "total_steps": total_steps,
        "episode_length": args.episode_length,
        "n_rollout_threads": args.n_rollout_threads,
    }
    run_dir = run_dir_override or os.path.join(
        "runs", f"stage{stage}", "mat", exp_name, f"seed_{seed}"
    )
    logger = ExperimentLogger(run_dir, resolved_cfg)

    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
    from mat.algorithms.mat.mat_trainer import MATTrainer
    from mat.utils.shared_buffer import SharedReplayBuffer

    train_env = None
    eval_env = None
    try:
        train_env = _make_envs(
            args.n_rollout_threads, seed, stage, partial_obs, reward_alpha, max_steps
        )
        eval_env = _make_envs(
            n_eval_threads, seed + 10_000, stage, partial_obs, reward_alpha, max_steps
        )
        n_agents = train_env.envs[0].n_agents
        obs_space = train_env.observation_space[0]
        share_obs_space = train_env.share_observation_space[0]
        act_space = train_env.action_space[0]

        policy = TransformerPolicy(
            args, obs_space, share_obs_space, act_space, n_agents, device=torch_device
        )
        trainer = MATTrainer(args, policy, n_agents, device=torch_device)
        buffer = SharedReplayBuffer(
            args, n_agents, obs_space, share_obs_space, act_space, args.env_name
        )

        obs_dim = int(obs_space.shape[0])
        state_dim = int(share_obs_space.shape[0])
        act_dim = int(act_space.shape[0])
        meta = {
            "obs_dim": obs_dim,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "n_agents": n_agents,
            "stage": stage,
            "n_block": int(args.n_block),
            "n_embd": int(args.n_embd),
            "n_head": int(args.n_head),
            "encode_state": bool(args.encode_state),
            "dec_actor": bool(args.dec_actor),
            "share_actor": bool(args.share_actor),
        }
        best_dir = Path(logger.ckpt_dir) / "best"
        final_dir = Path(logger.ckpt_dir) / "final"
        max_eval_steps = int(max_steps or train_env.envs[0].env.engine.cfg.max_steps)

        obs, share_obs, _ = train_env.reset()
        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
        buffer.masks[0] = 1.0

        def evaluate_and_log(step: int) -> Dict[str, float]:
            metrics = _evaluate(
                policy, eval_env, n_eval_episodes, max_eval_steps, args.hidden_size, n_agents
            )
            logger.log_metrics(step, metrics)
            if optuna_trial is not None:
                import optuna

                optuna_trial.report(metrics["eval/return_mean"], step=step)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()
            return metrics

        best_return = float("-inf")
        final_metrics = evaluate_and_log(0)
        best_return = final_metrics["eval/return_mean"]
        last_eval_step = 0
        _save_checkpoint(best_dir, policy, meta)

        updates = max(
            1, total_steps // (args.episode_length * args.n_rollout_threads)
        )
        next_eval = eval_freq
        start = time.time()

        for episode in range(updates):
            for step in range(args.episode_length):
                trainer.prep_rollout()
                values, actions, action_log_probs, rnn_states, rnn_states_critic = (
                    policy.get_actions(
                        np.concatenate(buffer.share_obs[step]),
                        np.concatenate(buffer.obs[step]),
                        np.concatenate(buffer.rnn_states[step]),
                        np.concatenate(buffer.rnn_states_critic[step]),
                        np.concatenate(buffer.masks[step]),
                    )
                )
                values = np.array(np.split(_t2n(values), args.n_rollout_threads))
                actions = np.array(np.split(_t2n(actions), args.n_rollout_threads))
                action_log_probs = np.array(
                    np.split(_t2n(action_log_probs), args.n_rollout_threads)
                )
                rnn_states = np.array(np.split(_t2n(rnn_states), args.n_rollout_threads))
                rnn_states_critic = np.array(
                    np.split(_t2n(rnn_states_critic), args.n_rollout_threads)
                )

                obs, share_obs, rewards, dones, infos, _ = train_env.step(actions)
                dones_env = np.all(dones, axis=1)

                rnn_states[dones_env] = 0.0
                rnn_states_critic[dones_env] = 0.0
                masks = np.ones(
                    (args.n_rollout_threads, n_agents, 1), dtype=np.float32
                )
                masks[dones_env] = 0.0
                active_masks = np.ones_like(masks)
                for t in range(args.n_rollout_threads):
                    if dones_env[t]:
                        continue
                    for i in range(n_agents):
                        if not bool(infos[t][i].get("agent_alive", True)):
                            active_masks[t, i, 0] = 0.0

                buffer.insert(
                    share_obs,
                    obs,
                    rnn_states,
                    rnn_states_critic,
                    actions,
                    action_log_probs,
                    values,
                    rewards,
                    masks,
                    None,
                    active_masks,
                    None,
                )

            trainer.prep_rollout()
            next_values = policy.get_values(
                np.concatenate(buffer.share_obs[-1]),
                np.concatenate(buffer.obs[-1]),
                np.concatenate(buffer.rnn_states_critic[-1]),
                np.concatenate(buffer.masks[-1]),
            )
            next_values = np.array(
                np.split(_t2n(next_values), args.n_rollout_threads)
            )
            buffer.compute_returns(next_values, trainer.value_normalizer)

            trainer.prep_training()
            train_info = trainer.train(buffer)
            buffer.after_update()

            total_num_steps = (
                (episode + 1) * args.episode_length * args.n_rollout_threads
            )
            train_metrics = {
                f"train/{key}": _as_float(value) for key, value in train_info.items()
            }
            elapsed = max(1e-9, time.time() - start)
            train_metrics["train/fps"] = total_num_steps / elapsed
            logger.log_metrics(total_num_steps, train_metrics)

            if total_num_steps >= next_eval:
                final_metrics = evaluate_and_log(total_num_steps)
                last_eval_step = total_num_steps
                next_eval += eval_freq
                if final_metrics["eval/return_mean"] > best_return:
                    best_return = final_metrics["eval/return_mean"]
                    _save_checkpoint(best_dir, policy, meta)

        if last_eval_step != total_num_steps:
            final_metrics = evaluate_and_log(total_num_steps)
            if final_metrics["eval/return_mean"] > best_return:
                best_return = final_metrics["eval/return_mean"]
                _save_checkpoint(best_dir, policy, meta)

        _save_checkpoint(final_dir, policy, meta)
        return {
            "run_dir": run_dir,
            "best_return": float(best_return),
            "final_metrics": final_metrics,
        }
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        logger.close()
