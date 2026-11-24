"""Thin runner that trains upstream MAT (factored discrete heads) on DotShot Stages 3 and 4."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from gymnasium import spaces

from benchmarks.eval.marl_eval import evaluate_marl
from benchmarks.official.mat_dotshot_env import DotShotMATEnv, DummyMATVecEnv
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import configure_torch, seed_everything

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
        "num_mini_batch": 4,
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
        # MAT's buffer ignores bad_masks; truncation is handled by bootstrap_truncation().
        "use_proper_time_limits": True,
        "use_linear_lr_decay": True,
        "hidden_size": 64,
        "recurrent_N": 1,
        "episode_length": 200,
        "n_rollout_threads": 32,
    }
    for key in defaults:
        for alias in _ALIASES.get(key, (key,)):
            if alias in config and config[alias] is not None:
                defaults[key] = config[alias]
                break
    for algo_key in ("algorithm_name", "algorithm", "algo"):
        if algo_key in config and config[algo_key] is not None:
            defaults["algorithm_name"] = str(config[algo_key])
            break
    return argparse.Namespace(**defaults)


def _save_checkpoint(ckpt_dir: Path, policy, meta: Dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.transformer.state_dict(), ckpt_dir / "transformer.pt")
    with open(ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


@torch.no_grad()
def bootstrap_truncation(rewards: np.ndarray, infos: List[List[dict]], policy, value_normalizer, gamma: float) -> np.ndarray:
    """Add gamma * V(s_T) to the last reward of episodes cut by the time limit.

    The vector env resets on any episode end and upstream MAT's buffer treats every
    end as terminal, so this is the only place a truncated (``bad_transition``)
    episode can still bootstrap from its final state.
    """
    for t, info in enumerate(infos):
        if not info[0].get("bad_transition", False):
            continue
        n_agents = rewards.shape[1]
        values = _t2n(policy.get_values(
            info[0]["terminal_share_obs"],
            info[0]["terminal_obs"],
            np.zeros((n_agents, 1, 1), dtype=np.float32),
            np.ones((n_agents, 1), dtype=np.float32),
        ))
        if value_normalizer is not None:
            values = value_normalizer.denormalize(values)
        rewards[t] = rewards[t] + gamma * values.reshape(rewards[t].shape)
    return rewards


class MATEvalPolicy:
    """Deterministic MAT behind the ``marl_eval`` policy interface (one env, batch 1)."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, env) -> None:
        self.policy.transformer.eval()

    @torch.no_grad()
    def act(self, env, obs) -> Dict[str, np.ndarray]:
        names = env.possible_agents
        n = len(names)
        obs_arr = np.stack([obs[a] for a in names]).astype(np.float32)
        share = np.repeat(env.state()[None, :], n, axis=0).astype(np.float32)
        actions, _ = self.policy.act(
            share, obs_arr, np.zeros((n, 1, 1), dtype=np.float32), np.ones((n, 1), dtype=np.float32),
            deterministic=True,
        )
        actions = _t2n(actions).astype(np.int64)
        return {name: actions[i] for i, name in enumerate(names)}


def run_mat_experiment(
    config: Dict[str, Any],
    seed: int,
    device: str = "cuda",
    run_dir_override: str | None = None,
    optuna_trial: object | None = None,
) -> Dict[str, Any]:
    """Train MAT on DotShot; same evaluation protocol as ``run_harl_experiment``."""
    seed_everything(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    configure_torch(device)
    torch_device = torch.device(device)

    stage = int(config.get("stage", 3))
    total_steps = int(config.get("total_steps", config.get("num_env_steps", 100_000)))
    env_kwargs = {
        "stage": stage,
        "partial_obs": config.get("partial_obs"),
        "reward_alpha": config.get("reward_alpha"),
        "max_steps": config.get("max_steps"),
        "reward_mode": config.get("reward_mode"),
        "env_overrides": config.get("env_overrides"),
    }
    eval_freq = int(config.get("eval_freq", 100_000))
    n_eval_episodes = int(config.get("n_eval_episodes", 16))
    n_test_episodes = int(config.get("n_test_episodes", 32))
    final_split = str(config.get("final_eval_split", "test"))

    args = _build_args(config)
    args.num_env_steps = total_steps
    if args.algorithm_name not in ("mat", "mat_dec"):
        raise ValueError(f"MAT runner supports 'mat' or 'mat_dec', got {args.algorithm_name!r}")
    args.dec_actor = args.algorithm_name == "mat_dec" or args.dec_actor

    run_dir = run_dir_override or str(Path("runs") / "official" / f"stage{stage}" / "mat" / f"seed_{seed}")
    logger = ExperimentLogger(run_dir, {**config, "seed": seed, "device": device, "mat_args": vars(args)})

    from benchmarks.official.mat_discrete import DotShotMATPolicy
    from mat.algorithms.mat.mat_trainer import MATTrainer
    from mat.utils.shared_buffer import SharedReplayBuffer

    train_env = None
    try:
        train_env = DummyMATVecEnv(
            [DotShotMATEnv(seed=seed + 1000 * i, **env_kwargs) for i in range(args.n_rollout_threads)]
        )
        n_agents = train_env.envs[0].n_agents
        obs_space = train_env.observation_space[0]
        share_obs_space = train_env.share_observation_space[0]
        act_space = train_env.action_space[0]

        policy = DotShotMATPolicy(args, obs_space, share_obs_space, act_space, n_agents, device=torch_device)
        if config.get("init_transformer_path"):
            policy.transformer.load_state_dict(
                torch.load(config["init_transformer_path"], map_location=torch_device)
            )
        if config.get("warm_start") in ("bc", "distill"):
            from benchmarks.official.behaviour_cloning import bc_mat, load_dataset

            teacher = config.get("teacher_path") if config["warm_start"] == "distill" else None
            data = load_dataset(stage, int(config.get("bc_episodes", 200)), env_kwargs, teacher)
            logger.log_metrics(0, bc_mat(policy, data, epochs=int(config.get("bc_epochs", 30))))
        trainer = MATTrainer(args, policy, n_agents, device=torch_device)
        # Integer actions need one column per branch; log-probs keep one joint column.
        act_box = spaces.Box(low=0, high=1, shape=(len(act_space.nvec),), dtype=np.float32)
        buffer = SharedReplayBuffer(args, n_agents, obs_space, share_obs_space, act_box, args.env_name)
        buffer.action_log_probs = np.zeros((*buffer.action_log_probs.shape[:-1], 1), dtype=np.float32)

        meta = {
            "obs_dim": int(obs_space.shape[0]),
            "state_dim": int(share_obs_space.shape[0]),
            "nvec": [int(n) for n in act_space.nvec],
            "n_agents": n_agents,
            "stage": stage,
            "n_block": int(args.n_block),
            "n_embd": int(args.n_embd),
            "n_head": int(args.n_head),
            "encode_state": bool(args.encode_state),
            "dec_actor": bool(args.dec_actor),
        }
        eval_policy = MATEvalPolicy(policy)
        ckpt_dir = Path(logger.ckpt_dir)
        best_return = -float("inf")
        last_val: Dict[str, float] = {}

        def record_eval(step: int) -> None:
            nonlocal best_return, last_val
            metrics, _ = evaluate_marl(env_kwargs, eval_policy, n_eval_episodes, split="val")
            logger.log_metrics(step, metrics)
            last_val = metrics
            if metrics["eval/return_mean"] > best_return:
                best_return = metrics["eval/return_mean"]
                _save_checkpoint(ckpt_dir / "best", policy, meta)
            if optuna_trial is not None and step > 0:
                import optuna

                optuna_trial.report(metrics["eval/return_mean"], step=step)
                if optuna_trial.should_prune():
                    raise optuna.TrialPruned()

        if optuna_trial is None:
            record_eval(0)

        obs, share_obs, _ = train_env.reset()
        buffer.share_obs[0] = share_obs.copy()
        buffer.obs[0] = obs.copy()
        buffer.masks[0] = 1.0

        updates = max(1, total_steps // (args.episode_length * args.n_rollout_threads))
        start = time.time()
        last_eval_step = 0
        env_steps = 0
        for update in range(updates):
            if args.use_linear_lr_decay:
                policy.lr_decay(update, updates)
            trainer.prep_rollout()
            for step in range(args.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = policy.get_actions(
                    np.concatenate(buffer.share_obs[step]),
                    np.concatenate(buffer.obs[step]),
                    np.concatenate(buffer.rnn_states[step]),
                    np.concatenate(buffer.rnn_states_critic[step]),
                    np.concatenate(buffer.masks[step]),
                )
                split = lambda x: np.array(np.split(_t2n(x), args.n_rollout_threads))  # noqa: E731
                values, actions, action_log_probs = split(values), split(actions), split(action_log_probs)
                rnn_states, rnn_states_critic = split(rnn_states), split(rnn_states_critic)

                obs, share_obs, rewards, dones, infos, _ = train_env.step(actions)
                rewards = bootstrap_truncation(rewards, infos, policy, trainer.value_normalizer, args.gamma)
                dones_env = np.all(dones, axis=1)
                rnn_states[dones_env] = 0.0
                rnn_states_critic[dones_env] = 0.0
                masks = np.ones((args.n_rollout_threads, n_agents, 1), dtype=np.float32)
                masks[dones_env] = 0.0
                active_masks = np.ones_like(masks)
                for t in range(args.n_rollout_threads):
                    if dones_env[t]:
                        continue
                    for i in range(n_agents):
                        if not bool(infos[t][i].get("agent_alive", True)):
                            active_masks[t, i, 0] = 0.0

                buffer.insert(
                    share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                    values, rewards, masks, None, active_masks, None,
                )

            trainer.prep_rollout()
            next_values = policy.get_values(
                np.concatenate(buffer.share_obs[-1]),
                np.concatenate(buffer.obs[-1]),
                np.concatenate(buffer.rnn_states_critic[-1]),
                np.concatenate(buffer.masks[-1]),
            )
            next_values = np.array(np.split(_t2n(next_values), args.n_rollout_threads))
            buffer.compute_returns(next_values, trainer.value_normalizer)

            trainer.prep_training()
            train_info = trainer.train(buffer)
            buffer.after_update()

            env_steps = (update + 1) * args.episode_length * args.n_rollout_threads
            train_metrics = {f"train/{k}": _as_float(v) for k, v in train_info.items()}
            train_metrics["train/fps"] = env_steps / max(1e-9, time.time() - start)
            logger.log_metrics(env_steps, train_metrics)

            if eval_freq > 0 and env_steps - last_eval_step >= eval_freq:
                record_eval(env_steps)
                last_eval_step = env_steps

        if last_eval_step != env_steps:
            record_eval(env_steps)
        _save_checkpoint(ckpt_dir / "final", policy, meta)

        prefix = "test" if final_split == "test" else "final_val"
        final_metrics, _ = evaluate_marl(env_kwargs, eval_policy, n_test_episodes, split=final_split, prefix=prefix)
        final_metrics["train/wall_time_s"] = time.time() - start
        logger.log_metrics(env_steps, final_metrics)
        return {
            "run_dir": run_dir,
            "best_val_return": float(best_return),
            "last_val_metrics": last_val,
            "final_metrics": final_metrics,
            "final_return": float(final_metrics[f"{prefix}/return_mean"]),
        }
    finally:
        if train_env is not None:
            train_env.close()
        logger.close()
