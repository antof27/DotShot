"""Config-driven single-agent runner supporting Random, Heuristic, SB3 A2C, and SB3 PPO."""

import argparse
import math
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import dotshot  # noqa: F401
from dotshot.core.config import ELEMENT_WEAKNESS
from dotshot.envs.single_agent import DotShotSingleEnv
from benchmarks.eval.marl_eval import episode_seeds
from benchmarks.utils.logger import ExperimentLogger
from benchmarks.utils.seeding import configure_torch, seed_everything


def cosine_lr_schedule(initial_lr: float, min_lr: float = 1e-5):
    def schedule(progress_remaining: float) -> float:
        cos_val = math.cos((1.0 - progress_remaining) * math.pi)
        return float(min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + cos_val))

    return schedule


def heuristic_policy_action(env: DotShotSingleEnv) -> np.ndarray:
    """Scripted target-leading, elemental-counter, and kiting baseline."""
    engine = env.engine
    agent = engine.agents[0]
    enemies = [e for e in engine.enemies if e.alive]
    if not enemies:
        return np.array([1, 1, 0, 0], dtype=np.int64)

    nearest = min(enemies, key=lambda e: np.hypot(e.x - agent.x, e.y - agent.y))
    dx = nearest.x - agent.x
    dy = nearest.y - agent.y
    dist = float(np.hypot(dx, dy))

    # Lead moving target linearly
    lead_t = dist / max(1.0, agent.bullet_speed)
    pred_x = nearest.x + nearest.vx * lead_t * 0.85
    pred_y = nearest.y + nearest.vy * lead_t * 0.85
    aim_dx = pred_x - agent.x
    aim_dy = pred_y - agent.y
    ang = float(np.arctan2(-aim_dy, aim_dx))
    if ang < 0.0:
        ang += 2.0 * np.pi
    aim_bin = int(round(ang / (2.0 * np.pi / engine.cfg.aim_bins))) % engine.cfg.aim_bins

    # Kite away if too close, otherwise strafe tangentially
    mx, my = 0, 0
    if dist < 240.0 and dist > 1e-5:
        mx = -1 if dx > 25.0 else (1 if dx < -25.0 else 0)
        my = -1 if dy > 25.0 else (1 if dy < -25.0 else 0)
    elif dist > 460.0 and dist > 1e-5:
        mx = 1 if dx > 25.0 else (-1 if dx < -25.0 else 0)
        my = 1 if dy > 25.0 else (-1 if dy < -25.0 else 0)

    weapon_type = ELEMENT_WEAKNESS[nearest.enemy_type]
    return np.array([mx + 1, my + 1, aim_bin, weapon_type + 1], dtype=np.int64)


def evaluate_single_agent(
    stage: int,
    policy_fn,
    n_episodes: int,
    split: str = "val",
    max_steps: int | None = None,
    prefix: str = "eval",
    env_overrides: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    """Exactly ``n_episodes`` episodes on a shared held-out seed split (see marl_eval)."""
    env = DotShotSingleEnv(stage=stage, max_steps=max_steps, env_overrides=env_overrides)
    rows: List[Dict[str, Any]] = []
    for ep_seed in episode_seeds(split, n_episodes):
        obs, _ = env.reset(seed=ep_seed)
        ep_ret = 0.0
        done = False
        info: Dict[str, Any] = {}
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy_fn(env, obs))
            ep_ret += float(reward)
            done = terminated or truncated
        rows.append({"return": ep_ret, **info})
    env.close()

    def mean(key: str) -> float:
        return float(np.mean([float(r.get(key, 0.0)) for r in rows]))

    returns = [r["return"] for r in rows]
    metrics = {
        f"{prefix}/n_episodes": float(len(rows)),
        f"{prefix}/return_mean": float(np.mean(returns)),
        f"{prefix}/return_std": float(np.std(returns)),
        f"{prefix}/win_rate": mean("win"),
        f"{prefix}/kills": mean("kills"),
        f"{prefix}/elemental_accuracy": mean("elemental_accuracy"),
        f"{prefix}/hit_accuracy": mean("hit_accuracy"),
        f"{prefix}/episode_length": mean("episode_length"),
        f"{prefix}/hold_frac": mean("hold_frac"),
        f"{prefix}/move_frac": mean("move_frac"),
        f"{prefix}/shots_per_100": mean("shots_per_100"),
        f"{prefix}/deaths": mean("deaths"),
    }
    for key in sorted({k for r in rows for k in r.get("reward_components", {})}):
        metrics[f"{prefix}/rc_{key}"] = float(np.mean([r["reward_components"].get(key, 0.0) for r in rows]))
    return metrics


def reset_weapon_head(model: PPO) -> None:
    """Zero the weapon branch of the action head so it restarts from a uniform distribution.

    Used when transferring a Stage 1 policy (one enemy element, so the weapon head
    collapsed onto Water) to Stage 2, where the element must be chosen per target.
    """
    nvec = [int(n) for n in model.action_space.nvec]
    lo = sum(nvec[:-1])
    with torch.no_grad():
        model.policy.action_net.weight[lo:].zero_()
        model.policy.action_net.bias[lo:].zero_()


class SingleAgentEvalCallback(BaseCallback):
    def __init__(
        self,
        stage: int,
        eval_freq: int,
        n_eval_episodes: int,
        logger: ExperimentLogger,
        max_steps: int | None = None,
        optuna_trial: Any = None,
        env_overrides: Dict[str, Any] | None = None,
    ):
        super().__init__(verbose=0)
        self.env_overrides = env_overrides
        self.stage = stage
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.exp_logger = logger
        self.max_steps = max_steps
        self.optuna_trial = optuna_trial
        self.best_return = -float("inf")
        self.last_eval_step = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            self.evaluate()
        return True

    def evaluate(self) -> None:
        self.last_eval_step = self.num_timesteps
        elapsed = max(1e-6, time.time() - self.start_time)
        metrics = evaluate_single_agent(
            self.stage,
            lambda _env, obs: self.model.predict(obs, deterministic=True)[0],
            self.n_eval_episodes,
            "val",
            self.max_steps,
            env_overrides=self.env_overrides,
        )
        metrics["train/fps"] = float(self.num_timesteps / elapsed)
        self.exp_logger.log_metrics(self.num_timesteps, metrics)
        if metrics["eval/return_mean"] > self.best_return:
            self.best_return = metrics["eval/return_mean"]
            self.model.save(str(self.exp_logger.ckpt_dir / "best.zip"))
        print(
            f"[Step {self.num_timesteps:>7d}] Return: {metrics['eval/return_mean']:6.2f} "
            f"| Win: {metrics['eval/win_rate']*100:5.1f}% | Kills: {metrics['eval/kills']:4.1f} "
            f"| ElemAcc: {metrics['eval/elemental_accuracy']*100:5.1f}% | FPS: {metrics['train/fps']:.0f}",
            flush=True,
        )
        if self.optuna_trial is not None and self.num_timesteps > 0:
            import optuna

            self.optuna_trial.report(metrics["eval/return_mean"], step=self.num_timesteps)
            if self.optuna_trial.should_prune():
                raise optuna.TrialPruned()


def make_env_fn(stage: int, seed: int, rank: int, max_steps: int | None, env_overrides=None):
    def _init():
        env = DotShotSingleEnv(stage=stage, max_steps=max_steps, env_overrides=env_overrides)
        env.reset(seed=seed + rank * 1000)
        return Monitor(env)

    return _init


def run_experiment(
    cfg: Dict[str, Any],
    seed: int,
    device: str,
    steps_override: int | None = None,
    run_dir_override: str | None = None,
    optuna_trial: Any = None,
    init_model_path: str | None = None,
) -> Dict[str, Any]:
    """Train (or evaluate a scripted policy) and score the final policy on held-out seeds.

    Periodic evaluations use the ``val`` split; the final policy is scored once on
    ``final_eval_split`` (``test`` by default, ``val`` while tuning).
    """
    seed_everything(seed)
    configure_torch(device)

    algo = cfg["algo"].lower()
    stage = int(cfg["stage"])
    max_steps = cfg.get("max_steps", None)
    total_steps = int(steps_override if steps_override is not None else cfg["total_steps"])
    n_eval = int(cfg.get("n_eval_episodes", 16))
    n_test = int(cfg.get("n_test_episodes", 32))
    final_split = str(cfg.get("final_eval_split", "test"))
    final_prefix = "test" if final_split == "test" else "final_val"
    eval_freq = int(cfg.get("eval_freq", max(10_000, total_steps // 10)))
    exp_name = cfg.get("exp_name", f"stage{stage}_{algo}")
    run_dir = run_dir_override or os.path.join("runs", f"stage{stage}", algo, exp_name, f"seed_{seed}")
    init_model_path = init_model_path or cfg.get("init_model_path")
    overrides = cfg.get("env_overrides")

    resolved_cfg = {**cfg, "seed": seed, "device": device, "total_steps": total_steps}
    if init_model_path:
        resolved_cfg["init_model_path"] = init_model_path
    logger = ExperimentLogger(run_dir, resolved_cfg)

    try:
        if algo in ("random", "heuristic"):
            rng = np.random.default_rng(seed)
            fn = (
                (lambda env, _obs: rng.integers(0, env.action_space.nvec).astype(np.int64))
                if algo == "random"
                else (lambda env, _obs: heuristic_policy_action(env))
            )
            metrics = evaluate_single_agent(stage, fn, n_test, final_split, max_steps, prefix=final_prefix,
                                            env_overrides=overrides)
            logger.log_metrics(total_steps, metrics)
            ret = metrics[f"{final_prefix}/return_mean"]
            print(f"[{algo.upper()} seed={seed}] {final_prefix} return {ret:.2f}")
            return {"run_dir": run_dir, "final_return": float(ret), "final_metrics": metrics}

        num_envs = int(cfg.get("num_envs", 8))
        # In-process envs: on Windows, subprocess IPC costs more than a simulator step.
        vec_type = cfg.get("vec_env", "dummy")
        env_fns = [make_env_fn(stage, seed, i, max_steps, overrides) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns) if (vec_type == "subproc" and num_envs > 1) else DummyVecEnv(env_fns)

        lr = float(cfg.get("lr", 3e-4))
        lr_schedule = cosine_lr_schedule(lr, min_lr=1e-5) if cfg.get("cosine_lr", True) else lr
        arch = [int(h) for h in cfg.get("net_arch", [128, 128])]
        policy_kwargs = {"net_arch": dict(pi=list(arch), vf=list(arch)), "ortho_init": True}

        if algo == "ppo":
            if init_model_path:
                model = PPO.load(
                    init_model_path,
                    env=vec_env,
                    device=device,
                    custom_objects={"learning_rate": lr_schedule},
                )
                model.learning_rate = lr_schedule
                model.lr_schedule = lr_schedule if callable(lr_schedule) else (lambda _: lr)
                model.ent_coef = float(cfg.get("ent_coef", model.ent_coef))
                model.gamma = float(cfg.get("gamma", model.gamma))
                model.gae_lambda = float(cfg.get("gae_lambda", model.gae_lambda))
                model.n_epochs = int(cfg.get("n_epochs", model.n_epochs))
            else:
                model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=lr_schedule,
                    n_steps=int(cfg.get("n_steps", 512)),
                    batch_size=int(cfg.get("batch_size", 256)),
                    n_epochs=int(cfg.get("n_epochs", 8)),
                    gamma=float(cfg.get("gamma", 0.99)),
                    gae_lambda=float(cfg.get("gae_lambda", 0.95)),
                    clip_range=float(cfg.get("clip_range", 0.2)),
                    ent_coef=float(cfg.get("ent_coef", 0.01)),
                    vf_coef=float(cfg.get("vf_coef", 0.5)),
                    max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    device=device,
                    verbose=0,
                )
        elif algo == "a2c":
            model = A2C(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_schedule,
                n_steps=int(cfg.get("n_steps", 16)),
                gamma=float(cfg.get("gamma", 0.99)),
                gae_lambda=float(cfg.get("gae_lambda", 0.95)),
                ent_coef=float(cfg.get("ent_coef", 0.01)),
                vf_coef=float(cfg.get("vf_coef", 0.5)),
                max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
                policy_kwargs=policy_kwargs,
                seed=seed,
                device=device,
                verbose=0,
            )
        else:
            raise ValueError(f"Unsupported single-agent algo: {algo}")

        if init_model_path and cfg.get("reset_weapon_head", False):
            reset_weapon_head(model)

        act_fn = lambda _env, obs: model.predict(obs, deterministic=True)[0]  # noqa: E731
        cb = SingleAgentEvalCallback(stage, eval_freq, n_eval, logger, max_steps, optuna_trial=optuna_trial,
                                     env_overrides=overrides)
        if optuna_trial is None:
            init_metrics = evaluate_single_agent(stage, act_fn, n_eval, "val", max_steps, env_overrides=overrides)
            logger.log_metrics(0, init_metrics)
            cb.best_return = float(init_metrics["eval/return_mean"])
            model.save(str(logger.ckpt_dir / "best.zip"))
        start = time.time()
        try:
            model.learn(total_timesteps=total_steps, callback=cb, progress_bar=False)
            if cb.last_eval_step != cb.num_timesteps:
                cb.evaluate()
            model.save(str(logger.ckpt_dir / "final.zip"))
            final_metrics = evaluate_single_agent(stage, act_fn, n_test, final_split, max_steps, prefix=final_prefix,
                                                  env_overrides=overrides)
            final_metrics["train/wall_time_s"] = time.time() - start
            logger.log_metrics(total_steps, final_metrics)
        finally:
            vec_env.close()
    finally:
        logger.close()
    return {
        "run_dir": run_dir,
        "best_val_return": float(cb.best_return),
        "final_metrics": final_metrics,
        "final_return": float(final_metrics[f"{final_prefix}/return_mean"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-Agent Benchmark Runner for DotShot Game v2")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="PyTorch device")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for total timesteps")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional output directory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    run_experiment(cfg, seed=args.seed, device=device, steps_override=args.steps, run_dir_override=args.run_dir)


if __name__ == "__main__":
    main()
