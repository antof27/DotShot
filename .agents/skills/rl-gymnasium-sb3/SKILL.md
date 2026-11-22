---
name: rl-gymnasium-sb3
description: >-
  Design, verify, and train single-agent Gymnasium environments and
  Stable-Baselines3 (PPO, A2C) pipelines on Windows with CUDA, deterministic
  seeding, reward-shaping guardrails, and Optuna hyperparameter optimization.
---

# RL Gymnasium & Stable-Baselines3 Engineering

Use this skill when building or modifying `dotshot/envs/single_agent.py`, reward shaping modules, or single-agent benchmark scripts under `benchmarks/runners/`.

---

## 1. Gymnasium API & Determinism Checklist

Every single-agent environment (`DotShotSingleEnv`) must pass `gymnasium.utils.env_checker.check_env(env, skip_render_check=False)` with zero warnings:

- **RNG Hygiene**: Never call global `np.random.*` or `random.*` inside environment physics or spawners. Always call `super().reset(seed=seed)` and derive all stochastic draws from `self.np_random`.
- **Exact Space Contracts**:
  - Return observations as `np.float32` matching `observation_space.shape` and finite bounds (clamp or normalize features inside `[-1.0, 1.0]` or explicit bounds).
  - Keep `MultiDiscrete` action parsing defensive against integer types (`np.int32`/`np.int64`) and batch dimensions.
- **Termination vs Truncation**:
  - `terminated = True` strictly on domain end states (agent death, Core destroyed, wave/objective cleared).
  - `truncated = True` strictly when `step_count >= max_steps` (so RL algorithms bootstrap $V(s_{T})$ correctly on time limits rather than treating timeouts as zero-value death).
- **Info Telemetry**: Populate `info` at episode end (or every step) with exact diagnostic scalars: `episode_return`, `hits_correct`, `hits_wrong`, `shots_fired`, `kills`, `damage_taken`, `cover_usage_ratio`, `win`.

---

## 2. Reward Shaping Guardrails

- **Do not obscure the true objective.** Keep sparse task signals (kills, Core survival, wave clear) dominant over dense shaping terms (alignment, distance).
- **Respect Potential-Based Reward Shaping (Ng et al., 1999)**: Any dense guidance potential $\Phi(s)$ should satisfy $F(s, a, s') = \gamma \Phi(s') - \Phi(s)$ when policy invariance is desired. Avoid unbounded cumulative bonuses per step that reward stalling ("camping" without engaging enemies).
- **Log raw components.** Always keep unscaled component sums in `info["reward_breakdown"]` so reward hacking (e.g., spinning in place to farm aim alignment without shooting) is caught immediately in TensorBoard.

---

## 3. Windows + CUDA Training Standards (SB3)

- **Multiprocessing Guard**: On Windows, `SubprocVecEnv` spawns fresh Python interpreters. Wrap every training entry point in `if __name__ == "__main__":` and set `torch.set_num_threads(1)` in worker processes to prevent CPU core thrashing.
- **Device Choice**:
  - Small MLPs (`64x64` or `128x128`) with fast CPU simulation often run fastest with `SubprocVecEnv` on CPU workers feeding a CUDA learner or `cuda` device for larger batches (`n_steps * num_envs >= 4096`). Benchmark throughput (FPS) explicitly before locking defaults.
- **PPO Best Practices**:
  - Use orthogonal initialization (`ortho_init=True`), tuned `clip_range` (`0.1`–`0.2`), `gae_lambda=0.95`, and cosine or linear learning-rate decay.
  - Pair `VecNormalize` (when raw observations are unscaled) with synchronized normalization stats during `EvalCallback` (`training=False, norm_reward=False`).
- **Baseline Anchors**: Always train and log a `RandomPolicy` and a clean `RuleBasedHeuristic` alongside RL agents so normalized scores show whether PPO actually exceeds scripted target-leading.

---

## 4. Reproducibility Contract

Every run script must:
1. Accept `--config <yaml>` and `--seed <int>`.
2. Seed `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all`, and `env.reset(seed=seed)`.
3. Save resolved `config.yaml`, `git_sha`, Python/PyTorch/CUDA versions, and `metrics.jsonl` inside `runs/<stage>/<algo>/<exp>/seed_<k>/`.
