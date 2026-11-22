---
name: experiment-reproducibility-optuna
description: >-
  Configure YAML-driven RL/MARL experiments, structured telemetry logging,
  Git SHA and hardware snapshots, cross-device portability, and Optuna
  hyperparameter optimization studies.
---

# Experiment Reproducibility & Optuna HPO

Use this skill when authoring YAML configs under `configs/`, telemetry loggers under `benchmarks/utils/`, or hyperparameter sweeps.

---

## 1. Zero Ad-Hoc CLI Runs

Every benchmark run must resolve from a YAML configuration file under `configs/` (with explicit CLI overrides only for `--seed`, `--device`, and optional `--steps` smoke-test caps). Hardcoding hyperparameters inside Python scripts breaks cross-machine comparison.

---

## 2. Standard Run Directory Layout

Write every training run to:

```text
runs/<stage>/<algo>/<exp_name>/seed_<k>/
|-- config.yaml        # Full resolved config + git_sha + git_dirty + hardware + torch/cuda versions
|-- metrics.jsonl      # One JSON line per evaluation or log interval
|-- tb/                # TensorBoard SummaryWriter events
|-- checkpoints/
    |-- best.pt        # Best evaluation IQM / Win Rate checkpoint
    |-- final.pt       # End-of-training checkpoint
```

Always record in `config.yaml`:
- `seed`
- `git_sha` (`git rev-parse HEAD`) and `git_dirty` boolean
- `python_version`, `torch_version`, `cuda_device` (or `"cpu"`), `cpu_count`
- `env_id` / `stage`, `reward_alpha`, `partial_obs`

---

## 3. Dual Telemetry (`metrics.jsonl` + TensorBoard)

Log identical core keys across all algorithms (`SB3 PPO`, `IPPO`, `QMIX`, `MAPPO`, `HAPPO`, `MAT`) so plotting scripts parse one uniform schema:
- `step` (total environment frames/steps across all agents or env steps—document clearly as `env_steps`)
- `eval/return_mean`, `eval/return_std`
- `eval/win_rate`
- `eval/core_hp_ratio`
- `eval/elemental_accuracy`
- `eval/hit_accuracy`
- `eval/synergy_combos`
- `train/fps`, `train/policy_loss`, `train/value_loss`, `train/entropy`

---

## 4. Optuna HPO Discipline

- Store SQLite studies outside tracked Git history (`*.db` ignored in `.gitignore`).
- Always evaluate Optuna trial objectives over at least **2 seeds** or deterministic multi-episode evaluation (`n_eval_episodes >= 16`) so hyperparameter rankings do not overfit a single lucky seed.
- Export winning hyperparameter sets cleanly into `configs/` YAML files rather than leaving them buried in `.db` files.
