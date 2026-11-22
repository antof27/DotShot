# DotShot Game v2

**DotShot Game v2** is a tactical 2D single-to-multi-agent Reinforcement Learning (RL) simulator (`dotshot/`), paired with a reproducible benchmark suite (`benchmarks/`) and a living English MARL research handbook (`docs/`). Agents navigate a `1440×960` cover-filled arena, switch across three counter-elements (**Water beats Fire**, **Grass beats Water**, **Fire beats Grass**), defend a Central Energy Core, and coordinate to shatter periodic **Shielded Bosses** under Line-of-Sight (LOS) partial observability.

---

## Visual Overview (Stages 3 & 4)

| Stage 3: Cooperative Core Defense & Boss Overlay | Stage 4: Heterogeneous Squad (`Vanguard`, `Striker`, `Support`) |
| :---: | :---: |
| ![Stage 3 Cooperative Boss Overlay](docs/assets/screenshots/stage3_coop_boss_overlay.png) | ![Stage 4 Heterogeneous Squad](docs/assets/screenshots/stage4_heterogeneous_squad.png) |

Pressing `Tab` during manual play or evaluation toggles the **MARL Explainability Overlay**, rendering 8-ray obstacle LIDAR, aim lasers, and live `MAT`/`PMAT` execution-priority badges (`#1`, `#2`, `#3`).

---

## Four-Stage Curriculum

| Stage | Environment / API | Agents & Roles | Hazards & Objective |
| :---: | :--- | :--- | :--- |
| **Stage 1** | `DotShot-Stage1-v1` (Gymnasium) | 1 `Striker` | `Fire` enemies (max 4), 4 cover pillars, kiting & target leading |
| **Stage 2** | `DotShot-Stage2-v1` (Gymnasium) | 1 `Striker` | `Water/Grass/Fire` enemies (max 6), incoming shots, neutral allies |
| **Stage 3** | `DotShotParallelEnv(stage=3)` (PettingZoo) | 2 `Striker` agents | Mixed waves (max 8), Shielded Bosses, 6 cover blocks, LOS masking, Core defense |
| **Stage 4** | `DotShotParallelEnv(stage=4)` (PettingZoo) | `Vanguard`, `Striker`, `Support` | Heavy waves (max 10), Shielded Bosses, asymmetric class abilities, Core defense |

- **Fixed `129`-dim observation vector** across all stages (`Self 14 + Core 5 + LIDAR 8 + Top-5 Hostiles 75 + Allies 27`) plus privileged global state `env.state()` for Centralized Training for Decentralized Execution (CTDE).
- **Factored action space** `MultiDiscrete([3, 3, 32, 4])` (`dx`, `dy`, `11.25°` aim bin, and hold/ability or `Water`/`Grass`/`Fire`).

---

## Windows CUDA Setup (`.venv`)

All simulator tests and training scripts target Python 3.11 inside the project `.venv` with CUDA enabled (verified on NVIDIA GeForce RTX 3060, PyTorch `2.6.0+cu124`):

```powershell
# Create and activate the virtual environment (Miniconda3 Python 3.11)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch CUDA + package in editable mode
pip install -e ".[dev]"
```

---

## Manual Play & Checkpoint Replay

Play any stage interactively with keyboard and mouse, or replay/record a trained checkpoint (`.zip` for Stable-Baselines3, `.pt` for MARL):

```powershell
# Play Stage 3 cooperatively (controls Agent 0 with WASD + mouse; 1/2/3 elements; Space/Click fire; Tab overlay)
.\.venv\Scripts\python.exe enjoy.py --stage 3 --manual

# Evaluate a trained MARL checkpoint or export an MP4 clip headlessly
.\.venv\Scripts\python.exe enjoy.py --stage 3 --model runs\stage3\mat\stage3_mat\seed_0\checkpoints\final.pt --record docs\assets\videos\dotshot_stage3_demo.mp4 --fps 30
```

---

## Running Single-Agent & MARL Benchmarks

Every experiment is driven by versioned YAML configs under `configs/`:

```powershell
# Single-agent baselines and SB3 PPO / A2C (Stages 1–2)
.\.venv\Scripts\python.exe -m benchmarks.runners.run_single_agent --config configs\stage1_ppo.yaml --seed 0 --device auto
.\.venv\Scripts\python.exe -m benchmarks.runners.run_single_agent --config configs\stage2_heuristic.yaml --seed 0 --device cpu

# Cooperative MARL algorithms (Stages 3–4: IPPO, VDN, QMIX, MAPPO, HAPPO, HATRPO, MAT, PMAT, RoMAT)
.\.venv\Scripts\python.exe -m benchmarks.marl.train_marl --config configs\stage3_mat.yaml --seed 0 --device cuda
.\.venv\Scripts\python.exe -m benchmarks.marl.train_marl --config configs\stage3_qmix.yaml --seed 0 --device cuda
.\.venv\Scripts\python.exe -m benchmarks.marl.train_marl --config configs\stage4_romat.yaml --seed 0 --device cuda

# Aggregate multi-seed IQM + 95% bootstrap CIs and plot 300 DPI learning curves
.\.venv\Scripts\python.exe -m benchmarks.eval.evaluate --stage-dir runs\stage3 --out-json runs\stage3\eval_summary.json
.\.venv\Scripts\python.exe -m benchmarks.eval.plot_results --stage-dir runs\stage3 --out docs\assets\figures\stage3_marl_smoke_comparison.png --metrics eval/return_mean eval/core_hp_ratio eval/elemental_accuracy

# Run the full Milestone 2 5-seed MARL sweeps and alpha/LOS ablations
.\.venv\Scripts\python.exe scripts\run_m2_suite.py --part all --workers 3 --device cuda
```

---

## Milestone 2 Highlights (Reward Shaping v2.1)

- **Converged Stage 1 (`200k` steps, 3 seeds)**: Smooth `(best_dot ** 2) * 0.25` aim shaping lifts SB3 `PPO` to **`38.9%` win rate** and **`89.5%` elemental accuracy** (`3.31 [1.72, 6.41]` return IQM).
- **Stage 3 Homogeneous Co-op (`60k` env steps, 5 seeds)**: `Branching-QMIX` leads at **`-1.72 [-1.97, -1.42]`** return IQM and **`32.0%` Core HP** survival, doubling on-policy Core survival (`10.1%–17.0%`).
- **Stage 4 Heterogeneous Squad (`60k` env steps, 5 seeds)**: `RoMAT` achieves the top final return IQM (**`-2.27 [-2.47, -2.16]`**) and the tightest 95% CI across asymmetric `Vanguard`, `Striker`, and `Support` roles ([Stage 4 Replay](docs/assets/videos/dotshot_stage4_heterogeneous.mp4)).
- **Throughput & Ablations**: Pre-cached LOS geometry (`24,388 steps/sec` single-env) and `DummyMARLEnv` (`2,349 env_steps/sec` vs `~470 FPS` `SubprocMARLEnv` on Windows) power 3-seed $\alpha \in [0, 1]$ and LOS occlusion ablations in [Chapter 4](docs/04-experiments-log.md).

---

## Research Handbook, Scientific Report & Quality Gates

- **Docsify Handbook**: Serve chapters 1–4 (RL-to-MARL foundations, CTDE taxonomy, simulator spec, and Milestone 1–2 empirical logs) locally:
  ```powershell
  python -m http.server 8080 --directory docs
  ```
- **LaTeX Scientific Report**: [docs/paper/dotshot_marl_report.tex](docs/paper/dotshot_marl_report.tex) (with verified BibTeX references in [docs/references.bib](docs/references.bib)).
- **Simulator & MARL Unit Tests**:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
- **Documentation, Citation & Humanizer Lint Gate**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/validate_docs.py
  ```
