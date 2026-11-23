# DotShot Game v2

**DotShot Game v2** is a tactical 2D single-to-multi-agent Reinforcement Learning (RL) simulator (`src/dotshot/`), paired with a reproducible benchmark suite (`src/benchmarks/`) and a living English MARL research handbook (`docs/`). Agents navigate a `1440×960` cover-filled arena, switch across three counter-elements (**Water beats Fire**, **Grass beats Water**, **Fire beats Grass**), defend a Central Energy Core, and coordinate to shatter periodic **Shielded Bosses** under Line-of-Sight (LOS) partial observability.

---

## Repository Structure

```text
RL-game1/
├── src/
│   ├── dotshot/        # Simulator physics (core/), Gymnasium & PettingZoo envs (envs/), Pygame renderer (render/)
│   └── benchmarks/     # Single-agent SB3 runners, modular CTDE MARL algos, rliable evaluator & plotter
├── configs/            # Versioned YAML experiment definitions (Stage 1–4)
├── third_party/        # Pinned PKU-MARL HARL and Multi-Agent-Transformer submodules
├── docs/               # Docsify SOTA handbook (01..04), BibTeX, LaTeX paper/, results/, assets/ (figures, screenshots, videos, legacy_v1_plots)
├── scripts/            # enjoy.py (manual play / MP4 capture), run_m2_suite.py, plot_m2_ablations.py, validate_docs.py
├── tests/              # pytest simulator & CUDA MARL verification suite
├── legacy_v1/          # Archived DotShot v1 prototype (env/, marl/, models/, v1 CLI scripts)
└── .agents/            # Workspace agent skills (.agents/skills/) & architecture/roadmap specs (.agents/docs/)
```

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

Play any stage interactively with keyboard and mouse, or replay/record a trained checkpoint (`.zip` for Stable-Baselines3, `.pt` for MARL, or a checkpoint directory from the official HARL/MAT runners):

```powershell
# Play Stage 3 cooperatively (controls Agent 0 with WASD + mouse; 1/2/3 elements; Space/Click fire; Tab overlay)
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --manual

# Replay the official Stage 3 MAT checkpoint
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --official-mat runs\m3_official\stage3\official_mat\seed_0\checkpoints\best

# Replay an official HARL checkpoint (MAPPO shown; swap the folder for HAPPO, HATRPO, or HAA2C)
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --official-harl runs\m3_official\stage3\harl_mappo\seed_0\checkpoints\best

# Evaluate a trained MARL checkpoint or export an MP4 clip headlessly
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --model runs\stage3\mat\stage3_mat\seed_0\checkpoints\final.pt --record docs\assets\videos\dotshot_stage3_demo.mp4 --fps 30

# Record the official Stage 4 MAT policy
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 4 --official-mat runs\m3_official\stage4\official_mat\seed_0\checkpoints\best --record docs\assets\videos\m3_official_stage4_best.mp4 --fps 30
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

## Milestone 3 Highlights (Official SOTA and Optuna BOHB)

Milestone 3 runs pinned upstream implementations on the MARL stages and Stable-Baselines3 on Stages 1 and 2. `third_party/HARL` (Zhong et al., 2024) covers MAPPO, HAPPO, HATRPO, and HAA2C on the native `MultiDiscrete([3, 3, 32, 4])` space, while `third_party/Multi-Agent-Transformer` (Wen et al., 2022) runs MAT through the 8-D `Box` adapter in `src/benchmarks/official/mat_dotshot_env.py`.

- Optuna BOHB ran 16 trials per study with `TPESampler(multivariate=True, seed=42)` and `HyperbandPruner(reduction_factor=3)`. Best objectives are `14.57` for Stage 1 PPO, `9.54` for Stage 2 transfer, `-0.777` for Stage 3 MAT, and `-2.799` for Stage 3 HARL MAPPO on the 2-agent team-sum objective (`-1.40` per agent).
- Stage 1 PPO reaches `14.95` return IQM, `50.0%` win rate, `100.0%` elemental accuracy, and `8.7` kills per episode over 1,000,000 steps and 3 seeds.
- Stage 2 `PPO_TRANSFER` reaches `8.91` return IQM and `66.7%` win rate, with a higher win rate than scratch `PPO` (`2.81`, `13.9%`) and the un-evasive `HEURISTIC` (`0.0%`).
- `OFFICIAL_MAT` leads final return on Stage 3 (`-1.62 [-1.67, -1.56]`) and Stage 4 (`-2.29 [-2.34, -2.19]`). On the asymmetric Stage 4 squad, `HARL_HAA2C` (`-2.50`), `HARL_HATRPO` (`-2.53`), and `HARL_HAPPO` (`-2.54`) all beat `HARL_MAPPO` (`-2.80`).
- Replays: [Stage 3](docs/assets/videos/m3_official_stage3_best.mp4) (`357` frames, Core HP `76.7%`) and [Stage 4](docs/assets/videos/m3_official_stage4_best.mp4) (`450` frames, Score `300`, Core HP `86.7%`). The compiled report is [dotshot_marl_report.pdf](docs/paper/dotshot_marl_report.pdf), and the full IQM tables are in [Chapter 4](docs/04-experiments-log.md).

---

## Research Handbook, Scientific Report & Quality Gates

- **Docsify Handbook**: Serve chapters 1–4 (RL-to-MARL foundations, CTDE taxonomy, simulator spec, and Milestone 1–3 empirical logs) locally:
  ```powershell
  python -m http.server 8080 --directory docs
  ```
- **Scientific Report**: [LaTeX source](docs/paper/dotshot_marl_report.tex) and [compiled PDF](docs/paper/dotshot_marl_report.pdf) (with verified BibTeX references in [docs/references.bib](docs/references.bib)).
- **Simulator & MARL Unit Tests**:
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/ -v
  ```
- **Documentation, Citation & Humanizer Lint Gate**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/validate_docs.py
  ```
