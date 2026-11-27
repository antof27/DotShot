# DotShot

DotShot is a 2D arena shooter for single- and multi-agent reinforcement learning, with a benchmark suite and a research handbook in the same repository. Agents move, aim in 32 directions and fire one of three elemental weapons (Water beats Fire, Grass beats Water, Fire beats Grass) at waves of enemies. Four stages add one difficulty at a time:

| Stage | Interface | Agents | What it adds |
| :---: | :--- | :--- | :--- |
| 1 | Gymnasium | 1 Striker | aiming, leading targets, using cover |
| 2 | Gymnasium | 1 Striker | three enemy elements, enemy fire, neutral allies to avoid |
| 3 | PettingZoo | 2 Strikers | defending a central Core with limited vision, shielded bosses |
| 4 | PettingZoo | Vanguard, Striker, Support | heterogeneous roles with abilities |

| Stage 3 | Stage 4 |
| :---: | :---: |
| ![Stage 3](docs/assets/screenshots/stage3_coop_boss_overlay.png) | ![Stage 4](docs/assets/screenshots/stage4_heterogeneous_squad.png) |

Every agent acts in `MultiDiscrete([3, 3, 32, 4])` (horizontal move, vertical move, aim, weapon) once every 8 simulator ticks and observes a 166-value vector. Centralised critics also get a global state. The full specification, including every reward term, is in [docs/03-dotshot-simulator-spec.md](docs/03-dotshot-simulator-spec.md).

## Results

Final policies, 5 seeds each, 32 held-out test episodes per seed. Return: mean over seeds [95% bootstrap CI]. "Cloned" runs start from an actor cloned from a scripted expert and train 4M ticks; "scratch" runs train from random weights.

| Stage | Policy | Return | Win rate |
| :---: | :--- | :---: | :---: |
| 1 | PPO, 4M ticks | 43.15 [39.85, 45.69] | 100.0% |
| 2 | PPO from scratch, 4M ticks | 22.46 [21.35, 23.29] | 93.1% |
| 2 | PPO from the Stage 1 policy, 4M ticks | 44.17 [42.46, 46.34] | 78.1% |
| 3 | best cloned: HAA2C | 84.87 [82.48, 87.26] | 63.1% |
| 3 | best scratch (16M ticks): MAPPO | 22.68 [16.91, 28.83] | 1.2% |
| 3 | scripted expert (96 episodes) | 87.59 [83.97, 91.15] | 78.1% |
| 4 | best cloned: MAPPO | **107.90** [106.69, 109.11] | **88.1%** |
| 4 | best scratch (16M ticks): MAPPO | 34.32 [31.38, 37.58] | 5.6% |
| 4 | scripted expert (96 episodes) | 100.17 [96.89, 103.34] | 75.0% |

- Stage 1 is solved, and transfer from Stage 1 doubles the Stage 2 return.
- Cloning the expert and fine-tuning with RL reaches the expert on Stage 3 and beats it on Stage 4.
- From scratch, the learners that share parameters across agents (MAPPO, MAT) learn fastest, but 16M ticks are not enough to defend the Core reliably.
- On the heterogeneous Stage 4, the shared-actor MAPPO beats HAPPO and HATRPO, the methods designed for heterogeneous teams.

Replays of won held-out test episodes: [Stage 3, MAT](docs/assets/videos/stage3_mat.mp4) and [Stage 4, MAPPO](docs/assets/videos/stage4_mappo.mp4).

Every learner, tables, learning curves, behaviour analysis and screenshots: [experiment log](docs/04-experiments-log.md) and [report](docs/paper/dotshot_marl_report.pdf).

## Learners

| Stages | Learner | Implementation |
| :---: | :--- | :--- |
| 1–2 | PPO | Stable-Baselines3 |
| 3–4 | MAPPO, HAPPO, HATRPO, HAA2C | [PKU-MARL/HARL](https://github.com/PKU-MARL/HARL), pinned submodule |
| 3–4 | MAT | [PKU-MARL/Multi-Agent-Transformer](https://github.com/PKU-MARL/Multi-Agent-Transformer), pinned submodule |

The upstream code is never edited. Adapters and corrections live in `src/benchmarks/official/`, and each correction is covered by a test. Every learner and every scripted baseline is scored by the same evaluator on held-out episode seeds.

## Repository Layout

```text
src/dotshot/        simulator: engine, rewards, Gymnasium and PettingZoo envs, Pygame renderer
src/benchmarks/     learners, scripted baselines, behaviour cloning, Optuna tuning, evaluation
configs/            official/ (tuned final configs), baselines/, ablations/
scripts/            run_jobs.py, monitor_runs.py, report_final.py, enjoy.py, ...
tests/              simulator, adapter and regression tests
docs/               handbook chapters, LaTeX report, curated results, screenshots
third_party/        HARL and Multi-Agent-Transformer (git submodules)
```

## Setup (Windows, Python 3.11)

```powershell
git clone --recursive https://github.com/antof27/DotShot.git
cd DotShot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

PyTorch with CUDA is only needed for MAT; the simulator, SB3 and HARL run faster on the CPU.

## Usage

```powershell
# Tests
.\.venv\Scripts\python.exe -m pytest

# Play a stage yourself (WASD, mouse aim, 1/2/3 elements, click or Space to fire, Tab for the overlay)
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --manual

# Scripted baselines (96 test episodes per policy)
.\.venv\Scripts\python.exe -m benchmarks.runners.run_marl_baselines --config configs\baselines\stage3_baselines.yaml

# Train a tuned configuration on 5 seeds (resumable)
.\.venv\Scripts\python.exe scripts\run_jobs.py --config configs\official\s3_harl_mappo_bc.yaml --seeds 0-4 --out runs\final\stage3 --workers 5

# Replay a trained run on a held-out test episode, or record it
.\.venv\Scripts\python.exe scripts\enjoy.py --run runsinal\stage4\s4_harl_mappo_bc\seed_0
.\.venv\Scripts\python.exe scripts\enjoy.py --run runsinal\stage4\s4_harl_mappo_bc\seed_0 --record stage4.mp4

# Watch training curves and behaviour flags
.\.venv\Scripts\python.exe scripts\monitor_runs.py runs\final\stage3

# Tables and learning curves
.\.venv\Scripts\python.exe scripts\report_final.py --stage 3 --runs runs\final\stage3 --baselines runs\baselines\stage3 --out docs\results\final_stage3.json --figure docs\assets\figures\final_stage3_curves.png

# Hyperparameter search for one stage and learner
.\.venv\Scripts\python.exe -m benchmarks.official.bohb_tune --study s3_harl_mappo --device cpu
```

Run commands from the repository root with `PYTHONPATH=src` if the package is not installed.

## Documentation

- [Handbook](docs/README.md): from single-agent RL to cooperative MARL, the simulator specification and the experiment log. Serve it locally with `python -m http.server 8080 --directory docs`.
- [Report](docs/paper/dotshot_marl_report.pdf) ([LaTeX source](docs/paper/dotshot_marl_report.tex)).
- [Architecture](.agents/docs/ARCHITECTURE.md), [design decisions](.agents/docs/DECISIONS.md), [roadmap](.agents/docs/ROADMAP.md).
