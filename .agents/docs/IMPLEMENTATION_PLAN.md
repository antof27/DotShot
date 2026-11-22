# Implementation Plan: DotShot Game v2

**Status:** Approved Roadmap (Milestone 0 Active)  
**Target Repository:** `antof27/RL-game1`  
**Related Specs:** [PRD.md](PRD.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [ROADMAP.md](ROADMAP.md)

---

## Phase Overview

Work progresses in lockstep across four milestones (`M0 -> M3`). Each milestone ships verifiable code, automated tests, reproducible experiment artifacts, and updated handbook chapters together.

```mermaid
flowchart LR
    M0["M0: Governance, Skills & Planning Docs"] --> M1["M1: CUDA Env + DotShot v2 + MARL Engine + SOTA Handbook + Stage 1-3 Initial Benchmarks + Notelia Skill"]
    M1 --> M2["M2: Full 5-Seed Stage 3 & Stage 4 Heterogeneous MARL Sweeps + Ablations"]
    M2 --> M3["M3: Curriculum Surgery v2, HPO, Replay Gallery & GitHub Pages Release"]
```

---

## Milestone 0: Governance, Skills, and Planning Docs (Current Pass)

**Objective:** Establish permanent workspace rules, modular agent skills, and engineering blueprints before modifying runtime code.

### Deliverables
- Git branch `feat/dotshot-v2-foundation` created safely off `main`.
- Root [AGENTS.md](../../AGENTS.md).
- Five workspace skills under `.agents/skills/`: `humanizer`, `notelia-docs`, `rl-gymnasium-sb3`, `marl-benchmark-design`, `pygame-simulator-craft`.
- Six planning documents under `.agents/docs/`: `PRE_PRD.md`, `PRD.md`, `IMPLEMENTATION_PLAN.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `ROADMAP.md`.

### Verification
- Automated script confirms all 12 files exist, YAML skill frontmatter is valid, zero relative Markdown links are broken, and zero banned Humanizer clichés or non-English terms appear.

---

## Milestone 1: Core Platform, Simulator v2, Handbook, and Initial Benchmarks

**Objective:** Stand up the Windows CUDA environment, build DotShot Game v2 and the modular MARL engine, publish the complete SOTA theoretical handbook + verified bibliography, run Stage 1–2 single-agent and Stage 3 MARL smoke-test benchmarks, and contribute the portable English Notelia skill upstream.

### Step 1.1: Windows CUDA Virtual Environment & Package Setup
- Create Python 3.11 environment at `RL/RL-game1/.venv` via Miniconda3 (`C:\Users\anton\Miniconda3\Scripts\conda.exe create -p RL\RL-game1\.venv python=3.11 -y`).
- Install PyTorch CUDA wheels (`cu124`/`cu126` compatible with RTX 3060), `gymnasium`, `pettingzoo`, `stable-baselines3[extra]`, `pygame`, `optuna`, `pyyaml`, `matplotlib`, `seaborn`, `scipy`, `imageio[ffmpeg]`, `tensorboard`, `pytest`.
- Create `pyproject.toml`, update `requirements.txt` and `.gitignore`.
- **Gate**: `.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"` outputs `NVIDIA GeForce RTX 3060`.

### Step 1.2: DotShot Game v2 Simulator (`dotshot/`)
- `dotshot/core/config.py`: Stage 1–4 dataclasses and arena obstacle presets.
- `dotshot/core/arena.py`: `1440×960` arena, circle-AABB obstacle collision, fast segment-AABB LOS raycasting, 8-ray LIDAR.
- `dotshot/core/entities.py`: `Agent` (with `Vanguard`, `Striker`, `Support` class stats & abilities), `CentralCore`, `Enemy` (`Water`, `Grass`, `Fire`), `ShieldedBoss`, `Projectile`, `NeutralAlly`.
- `dotshot/core/rewards.py`: Decomposed reward engine $r_i^t = (1-\alpha)r_{i,\text{indiv}}^t + \alpha r_{\text{team}}^t$.
- `dotshot/envs/single_agent.py`: `DotShotSingleEnv` (`DotShot-Stage1-v1`, `DotShot-Stage2-v1`).
- `dotshot/envs/multi_agent.py`: `DotShotParallelEnv` (`DotShot-Stage3-v1`, `DotShot-Stage4-v1`) implementing PettingZoo `ParallelEnv` and CTDE `state()`.
- `dotshot/render/pygame_renderer.py` & `enjoy.py`: 60 FPS neon tactical renderer, pre-cached glow surfaces, HUD, `Tab` MARL explainability overlay, manual play, and MP4 export.
- **Gate**: `pytest tests/test_single_env.py tests/test_multi_env.py` passes `check_env`, `parallel_api_test`, bitwise seed determinism, LOS blocking, elemental counters, boss shield break, and heterogeneous class mechanics.

### Step 1.3: Reproducible Benchmark Engine (`benchmarks/` & `configs/`)
- `benchmarks/utils/seeding.py`, `benchmarks/utils/logger.py`: Deterministic seeding, config + Git SHA + hardware snapshot, TensorBoard + `metrics.jsonl`.
- `benchmarks/runners/run_single_agent.py`: `Random`, `RuleBasedHeuristic`, `SB3 A2C`, `SB3 PPO`.
- `benchmarks/marl/networks.py` & `buffer.py`: `FactoredMultiDiscreteActor`, `CentralizedCritic` (orthogonal init + running ValueNorm), `BranchingQNetwork`, `VDNMixer`, `QMIXMixer`, `FactoredMATNetwork` (`MAT`, `PMAT`, `RoMAT`), on-policy rollout buffer, and episode replay buffer.
- `benchmarks/marl/algos/`: `ippo.py`, `qmix.py`, `mappo.py`, `happo.py` (HAPPO + HATRPO sequential advantage decomposition), `mat.py` (MAT + PMAT + RoMAT), unified CLI `train_marl.py`.
- `benchmarks/eval/evaluate.py` & `plot_results.py`: Multi-seed evaluator computing Mean, Median, IQM, and 95% stratified bootstrap CIs, writing PNG plots, MP4 clips, and Markdown tables.
- **Gate**: `pytest tests/test_marl_algos.py` passes forward/backward and training update smoke tests on CUDA for every algorithm.

### Step 1.4: Living Docsify SOTA Handbook (`docs/` & `scripts/validate_docs.py`)
- `docs/index.html`, `docs/_sidebar.md`, `docs/README.md` (Docsify + KaTeX + Mermaid).
- `docs/references.bib`: Verified BibTeX entries for foundational RL, CTDE, value factorization, centralized-critic PG, MAPPO, HAPPO/HATRPO/HARL, MAT/PMAT/RoMAT, and statistical evaluation papers.
- `docs/01-evolution-rl-to-marl.md`: MDP to Dec-POMDP mathematical trajectory and independent learner pathologies.
- `docs/02-sota-marl-taxonomy.md`: Deep SOTA survey (intuition, math, comparative tables, diagrams, pitfalls) across all algorithm families.
- `docs/03-dotshot-simulator-spec.md`: Formal Dec-POMDP specification of DotShot v2 and v1-to-v2 engineering post-mortem.
- `docs/04-experiments-log.md`: Experiment protocol and Milestone 1 empirical results.
- `scripts/validate_docs.py`: Automated citation, link, asset, and Humanizer style validator.
- **Gate**: `.\.venv\Scripts\python.exe scripts/validate_docs.py` exits 0 with zero errors.

### Step 1.5: Milestone 1 Reproducible Experiments
- **Single-Agent Stage 1 & Stage 2** (3 seeds each on CUDA/SubprocVecEnv): `Random` vs `RuleBasedHeuristic` vs `SB3 A2C` vs `SB3 PPO`.
- **MARL Stage 3 Smoke-Test Comparison** (short aligned budget, 2–3 seeds on CUDA): `IPPO` vs `Branching-QMIX` vs `MAPPO` vs `HAPPO` vs `MAT` verifying full pipeline convergence, saved checkpoints, MP4 replay clips, and auto-generated figures/tables injected into `docs/04-experiments-log.md`.

### Step 1.6: Notelia Upstream Skill Contribution (`C:\Users\anton\Documents\Notelia`)
- Checkout `dev`, pull `origin/dev`, create `feature/notelia-portable-skill-en`.
- Add portable `.agents/skills/notelia/` (integrating English Scientific Humanizer + Docsify/Markdown + BibTeX validation alongside LaTeX/Typst workflows) and automated white-box/black-box tests (`tests/test_notelia_skill.py`).
- Merge `--no-ff` into local `dev` only after 100% test pass; delete local feature branch; do not push without explicit user instruction.

---

## Milestone 2: Full Multi-Seed MARL Benchmark & Ablations

**Objective:** Execute full compute-budget multi-seed evaluations across Stage 3 (Homogeneous) and Stage 4 (Heterogeneous) to empirically test the SOTA hypotheses documented in Chapter 2.

### Deliverables
- **Stage 3 Full 5-Seed Sweep**: `IPPO`, `Branching-VDN`, `Branching-QMIX`, `MAPPO`, `HAPPO`, `HATRPO`, `MAT`, `PMAT`.
- **Stage 4 Heterogeneous 5-Seed Sweep** (`Vanguard` / `Striker` / `Support`): `IPPO`, `MAPPO` (with and without role IDs), `HAPPO`, `HATRPO`, `MAT`, `PMAT`, `RoMAT`.
- **Controlled Ablations**:
  - Credit-assignment mixing $\alpha \in \{0.0, 0.25, 0.5, 0.75, 1.0\}$.
  - Partial observability (full visibility vs LOS-occluded local vision cone).
  - Factored `MultiDiscrete` action heads vs v1 flattened `2304` action head.
- Full IQM + 95% bootstrap CI plots, synergy heatmaps, and failure analyses added to `docs/04-experiments-log.md`.

---

## Milestone 3: Transfer, Tuning, and Release Polish

**Objective:** Complete curriculum transfer studies, automated hyperparameter tuning, and repository-wide release verification.

### Deliverables
- Curriculum Weight Surgery v2 (bootstrapping Stage 3/4 critics and actors from Stage 2 single-agent policies).
- Optuna multi-objective MARL HPO (`scripts/` + `configs/`) balancing Win Rate, Core HP, and Sample Efficiency.
- Interactive Docsify polish, curated MP4 replay gallery in `docs/assets/videos/`, and final pre-push audit across `RL/RL-game1` and `Notelia`.
