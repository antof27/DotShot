# Product Requirements Document (PRD): DotShot Game v2

**Status:** Approved Baseline (Milestone 0)  
**Target Repository:** `antof27/RL-game1`  
**Related Specs:** [PRE_PRD.md](PRE_PRD.md) | [ARCHITECTURE.md](ARCHITECTURE.md) | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

---

## 1. Product Vision

**DotShot Game** is both an engaging 2D tactical top-down arena shooter and an academic-grade Multi-Agent Reinforcement Learning (MARL) benchmark and handbook. Players and RL agents fight across a `1440×960` cover-filled arena using elemental weapons (`Water`, `Grass`, `Fire`) where tactical positioning, target leading, shield breaking, and role coordination determine survival. Every simulator mechanic maps cleanly onto formal MDP and Dec-POMDP abstractions documented in the repository's English Docsify handbook (`docs/`).

---

## 2. Functional Requirements (`FR`)

### 2.1 Simulator & 4-Stage Progressive Curriculum (`dotshot/`)

- **FR-1 (Tactical Arena & Physics)**: The simulator shall run a deterministic `1440×960` 2D arena containing configurable rectangular cover obstacles with circle-AABB collision resolution and segment-AABB Line-of-Sight (LOS) occlusion for projectiles and perception.
- **FR-2 (Elemental Counter System)**: The simulator shall implement a cyclic elemental advantage graph (`Water > Fire > Grass > Water`). Matching hits deal full elemental damage; mismatched hits deal zero damage (or heavily reduced damage on non-shielded targets if configured) and incur an explicit wrong-element penalty; shield-bearing bosses require counter-element hits to shatter their elemental shield before taking full hull damage.
- **FR-3 (4-Stage Curriculum)**:
  - **Stage 1 (`DotShot-Stage1-v1`)**: Single-agent Striker vs Fire enemies on a cover-equipped arena; evaluates basic movement, cover peeking, target leading, and trigger discipline.
  - **Stage 2 (`DotShot-Stage2-v1`)**: Single-agent Striker vs all three enemy elements + enemy projectiles + neutral wandering allies; evaluates dynamic weapon switching, dodging, and friendly-fire avoidance.
  - **Stage 3 (`DotShot-Stage3-v1`)**: Homogeneous Co-op (2–3 Striker agents) defending a Central Energy Core against multi-element waves and periodic Elemental-Shielded Bosses; evaluates focus-fire, Core protection, and cross-agent credit assignment.
  - **Stage 4 (`DotShot-Stage4-v1`)**: Heterogeneous Co-op (3 distinct classes: `Vanguard`, `Striker`, `Support`) defending the Central Energy Core under heavier wave pressure; evaluates asymmetric role specialization where homogeneous parameter sharing is suboptimal.
- **FR-4 (Factored Action Space)**: Each agent shall act over `MultiDiscrete([3, 3, 32, 4])` representing `move_x` (`-1, 0, +1`), `move_y` (`-1, 0, +1`), `aim_bin` (32 discrete angles at `11.25°` increments), and `weapon_action` (`0: Hold/Ability, 1: Water, 2: Grass, 3: Fire`).
- **FR-5 (CTDE Observation & State Contract)**:
  - Single-agent stages (`1–2`) shall comply with Gymnasium `gym.Env` (`env_checker.check_env`).
  - Multi-agent stages (`3–4`) shall comply with PettingZoo `ParallelEnv` (`parallel_api_test`), exposing decentralized local observations $o_i^t$ (capped by vision/LOS radius and local LIDAR) and a privileged global state $s^t$ via `env.state()` exclusively for centralized training.
- **FR-6 (Decomposed Reward Mixing)**: The reward module shall compute explicit individual and team components:
  $$r_i^t = (1 - \alpha) r_{i,\text{indiv}}^t + \alpha r_{\text{team}}^t, \qquad \alpha \in [0, 1]$$
  and expose the unscaled per-component breakdown inside `info["reward_breakdown"]`.

### 2.2 Graphics, Manual Play & Explainability (`dotshot/render/`)

- **FR-7 (Visual Craft at 60 FPS)**: The Pygame renderer shall run decoupled from physics at `1440×960` (scalable window), featuring pre-cached neon glow sprites, beveled cover blocks with drop shadows, colorblind-safe element geometry, distinct class silhouettes (`Vanguard` hexagon + shield arc, `Striker` chevron + laser sight, `Support` diamond + repair ring), boss shield bars, floating combat/combo callouts, damped screen shake, and a clean translucent HUD.
- **FR-8 (Manual & Policy Playback via `enjoy.py`)**: Users shall be able to play any stage manually (`WASD` + mouse aim + `1/2/3` + click/space, plus class ability trigger) and watch trained SB3 (`.zip`) or MARL (`.pt`) checkpoints with a toggleable (`Tab`) **MARL Explainability Overlay** (LOS/LIDAR ticks, aim locks, MAT/PMAT priority order `#1..#n`, and attention arcs).
- **FR-9 (Video Export)**: Headless evaluation shall export deterministic MP4/GIF gameplay clips via `rgb_array` rendering into `docs/assets/videos/`.

### 2.3 Benchmark & Experiment Suite (`benchmarks/`)

- **FR-10 (Single-Agent Suite)**: Support `Random`, `Rule-Based Heuristic` (target-leading + counter switching), `SB3 A2C`, and `SB3 PPO` (with optional ICM/DPBRS curiosity and Optuna tuning hooks).
- **FR-11 (Modular MARL Suite)**: Implement a CleanRL/HARL-inspired PyTorch suite with factored action heads supporting:
  - `IPPO` (decentralized baseline),
  - `Branching-VDN` and `Branching-QMIX` (value factorization with nonnegative mixing weights and episode replay buffer),
  - `MAPPO` (CTDE centralized critic + ValueNorm/PopArt + death masking),
  - `HAPPO` and `HATRPO` (sequential random-permutation updates using the Multi-Agent Advantage Decomposition Lemma without parameter sharing),
  - `MAT`, `PMAT`, and `RoMAT` (factored Transformer encoder-decoder with causal action conditioning, Plackett-Luce ordering, and role embeddings).
- **FR-12 (Statistical Aggregation & Auto-Reporting)**: Provide a multi-seed evaluation CLI that computes Mean, Median, Interquartile Mean (IQM), and 95% stratified bootstrap confidence intervals across episodes/seeds, auto-generating PNG figures (`docs/assets/figures/`) and Markdown tables for the handbook.

### 2.4 Living SOTA Handbook & Quality Gates (`docs/` & `scripts/`)

- **FR-13 (Docsify Handbook)**: Serve a 100% English Docsify site (`docs/index.html`, `_sidebar.md`, `README.md`) with KaTeX math and Mermaid diagrams across four core chapters:
  - `01-evolution-rl-to-marl.md` (Single-agent RL foundations -> Dec-POMDPs -> independent learner failures),
  - `02-sota-marl-taxonomy.md` (Full CTDE taxonomy: VDN/QMIX/QTRAN/QPLEX, MADDPG/COMA/MA-POCA, IPPO/MAPPO, HAPPO/HATRPO/HARL, MAT/PMAT/RoMAT, 2024–2026 frontiers & evaluation theory),
  - `03-dotshot-simulator-spec.md` (Formal Dec-POMDP spec + v1-to-v2 engineering post-mortem),
  - `04-experiments-log.md` (Living incremental benchmark log updated at each milestone).
- **FR-14 (Verified Bibliography & Validator)**: Maintain `docs/references.bib` with verified primary citations only, enforced by `scripts/validate_docs.py` (citation resolution, broken-link check, and Humanizer cliché linter).

---

## 3. Non-Functional Requirements (`NFR`)

- **NFR-1 (Reproducibility)**: Every run executes from a YAML config + explicit integer seed (`random`, `numpy`, `torch`, env) and writes `config.yaml`, Git SHA, hardware specs, TensorBoard events, and `metrics.jsonl` into `runs/<stage>/<algo>/<exp>/seed_<k>/`.
- **NFR-2 (Headless Throughput)**: Importing and stepping `dotshot.envs` headlessly must never import `pygame` or require a display server.
- **NFR-3 (Windows CUDA Compatibility)**: The project environment (`RL/RL-game1/.venv`, Python 3.11) must execute `SubprocVecEnv` and PyTorch MARL training cleanly on Windows with NVIDIA CUDA (`RTX 3060 12GB`).
- **NFR-4 (Prose & Language Purity)**: 100% English across all files; zero Humanizer cliché lint failures.
- **NFR-5 (Git Safety & Upstream Parity)**: Legacy v1 files stay preserved on branch history; no commits directly to `main`; upstream Notelia contributions adhere strictly to Notelia's Gitflow and automated tests.

---

## 4. Milestone Acceptance Criteria

| Milestone | Scope | Exit Gate Criteria |
| :--- | :--- | :--- |
| **M0 (Current)** | Governance, Skills, Planning Docs | `AGENTS.md`, 5 `SKILL.md` files, and 6 `.agents/docs/*.md` files pass structural, link, English, and Humanizer lint checks on branch `feat/dotshot-v2-foundation`. |
| **M1** | CUDA `.venv` + DotShot v2 + Benchmark Engine + SOTA Handbook + Initial Benchmarks + Notelia Skill | CUDA verified on RTX 3060; `pytest` passes `check_env`, `parallel_api_test`, physics determinism, and all MARL algorithm smoke tests; `scripts/validate_docs.py` passes with 0 errors; Stage 1–2 3-seed benchmarks + Stage 3 MARL smoke-test comparison populate `docs/04-experiments-log.md`; Notelia `dev` branch passes 100% tests. |
| **M2** | Full Multi-Seed MARL Sweeps & Ablations | 5-seed Stage 3 (`IPPO` vs `Branching-QMIX` vs `MAPPO` vs `HAPPO` vs `MAT`) & Stage 4 (`MAPPO` vs `HAPPO` vs `HATRPO` vs `MAT` vs `PMAT` vs `RoMAT`) completed with IQM + 95% CI plots, $\alpha$ credit-assignment ablations, and LOS ablations documented. |
| **M3** | Advanced Frontiers & Release Polish | Curriculum weight-surgery v2, Optuna MARL HPO, interactive replay gallery, and GitHub Pages publication review. |
