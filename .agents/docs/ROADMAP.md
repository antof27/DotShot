# Living Project Roadmap & Parity Tracker

This roadmap tracks progress across all four coupled pillars of **DotShot Game v2**: Simulator, Benchmark Suite, Living SOTA Handbook, and Upstream Notelia Collaboration. No milestone is marked complete until all columns for that row pass verification.

---

## Milestone Status Overview

| Milestone | Scope Summary | Simulator (`dotshot/`) | Benchmarks (`benchmarks/`) | Handbook (`docs/`) | Upstream (`Notelia`) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M0** | Governance, Skills, Planning Docs | Audit complete | Audit complete | Structure planned | `origin/dev` inspected | **Complete** |
| **M1** | Windows CUDA `.venv` + DotShot v2 + MARL Engine + SOTA Handbook + Initial Benchmarks + Notelia Skill | Stages 1–4 + Renderer + Tests | SB3 SA + Modular MARL Suite + `rliable` Eval + Stage 1–3 M1 Runs | Chapters 1–4 + `references.bib` + Validator + M1 Results | Portable EN skill + tests merged to local `dev` | **Complete** |
| **M2** | Full 5-Seed MARL Sweeps & Credit-Assignment Ablations | Reward v2.1 + pre-cached LOS + `DummyMARLEnv` verified | Converged Stage 1–2 + 5-seed Stage 3 & Stage 4 sweeps + $\alpha$/LOS ablations | Complete Milestone 2 results, 300 DPI figures & Stage 4 MP4 in Ch. 3–4 | Doc validator gate passing (`0` errors) | **Complete** |
| **M3** | Transfer Surgery v2, MARL HPO & Release Polish | Curriculum weight surgery v2 + MP4 gallery | Optuna MARL HPO presets | GitHub Pages launch & final citation audit | Release tag parity check | **Next** |

---

## Detailed Milestone Checklists

### Milestone 0: Foundation, Governance & Skills (Complete)
- [x] Safe feature branch `feat/dotshot-v2-foundation` created in `RL/RL-game1`.
- [x] Root governance rules authored in [AGENTS.md](../../AGENTS.md).
- [x] 5 workspace skills authored under `.agents/skills/` (`humanizer`, `notelia-docs`, `rl-gymnasium-sb3`, `marl-benchmark-design`, `pygame-simulator-craft`).
- [x] 6 engineering planning docs authored under `.agents/docs/` (`PRE_PRD`, `PRD`, `IMPLEMENTATION_PLAN`, `ARCHITECTURE`, `DECISIONS`, `ROADMAP`).
- [x] Automated validation of M0 files (existence, YAML frontmatter, links, Humanizer/English lint).

### Milestone 1: Platform Build & Initial Benchmarks (Complete)
- [x] Windows Python 3.11 CUDA `.venv` built via Miniconda3 and verified on RTX 3060.
- [x] `dotshot/` v2 simulator implemented (`1440×960`, cover, LOS, Stages 1–4, Gymnasium + PettingZoo `ParallelEnv`, 60 FPS renderer + `Tab` overlay).
- [x] `pytest` suite passing `check_env`, `parallel_api_test`, seed determinism, and physics/role tests.
- [x] `benchmarks/` single-agent (`Random`, `Heuristic`, `SB3 A2C`, `SB3 PPO`) + modular MARL (`IPPO`, `Branching-VDN/QMIX`, `MAPPO`, `HAPPO/HATRPO`, `MAT/PMAT/RoMAT`) passing CUDA smoke tests.
- [x] `docs/` Docsify SOTA handbook (Chapters 1–4 + verified `references.bib`) passing `scripts/validate_docs.py`.
- [x] Stage 1–2 Single-Agent (3 seeds) + Stage 3 MARL smoke-test comparison executed; figures, MP4 clip, and IQM tables published in `docs/04-experiments-log.md`.
- [x] Upstream Notelia feature branch (`feature/notelia-portable-skill-en`) tested 100% and merged `--no-ff` to local `dev`.

### Milestone 2: Full Multi-Seed MARL Sweeps & Ablations (Complete)
- [x] Reward Shaping v2.1 (`(best_dot ** 2) * 0.25` aim gradient, hold-fire farming removal, flat `-0.04` miss penalty) + pre-cached LOS (`24,388 steps/sec`) + `DummyMARLEnv` (`2,349 env_steps/sec` on Windows).
- [x] Converged 200k-step Stage 1 & Stage 2 single-agent benchmarks (`HEURISTIC`, `PPO`, `A2C`, `RANDOM`, 3 seeds).
- [x] 5-seed Stage 3 homogeneous benchmark (`QMIX`, `HAPPO`, `MAPPO`, `MAT`, `IPPO`, 60k env steps, CUDA).
- [x] 5-seed Stage 4 heterogeneous benchmark (`RoMAT`, `MAPPO`, `HAPPO`, `IPPO`, 60k env steps, CUDA) + Stage 4 MP4 replay.
- [x] Controlled credit-assignment $\alpha$ ablation (`0.0, 0.25, 0.5, 0.75, 1.0`) and LOS partial-observability ablation (`LOS_OFF` vs `LOS_ON`, 3 seeds, 50k steps, CUDA).
- [x] 300 DPI `rliable` figures, Stage 4 MP4 replay, and quantitative + qualitative analysis published in `docs/03-dotshot-simulator-spec.md`, `docs/04-experiments-log.md`, and `README.md`.

### Milestone 3: Transfer, Tuning & GitHub Release (Next)
- [ ] Stage 2 -> Stage 3/4 checkpoint weight surgery v2 validated.
- [ ] Optuna MARL hyperparameter tuning configs verified.
- [ ] Docsify GitHub Pages site live-tested + final humanizer & BibTeX gate before push.
