# Pre-PRD: DotShot Game v2 Research Platform

**Document Status:** Approved Baseline (Milestone 0)  
**Target Repository:** `antof27/RL-game1` (`RL/RL-game1`)  
**Companion Upstream Repo:** `Prv2106/Notelia` (`C:\Users\anton\Documents\Notelia`)  
**Primary Language:** 100% English (code, documentation, skills, experiment logs)

---

## 1. Origin and Motivation

Most Reinforcement Learning tutorials separate simulator engineering from algorithm benchmarking and academic literature. A developer builds a toy gridworld, runs one PPO script with an arbitrary seed, and writes a short `README.md` that omits the mathematics, baseline comparisons, and failure modes.

**DotShot Game** takes the opposite path. It treats the game simulator, the multi-agent RL benchmark suite, and the research handbook as one tightly coupled system:

1. **Incremental Skill Progression**: Agents first master single-agent mechanics (movement, target leading, cover, elemental counters, dodging) before entering cooperative multi-agent scenarios that require coordinated fire, Core defense, and heterogeneous role specialization.
2. **Algorithmic Coverage Grounded in Theory**: Instead of running one algorithm in isolation, the project benchmarks single-agent baselines (`Random`, `Heuristic`, `SB3 A2C`, `SB3 PPO`) alongside the primary cooperative MARL families (`IPPO`, `Branching-VDN/QMIX`, `MAPPO`, `HAPPO/HATRPO`, `MAT/PMAT/RoMAT`).
3. **Living SOTA Handbook**: Every mathematical formulation, architectural shift, and empirical experiment goes straight into a Docsify research handbook under `docs/`, built with verified citations and automated quality gates adapted from Notelia.

---

## 2. Audit of DotShot v1 (Current Codebase)

Inspecting [RL/RL-game1](../../) and the companion literature repo (`RL/RL-Literature`) revealed a solid early prototype with four structural bottlenecks that limit both gameplay depth and MARL research validity.

| Area | Current v1 State (Observed) | Root Limitation for v2 Research |
| :--- | :--- | :--- |
| **Runtime Environment** | Existing `.venv/` contains `bin/`, built on macOS. `requirements.txt` mixes `gymnasium>=0.29.1` and `google-generativeai`. | Cannot execute on Windows. No CUDA wheel configuration for the local RTX 3060 GPU, and lacks `pettingzoo`, `pyyaml`, `scipy`, and `pytest`. |
| **Arena & Physics** | [env/game_env.py](../../env/game_env.py) models an empty `800×800` square with 3 enemy classes (`Water`, `Grass`, `Fire`), bullet in-flight capping, and wandering neutral `Ally` units. | Zero obstacles, zero cover geometry, and zero line-of-sight blocking. Without spatial occlusion or shared tactical objectives, decentralized agents gain little from real coordination. |
| **Multi-Agent Design** | Level 3 switches `self.num_agents = 2`, returning a concatenated `140`-dim observation vector inside a single Gymnasium `Box` and `MultiDiscrete` space. | Does not implement PettingZoo `ParallelEnv` or expose a clean centralized state `s^t` distinct from local observations $o_i^t$. Both agents share identical stats and lack distinct class roles. |
| **MARL Action Encoding Bug** | [marl/train_marl.py](../../marl/train_marl.py#L20-L39) flattens `MultiDiscrete([3, 3, 64, 4])` into a single `2304`-way discrete integer action (`a*768 + b*256 + c*4 + d`). | Destroys directional locality across the 64 aim bins and forces a 2,304-class softmax per agent, inflating policy-gradient variance and slowing convergence. |
| **Renderer Performance** | [env/renderer.py](../../env/renderer.py) allocates a new `pygame.Surface(..., pygame.SRCALPHA)` on every bullet, particle, and glow draw call each frame. | Per-frame surface allocations hurt frame pacing under heavy combat and mix rendering details closely with environment state. |
| **Experiment Rigor** | [train.py](../../train.py) and `marl/train_marl.py` use ad-hoc CLI flags, single runs, and manual reward scaling in [env/reward_function.py](../../env/reward_function.py). | No YAML config snapshots, no multi-seed aggregation (`IQM` + bootstrap CIs), no scripted heuristic anchor, and no automated pipeline linking metrics to documentation. |
| **Literature Assets** | `RL/RL-Literature/` holds `RL_Curriculum.md`, `paradigma.md`, and 27 arXiv PDFs in a separate repo. | Disconnected from `RL-game1` experiments; does not cover `HAPPO`/`HATRPO`/`HARL`/`MAT` comparative derivations or DotShot empirical results. |

---

## 3. Workstation and Environment Baseline

All local verification and Milestone 1 training will target this verified workstation configuration:

- **OS**: Windows (PowerShell)
- **GPU**: NVIDIA GeForce RTX 3060 (12,288 MiB VRAM, Driver 617.42 / CUDA 13.4 UMD)
- **Python Distribution**: Miniconda3 installed at `C:\Users\anton\Miniconda3` (base Python 3.13.12; target project venv: Python 3.11 CUDA at `RL/RL-game1/.venv`)
- **Version Control**: Git `2.53.0.windows.3` (`git` available; GitHub CLI `gh` not installed globally)
- **Local Companion Repo**: `C:\Users\anton\Documents\Notelia` (`main` and `origin/dev` checked out locally)

---

## 4. Settled Design Decisions (From Project Interview)

During the initial design alignment, six core project choices were locked in:

1. **Single Unified Repository**: Evolve `antof27/RL-game1` directly so simulator code, MARL benchmarks, agent skills, and the SOTA handbook live together in one repository.
2. **100% English + Humanizer + Notelia**: Author every document, comment, skill, and report in English following `blader/humanizer` anti-cliché rules and Notelia's pedagogical and citation-verification standards. Also contribute improvements back to `Prv2106/Notelia` via its strict Gitflow (`dev` -> `feature/*`).
3. **4-Stage Tactical Arena (`1440×960`)**: Expand the map to a `1440×960` tactical arena with obstacles, cover, line-of-sight, and 4 progressive stages: Single-Agent Fundamentals -> Single-Agent Elemental Mastery -> Homogeneous Co-op (Core Defense + Elemental Shield Boss) -> Heterogeneous Co-op (`Vanguard`, `Striker`, `Support`).
4. **Standard APIs + Modular PyTorch Engine**: Expose `Gymnasium` for Stages 1–2 and `PettingZoo ParallelEnv` for Stages 3–4; build a modular CleanRL/HARL-inspired PyTorch benchmark suite covering `SB3 PPO/A2C`, `IPPO`, `Branching-VDN/QMIX`, `MAPPO`, `HAPPO/HATRPO`, and `MAT/PMAT/RoMAT`.
5. **Portable Local Experiment Tracking**: Store versioned YAML configs, TensorBoard logs, JSONL telemetry, `rliable`-style multi-seed statistics (`IQM` + 95% CIs), publication figures, and MP4 replays directly in the project layout.
6. **Living Docsify Handbook (`docs/`)**: Serve the SOTA MARL theory, mathematical derivations, verified BibTeX references, simulator post-mortem, and experiment tables via Docsify in `docs/` (GitHub Pages ready).

---

## 5. Risks, Mitigations, and Non-Goals

### Key Technical Risks

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| **Reward hacking** (agents camping behind pillars or spinning to farm aim shaping) | Agents score high shaping reward while letting the Core die. | Keep sparse task terms dominant, log raw reward decomposition per component, and evaluate strictly on task metrics (Win Rate, Core HP, Kills, Elemental Accuracy). |
| **Slow Python step loop bottlenecking GPU training** | RTX 3060 sits idle waiting on single-process Pygame/NumPy stepping. | Decouple `pygame` completely from `dotshot/core/`, vectorize raycasts/collisions in NumPy, and use `SubprocVecEnv` with `torch.set_num_threads(1)`. |
| **Credit assignment noise in Stage 3–4** | `MAPPO` and `QMIX` fail to distinguish individual utility from team luck. | Expose explicit reward mixing parameter $\alpha \in [0, 1]$ and run controlled $\alpha$-ablations alongside `HAPPO` and `MAT`. |
| **Unverified citations in SOTA survey** | Wrong arXiv IDs, years, or author lists undermine academic credibility. | Cross-check every citation against primary sources/PDFs and enforce `scripts/validate_docs.py` before any doc update closes. |

### Explicit Non-Goals (Out of Scope)

- **No 3D physics engine (Unity/MuJoCo/Godot)**: DotShot stays a crisp, deterministic 2D top-down simulator where physics speed enables rapid multi-seed MARL sweeps on a single RTX 3060.
- **No mandatory cloud tracking dependency**: Weights & Biases or remote databases are not required; all tracking remains local, inspectable, and portable across machines via Git.
- **No LLM-in-the-loop agent control in M1**: M1 focuses strictly on pure RL/MARL policy learning and SOTA benchmark rigor (LLM reasoning/GRPO extensions are documented theoretically or deferred beyond M3).
