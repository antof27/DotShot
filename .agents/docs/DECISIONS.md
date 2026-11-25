# Architectural Decision Records

The decisions that shape DotShot and its benchmark suite as they stand today. Each record gives the context, the decision, the alternatives that were rejected and the consequences. Records that no longer hold were removed when the repository was cleaned up; they remain in the git history.

---

## ADR-001: One Repository for Simulator, Benchmarks and Handbook
- **Status:** Accepted
- **Context:** Simulator changes, experiment results and the written analysis drift apart quickly when they live in different places.
- **Decision:** One repository holds the simulator (`src/dotshot/`), the learners and evaluation (`src/benchmarks/`), the configs, the Docsify handbook and the LaTeX report (`docs/`). Packages use the PyPA `src/` layout. Everything runs from the project-local Python 3.11 environment `.venv` on the Windows workstation (RTX 3060 12 GB).
- **Alternatives Rejected:** Separate repositories for code and documentation.
- **Consequences:** A change to the simulator, its results and its description land in the same branch. The lockstep rule in `AGENTS.md` makes that mandatory.

## ADR-002: English Prose, Verified Citations, Automated Doc Checks
- **Status:** Accepted
- **Context:** Unverified citations and generic prose undermine a research handbook.
- **Decision:** All code and documentation are in English and follow `.agents/skills/humanizer` and `.agents/skills/notelia-docs`. Every citation must resolve to a verified entry in `docs/references.bib`; `scripts/validate_docs.py` checks citations and links.
- **Alternatives Rejected:** Bilingual documentation; unchecked bibliographies.
- **Consequences:** A missing or invented citation fails the doc check.

## ADR-003: Arena, Curriculum and Factored Actions
- **Status:** Accepted
- **Context:** Algorithm differences only show up in a task that needs aiming, element choice, cooperation and role specialisation, and a flat action index over all combinations (2,304 values) explores poorly.
- **Decision:** A 1440 × 960 arena with cover and line-of-sight occlusion, four stages that add one difficulty at a time (aiming, element choice, cooperative Core defence, heterogeneous roles), Gymnasium for Stages 1–2 and PettingZoo `ParallelEnv` with an explicit `state()` for Stages 3–4. Every agent acts in `MultiDiscrete([3, 3, 32, 4])`, and all learners use one categorical head per branch.
- **Alternatives Rejected:** An empty arena; a flat 2,304-way action space; continuous actions.
- **Consequences:** The same 166-value local observation and action space on all stages, so a Stage 1 policy can initialise a Stage 2 run unchanged.

## ADR-004: Pinned Upstream Learners, Patched From Outside
- **Status:** Accepted
- **Context:** Reimplemented algorithms invite doubts about parity with the published versions.
- **Decision:** Single-agent stages use Stable-Baselines3 PPO. Multi-agent stages use `PKU-MARL/HARL` (commit `099d29e`: MAPPO, HAPPO, HATRPO, HAA2C) and `PKU-MARL/Multi-Agent-Transformer` (commit `be3ff49`) as git submodules under `third_party/`, never edited. Adapters and every correction live in `src/benchmarks/official/`, and each correction is pinned by a test in `tests/test_marl_fixes.py`.
- **Alternatives Rejected:** Copying upstream code into `src/`; editing the submodules in place; in-house reimplementations.
- **Consequences:** Updating an upstream learner is an explicit submodule bump followed by the test suite.

## ADR-005: Shared Team Reward, EP Critic and Corrected Adapters
- **Status:** Accepted
- **Context:** An audit of the first HARL/MAT integration found that HARL optimised r⁴ instead of r for multi-discrete actions (the joint log-prob was stored in four columns and the ratios multiplied), built a (B, B) entropy tensor, never masked dead agents, trained each agent on its own α-mixed reward while the critic saw one global state, and ran Stage 4 MAPPO unshared. MAT acted through a continuous Gaussian adapter, and its evaluation over-sampled short episodes.
- **Decision:** Stages 3–4 use `reward_mode: team`: every agent receives `mean_i r_indiv,i + r_team`, and all centralised learners use an EP critic on the global state (the α mix stays available as an ablation). HARL runs with `action_aggregation: mean`, a masked per-branch entropy, dead agents reported as done (never all at once mid-episode), linear LR decay and a shared MAPPO actor. MAT uses factored categorical heads and bootstraps time-limit truncations.
- **Alternatives Rejected:** Storing per-branch log-probs in the HARL buffer (same result, bigger diff); keeping per-agent rewards (violates the shared-reward assumption of HAPPO and HATRPO).
- **Consequences:** Results from the first integration are withdrawn and not compared with current ones.

## ADR-006: DotShot v2.2 Rules
- **Status:** Accepted
- **Context:** On v2.1 a policy that walked into enemies and never fired beat every baseline and learner, because the death of all agents ended the episode and stopped the Core penalties. Agents could not see enemy projectiles. Aim shaping paid every tick whether or not the agent fired and out-paid correct hits 3.6 to 1. Element choice was only rewarded on impact. No scripted policy won Stages 3–4 reliably.
- **Decision:** On Stages 3–4 dead agents respawn after 150 ticks, only the Core's death ends an episode early, and reaching the limit with the Core alive is a terminal victory paying 20 × Core HP fraction (terminal rather than truncated, because HARL drops truncated rewards). The local observation adds incoming projectiles, status, Core threat and crosshair blocks (129 → 166 values); the global state adds timers and 16 projectiles. Aim shaping is paid only on firing ticks, toward the lead point of the priority threat. Each shot at a target earns +1.5 for the counter element and −0.75 otherwise, zero in expectation for a random weapon. Agents may keep 2 bullets in flight per target. Stage 4 enemy projectiles deal 17 damage, calibrated against `heuristic_calibrated`.
- **Alternatives Rejected:** A terminal penalty for the expected remaining Core loss when everyone dies (harder to read than respawn); a time-limit truncation with bootstrapping (hides the win bonus from HARL); a symmetric ±0.6 element credit (made agents hold fire on 40% of actions).
- **Consequences:** v2.1 checkpoints do not load. `charge` becomes the worst baseline. At k = 8 the calibrated script wins 78.1% (Stage 3) and 75.0% (Stage 4) of 96 `test` episodes; `noop`, `random` and `charge` win none.

## ADR-007: Eight Ticks per Decision and a Behaviour Audit
- **Status:** Accepted
- **Context:** A shot leaves only every 12–16 ticks, so at one decision per tick the aim and weapon branches mattered on about one tick in twelve and their gradient was mostly noise. Policy inference also took about 40% of training wall time.
- **Decision:** Agents act every `action_repeat = 8` ticks on all stages; the action is held and rewards are summed. Budgets are reported in ticks. Every evaluation logs `hold_frac`, `move_frac`, `shots_per_100`, `deaths` and `core_dist`, and `scripts/monitor_runs.py` flags idle, non-firing, suicidal and element-blind policies.
- **Evidence:** Stage 2 PPO wins 96.9% of `test` episodes after 2M ticks at k = 8, 62.5% at k = 4, and 6.2% after 1M ticks at k = 1. HARL-MAPPO trains 3× faster per tick.
- **Alternatives Rejected:** Masking irrelevant branches in the upstream losses (needs per-branch masks in both buffers); shorter fire cooldowns (changes the game).
- **Consequences:** Dodging is coarser (one decision moves an agent up to 38 px).

## ADR-008: Warm Starts as Separate Results
- **Status:** Accepted
- **Context:** From random weights, Stage 3 learners need about 24M ticks to defend the Core, and Stage 2 PPO learns element choice slowly.
- **Decision:** Two warm starts are supported and always reported as separate rows: Stage 2 PPO initialised from the Stage 1 policy with a reset weapon head (`transfer`), and Stage 3–4 actors cloned from 200 episodes of `heuristic_calibrated` followed by 10 critic-only updates before RL (`bc`).
- **Alternatives Rejected:** Reporting only the best of the two; distillation from an SB3 teacher (implemented, not used in the final runs).
- **Consequences:** Tables show what RL achieves from scratch and what it adds on top of a demonstration.

## ADR-009: One Optuna Study per Stage and Learner
- **Status:** Accepted
- **Context:** The first tuning round tuned only Stage 3 MAPPO, copied its values to every other learner and stage, used one seed per trial, and scored the best checkpoint.
- **Decision:** Twelve studies (Stages 1–2 PPO; Stages 3–4 × MAPPO, HAPPO, HATRPO, HAA2C, MAT), 24 trials each, multivariate TPE with Hyperband pruning from 1M to 4M ticks. Full-budget trials train two seeds, and the objective is the mean final-policy return on 32 `val` episodes. The warm start is a searched parameter, and the best trial of each kind is exported. Storage is an Optuna journal file, because the workstation's application control blocks the SQLAlchemy extensions SQLite storage needs.
- **Alternatives Rejected:** Sharing one configuration across learners; tuning at the full final budget (several times the compute).
- **Consequences:** Scratch configurations for Stages 3–4 are chosen at 4M ticks, where scratch learners barely leave the random level; this is reported as a limitation.
