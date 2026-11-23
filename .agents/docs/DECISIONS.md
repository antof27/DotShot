# Architectural Decision Records (ADRs)

This log records the major architectural decisions behind **DotShot Game v2** (`antof27/RL-game1`), why each choice was made, what alternatives were rejected, and what downstream consequences apply.

---

## ADR-001: Single Unified Repository (`antof27/RL-game1`)
- **Status:** Accepted
- **Context:** Prototype code lived in `RL/RL-game1` while literature notes and PDFs lived separately in `RL/RL-Literature`, making it difficult to keep simulator iterations synchronized with mathematical docs and empirical benchmarks.
- **Decision:** Evolve `antof27/RL-game1` as a single unified repository containing simulator (`dotshot/`), benchmark engine (`benchmarks/`), agent governance (`.agents/`), and the Docsify handbook (`docs/`).
- **Alternatives Rejected:** Maintaining two separate GitHub repositories (`dotshot-game` + `RL-Literature`) or keeping documentation only as a plain `README.md`.
- **Consequences:** Every pull request or milestone can update simulator code, benchmark tables, and handbook theory atomically in one branch.

## ADR-002: 100% English + Humanizer + Notelia Documentation Standard
- **Status:** Accepted
- **Context:** Academic RL/MARL terminology and portfolio visibility benefit from consistent English, while raw LLM drafts often suffer from cliché transitions and unverified citations.
- **Decision:** Write all code, comments, agent specs, and handbook chapters in 100% English under `.agents/skills/humanizer/SKILL.md` and `.agents/skills/notelia-docs/SKILL.md`, enforced by `scripts/validate_docs.py`. Contribute the portable English skill back to `Prv2106/Notelia` via its strict `dev` Gitflow.
- **Alternatives Rejected:** Bilingual IT/EN split docs or manual unvalidated prose writing.
- **Consequences:** Eliminates translation drift and catches fabricated citations or AI clichés automatically before commits.

## ADR-003: Windows Python 3.11 CUDA Virtual Environment on RTX 3060
- **Status:** Accepted
- **Context:** The existing `.venv` in `RL/RL-game1` was created on macOS (`bin/`) and cannot execute on Windows, while base Miniconda runs Python 3.13.12 where some RL wheel ecosystems lag behind.
- **Decision:** Create a clean project-local **Python 3.11** environment at `RL/RL-game1/.venv` via Miniconda3 (`conda create -p`) with CUDA-enabled PyTorch targeting the local NVIDIA GeForce RTX 3060 (12 GB VRAM).
- **Alternatives Rejected:** Using global Python 3.13 without isolation or training purely on CPU.
- **Consequences:** Reproducible Windows CUDA execution for both SB3 and custom PyTorch MARL training without polluting global Miniconda base.

## ADR-004: `1440×960` Tactical Arena with Cover, LOS, and 4-Stage Curriculum
- **Status:** Accepted
- **Context:** DotShot v1's empty `800×800` arena allowed agents to survive purely by kiting in circles with identical stats, hiding the value of centralized coordination and heterogeneous policies.
- **Decision:** Expand the arena to `1440×960` with rectangular cover obstacles, segment-AABB Line-of-Sight (LOS) occlusion, a Central Energy Core, Elemental-Shielded Bosses, and four stages ending in a heterogeneous `Vanguard / Striker / Support` squad.
- **Alternatives Rejected:** Keeping a larger empty open-field arena or building a `2400×2400` scrolling Fog-of-War world.
- **Consequences:** Preserves full situational readability at 60 FPS for humans and papers while creating genuine spatial occlusion, focus-fire, and role-specialization challenges.

## ADR-005: Dual Environment API — Gymnasium (Stages 1–2) + PettingZoo `ParallelEnv` (Stages 3–4)
- **Status:** Accepted
- **Context:** DotShot v1 concatenated multi-agent observations into a flat Gymnasium `Box` vector, breaking compatibility with standard MARL conventions and blurring local observations vs centralized state.
- **Decision:** Expose `DotShotSingleEnv(gym.Env)` for single-agent Stages 1–2 and `DotShotParallelEnv(pettingzoo.ParallelEnv)` with explicit `env.state()` for multi-agent Stages 3–4, keeping a unified `129`-dim per-agent local observation schema across all stages.
- **Alternatives Rejected:** Custom non-standard multi-agent wrapper classes or concatenating observations inside single-agent `gym.Env`.
- **Consequences:** Guarantees compliance with `check_env` and `parallel_api_test` while keeping Stage 2 -> Stage 3/4 policy transfer straightforward.

## ADR-006: Factored `MultiDiscrete([3, 3, 32, 4])` Heads Replacing 2,304-Way Action Flattening
- **Status:** Accepted
- **Context:** Legacy [marl/train_marl.py](../../marl/train_marl.py#L20-L39) combined `move_x`, `move_y`, `aim`, and `weapon` into a single `2304`-class categorical index, severely hurting exploration and sample efficiency.
- **Decision:** Replace flat categorical actions with factored multi-head categorical policies ($\log \pi_i = \sum_{k=1}^4 \log \pi_i^{(k)}$) for policy gradients/Transformers and Branching Dueling Q-heads for `VDN/QMIX`.
- **Alternatives Rejected:** Keeping the 2,304-way flat action space or switching to purely continuous joystick outputs.
- **Consequences:** Reduces output parameters by over an order of magnitude, preserves angular neighborhood structure, and maintains crisp human/bot equivalence.

## ADR-007: Modular CleanRL/HARL-Style PyTorch MARL Suite
- **Status:** Accepted
- **Context:** Heavy monolithic RL libraries obscure algorithm mechanics and slow down custom architectural modifications (such as Branching-QMIX or HAPPO sequential ratio accumulation on factored `MultiDiscrete` spaces).
- **Decision:** Keep Stable-Baselines3 (`PPO`, `A2C`) for standard single-agent baselines while building a transparent, modular PyTorch MARL suite in `benchmarks/marl/` covering `IPPO`, `Branching-VDN/QMIX`, `MAPPO`, `HAPPO/HATRPO`, and `MAT/PMAT/RoMAT`.
- **Alternatives Rejected:** Wrapping TorchRL/BenchMARL or RLlib as a black box.
- **Consequences:** Every line of advantage decomposition, hypernetwork mixing, and Transformer causal masking is inspectable, testable, and directly mappable to equations in `docs/02-sota-marl-taxonomy.md`.

## ADR-008: Portable Local Experiment Tracking + `rliable` Statistics
- **Status:** Accepted
- **Context:** Multi-device GitHub collaboration works smoothest when experiment configs, raw telemetry, and statistical scripts live in the repository without external SaaS login barriers.
- **Decision:** Drive all runs via YAML configs (`configs/`), log `config.yaml` + Git SHA + TensorBoard + `metrics.jsonl` locally, and aggregate multi-seed runs with Mean, Median, IQM, and 95% stratified bootstrap CIs (`benchmarks/eval/`).
- **Alternatives Rejected:** Mandatory Weights & Biases cloud logging or single-seed ad-hoc scripts.
- **Consequences:** Fully offline-capable, reproducible across workstations, and mathematically sound against single-seed outliers.

## ADR-009: Living Docsify Handbook with Automated BibTeX & Humanizer Gate
- **Status:** Accepted
- **Context:** Static markdown notes quickly fall behind code changes and accumulate broken references or AI-sounding boilerplate.
- **Decision:** Host the SOTA MARL handbook in `docs/` using Docsify (KaTeX + Mermaid, GitHub Pages ready), backed by `docs/references.bib` and `scripts/validate_docs.py`.
- **Alternatives Rejected:** Plain unlinked Markdown dumps or external Notion-only pages disconnected from the code repo.
- **Consequences:** Anyone cloning the repo can browse the interactive handbook locally or on GitHub Pages, with automated CI-style checks on citations and prose quality.

## ADR-010: Reward Shaping v2.1 — Smooth Cosine Aim Gradient and Elimination of Hold-Fire Farming
- **Status:** Accepted (Milestone 2 Calibration)
- **Context:** Milestone 1 baselines (`60k–80k` single-agent PPO, `40k` Stage 3 MARL) revealed two exploration bottlenecks in v2.0 shaping: (1) a steep binary aim threshold (`best_dot > 0.90`) offered zero gradient across 90% of the 32 aim bins; (2) an escalating miss penalty (`-0.22 * miss_mult`) paired with a hold-fire bonus (`+0.15 * 0.08`) encouraged policies to stop shooting instead of refining aim.
- **Decision:** Replace the binary aim cliff with smooth forward-hemisphere cosine shaping (`(best_dot ** 2) * 0.25` for `best_dot > 0.0`), remove passive hold-fire rewards, flatten badly-aimed miss penalties to `-0.04` (`0.0` for well-aimed misses), and raise positive hit/kill rewards (`hits_correct: 2.2`, `kills: 5.0`, `boss_kills: 10.0`, `shield_breaks: 4.5`, `combo_kills: 5.0`), while keeping $r_i^t = (1-\alpha)r_{i,\text{indiv}}^t + \alpha r_{\text{team}}^t$ intact.
- **Alternatives Rejected:** Auto-aim assist or shrinking the aim space below 32 bins.
- **Consequences:** Preserves full 32-bin aiming skill while providing smooth policy-gradient credit toward target alignment and aggressive elemental engagement.

## ADR-011: PyPA `src/` Layout, Unified `docs/` Artifacts, and `legacy_v1/` Archival
- **Status:** Accepted
- **Context:** Keeping active packages (`dotshot/`, `benchmarks/`), legacy v1 prototypes (`env/`, `marl/`, `models/`, root v1 scripts), and old `plots/` at the repository root cluttered top-level navigation and split plots across two folders.
- **Decision:** Move active v2 packages under `src/` (`src/dotshot/`, `src/benchmarks/`) via `pyproject.toml`, consolidate all reports, figures, screenshots, MP4 replays, curated `rliable` JSON summaries (`docs/results/`), and legacy plots (`docs/assets/legacy_v1_plots/`) under `docs/`, centralize CLI scripts including `scripts/enjoy.py`, and archive v1 prototypes in `legacy_v1/`.
- **Alternatives Rejected:** Deleting `legacy_v1/` or keeping flat root package folders.
- **Consequences:** Standard `src/` editable packaging prevents accidental root shadow imports, `docs/` serves as the single self-contained home for every handbook chapter, paper, plot, replay, and metric table, and v1 reference code stays inspectable without cluttering root.

## ADR-012: Pinned Official Upstream Submodules (PKU-MARL HARL & MAT), Optuna BOHB, and MiKTeX PDF Pipeline
- **Status:** Accepted (Milestone 3)
- **Context:** Milestones 1 and 2 used in-repo reimplementations of MAPPO, HAPPO, HATRPO, and MAT. Those versions kept every tensor shape and update rule inspectable, but they could drift from the published algorithms and invited reviewer questions about parity. Hyperparameters were also chosen by hand across four stages, and the 1M-step campaign needed a reproducible PDF artifact for the scientific report.
- **Decision:** Pin two upstream repositories as Git submodules: `third_party/HARL` at commit `099d29e` from `PKU-MARL/HARL`, and `third_party/Multi-Agent-Transformer` at commit `be3ff49` from `PKU-MARL/Multi-Agent-Transformer`. Keep HARL on the native `MultiDiscrete([3, 3, 32, 4])` space. Route MAT through the 8-D `Box` adapter in `src/benchmarks/official/mat_dotshot_env.py`, which converts continuous outputs to the four factored heads. Tune four studies with Optuna BOHB (`TPESampler(multivariate=True, seed=42)` plus `HyperbandPruner(min_resource=50_000, max_resource=350_000, reduction_factor=3)`) for 16 trials each, store results in SQLite databases under `runs/`, and export `docs/results/m3_bohb_summary.json`. Align `src/benchmarks/official/harl_runner.py` to per-agent mean reward before the 1M-step Stage 3 and Stage 4 runs. Compile `docs/paper/dotshot_marl_report.tex` with MiKTeX 25.12 and track the 25-page PDF in `docs/paper/`.
- **Alternatives Rejected:** Copying upstream source into `src/`, patching MAT to emit discrete actions directly, hand-tuned hyperparameters without a search record, cloud-only experiment tracking, and an Overleaf-only report with no local PDF.
- **Consequences:** The pinned commits make official algorithm comparisons reproducible and keep the adapters small. BOHB prunes 10 to 12 of 16 trials per study and leaves parameter importances on record. The HARL alignment puts both official families on one reward scale. The PDF link works offline and inside the Docsify handbook. Upstream updates require an explicit submodule bump and a fresh 1M-step sweep before any table changes.
