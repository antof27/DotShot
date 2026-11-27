# DotShot Game: Agent Governance and Workspace Rules

This repository hosts **DotShot Game**, a unified research workspace combining a tactical single-to-multi-agent Reinforcement Learning simulator (`src/dotshot/`), a reproducible benchmark suite (`src/benchmarks/`), and a living State-of-the-Art (SOTA) MARL handbook (`docs/`).

Every coding agent operating in this repository must follow these permanent rules.

---

## 1. Language and Prose Standard (100% English)

- Write all code, docstrings, comments, commit messages, configuration files, workspace skills, agent planning documents, and handbook chapters strictly in **English**.
- Apply the rules in [humanizer/SKILL.md](.agents/skills/humanizer/SKILL.md) to all written prose. Remove inflated claims, LLM clichés, forced rule-of-three phrasing, and repetitive sentence structure.
- Apply the pedagogical and citation standards in [notelia-docs/SKILL.md](.agents/skills/notelia-docs/SKILL.md) to all technical handbook material and experiment reports.

## 2. Source Integrity and Scientific Rigor

- Never fabricate, guess, or extrapolate bibliographic details (authors, publication years, venues, arXiv IDs, DOIs, or equation numbers). Every cited work must be verified against primary sources and recorded in `docs/references.bib`.
- Never report a benchmark number, metric, or test pass unless the corresponding script or test actually ran after the latest edit and you observed the exact output.
- State exact seeds, sample counts, hardware specs, and confidence intervals (Mean, Median, IQM, and 95% stratified bootstrap CIs) when comparing algorithms.

## 3. Lockstep Simulator and Documentation Rule

DotShot Game evolves incrementally. No feature or experiment is complete until both simulator code and documentation match:

1. Any change to arena geometry, line-of-sight rules, agent classes, observation vectors, global state, action spaces, or reward mixing must update `docs/03-dotshot-simulator-spec.md`, `.agents/docs/ARCHITECTURE.md`, and `.agents/docs/DECISIONS.md` (when architectural).
2. Any completed training sweep or ablation must log its resolved YAML config, Git SHA, hardware info, and metrics under `runs/`, update `docs/04-experiments-log.md`, and update `.agents/docs/ROADMAP.md`.

## 4. Reproducibility and Environment Policy

- Execute all Python commands using the project Windows CUDA virtual environment (`.\.venv\Scripts\python.exe`). Never rely on global system Python for project runs.
- Drive every training and evaluation run from a versioned YAML config under `configs/` with explicit deterministic seeds (`random`, `numpy`, `torch`, Gymnasium/PettingZoo).
- Keep `third_party/` (pinned HARL and MAT) unmodified. Patch upstream behaviour from `src/benchmarks/official/` and pin each patch with a test.

## 5. Git Workflow and Safety

- Never commit directly to `main`. Develop on `dev` or dedicated `feat/*` / `fix/*` branches.
- Push every verified phase branch and `dev` merge to `origin` (`git push`) so multi-device progress stays synchronized on GitHub.
- Never run `git reset --hard`, `git clean -fd`, or `git push --force` without explicit user confirmation.
- Keep binary model weights (`*.pt`, `*.zip`), Optuna storage (`*.db`, `*.journal`), and raw TensorBoard logs (`tb_log/`, `runs/`) out of Git; commit only curated summary metrics, generated plots (`docs/assets/figures/`), and selected gameplay clips (`docs/assets/videos/`).

## 6. Upstream Notelia Collaboration

- A separate local clone of the companion repository **Notelia** tracks the upstream **Notelia** skill and LaTeX/Typst publishing infrastructure.
- When contributing improvements back to Notelia, strictly follow Notelia's own `AGENTS.md` Gitflow rules: inspect status, pull `origin/dev`, work on a dedicated `feature/*` branch, pass 100% of automated white-box and black-box tests, merge `--no-ff` into local `dev`, and push only when explicitly requested.

---

## Navigation Index

### Agent Planning Docs (`.agents/docs/`)
- [ARCHITECTURE.md](.agents/docs/ARCHITECTURE.md): Package layout, tensor schemas, learner adapters, and the run artifact format.
- [DECISIONS.md](.agents/docs/DECISIONS.md): Architectural Decision Records that still hold.
- [ROADMAP.md](.agents/docs/ROADMAP.md): Status of the simulator, the benchmark suite and the documentation.

### Workspace Skills (`.agents/skills/`)
- [ponytail](.agents/skills/ponytail/SKILL.md) (`DietrichGebert/ponytail`, global + workspace; paired with `ponytail-review`, `ponytail-audit`, `ponytail-debt`, `ponytail-gain`, `ponytail-help`): Lazy senior dev YAGNI discipline, minimal complete diffs, code review, and bloat audits.
- [humanizer](.agents/skills/humanizer/SKILL.md) (`blader/humanizer` v4.0.0, global + workspace): Scientific and technical prose cleanup against AI writing patterns.
- [notelia-docs](.agents/skills/notelia-docs/SKILL.md): Pedagogical handbook structure, KaTeX/Mermaid formatting, and BibTeX quality gates.
- [citation-fact-checking](.agents/skills/citation-fact-checking/SKILL.md): Primary-source academic citation verification and zero-fabrication rules.
- [rl-gymnasium-sb3](.agents/skills/rl-gymnasium-sb3/SKILL.md): Gymnasium compliance, reward shaping guardrails, and Stable-Baselines3 workflows.
- [marl-benchmark-design](.agents/skills/marl-benchmark-design/SKILL.md): Dec-POMDP formulation, PettingZoo `ParallelEnv`, CTDE algorithms, and *rliable* statistics.
- [pytorch-cuda-engineering](.agents/skills/pytorch-cuda-engineering/SKILL.md): Windows CUDA (RTX 3060) training stability, thread isolation, and deterministic seeding.
- [experiment-reproducibility-optuna](.agents/skills/experiment-reproducibility-optuna/SKILL.md): YAML experiment schemas, dual JSONL/TensorBoard telemetry, and Optuna HPO.
- [scientific-plotting-rliable](.agents/skills/scientific-plotting-rliable/SKILL.md): IQM, 95% bootstrap CIs, colorblind-safe plots, and Markdown table exports.
- [pygame-simulator-craft](.agents/skills/pygame-simulator-craft/SKILL.md): Headless/renderer decoupling, 60 FPS neon tactical VFX, and MARL explainability overlays.
- [scientific-latex-report](.agents/skills/scientific-latex-report/SKILL.md): Publication-grade English `.tex` report authoring with `booktabs`, BibTeX, and structural checks.
- [game-visual-documentation](.agents/skills/game-visual-documentation/SKILL.md): Deterministic offscreen `1440×960` screenshot capture and qualitative visual analysis.
- [gitflow-github-sync](.agents/skills/gitflow-github-sync/SKILL.md): Feature-branch Gitflow, `.gitignore` hygiene, and continuous GitHub synchronization.
