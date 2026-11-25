# Roadmap and Status

## Current status (2026-11-25)

Branch `fix/marl-performance`. The repository was cleaned up: DotShot v1, the in-house learners and every withdrawn result were removed. Hyperparameter search is running; the final 5-seed runs follow per stage as their studies finish.

| Area | State |
| :--- | :--- |
| Simulator v2.2 | Done: respawn, terminal victory, projectile/status/crosshair observations, fire-time shaping, Stage 4 calibration, 8 ticks per decision, behaviour audit |
| Learners | Done: SB3 PPO with Stage 1 → 2 transfer; HARL MAPPO, HAPPO, HATRPO, HAA2C and MAT with corrected adapters; behaviour-cloning warm start |
| Evaluation | Done: one evaluator, disjoint `val`/`test` seeds, Stage 3–4 baselines at k = 8 (96 `test` episodes) |
| Hyperparameter search | Stages 1–3 done; Stage 4 running |
| Final runs | Stage 1 done (5 seeds); Stage 2 running; Stages 3–4 pending |
| Documentation | Simulator spec, architecture, decisions, handbook chapters 1–3, report (method and development history) rewritten; results sections pending |
| Replays | `scripts/enjoy.py` needs a loader for the current HARL/MAT checkpoints before new videos can be recorded |

## Remaining work

1. Finish the Stage 4 studies and export every tuned configuration.
2. Final runs, 5 seeds each: Stage 2 (scratch, transfer), Stage 3 and Stage 4 (every learner, warm-started; scratch rows at long budgets).
3. Stage 1–2 scripted baselines (random, heuristic) at k = 8.
4. Tables and learning curves (`scripts/report_final.py`), results sections of the handbook, README and report, abstract.
5. Policy replay in `scripts/enjoy.py`, new screenshots and videos.
6. Pull request `fix/marl-performance` → `dev`.

## Known limitations

- Scratch hyperparameters for Stages 3–4 are selected at 4M ticks, where scratch learners barely leave the random level.
- Stage 2 transfer trials in the search are effectively single-seed (fixed before the final runs).
- MAT needs every agent's observation at execution time; it is not a decentralized policy.
