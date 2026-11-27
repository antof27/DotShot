# Roadmap and Status

## Current status (2026-11-27)

Branch `fix/marl-performance`. All experiments of the current round are finished and documented.

| Area | State |
| :--- | :--- |
| Simulator v2.2 | Done: respawn, terminal victory, projectile/status/crosshair observations, fire-time shaping, Stage 4 calibration, 8 ticks per decision, behaviour audit; renderer no longer touches the simulator's RNG |
| Learners | Done: SB3 PPO with Stage 1 → 2 transfer; HARL MAPPO, HAPPO, HATRPO, HAA2C and MAT with corrected adapters; behaviour-cloning warm start |
| Evaluation | Done: one evaluator, disjoint `val`/`test` seeds, baselines for all four stages, behaviour analysis (`scripts/analyze_behaviour.py`) |
| Hyperparameter search | Done: 12 studies (Stage 4 stopped at 16 trials), configs in `configs/official/` |
| Final runs | Done: 5 seeds per configuration; Stages 3–4 with cloned starts for all learners and the two best scratch learners per stage at 16M ticks |
| Documentation | Done: README, handbook chapters 1–4, report (`docs/paper/`), architecture and decisions |
| Replays | Done: `scripts/enjoy.py --run` replays and records any final run; `--snapshots` saves event screenshots |

## Possible next steps

1. Longer scratch runs on Stages 3–4 (24M ticks and beyond); the curves are still rising at 16M.
2. A curriculum that carries the Stage 2 policy into the cooperative stages.
3. Cloning from weaker experts, to measure how much of the cloned-start result depends on the demonstrations.
4. An observation or reward that makes undefended approaches to the Core visible to every agent (the main failure mode in the replays).

## Known limitations

- Scratch hyperparameters for Stages 3–4 are not tuned (one standard configuration for all learners).
- Only the two learners that led each scratch pilot were trained at 16M ticks.
- MAT needs every agent's observation at execution time; it is not a decentralized policy.
