# Task: Root-Cause Fixes for DotShot v2 MARL Performance (Stages 1–4)

You are working on the **DotShot Game v2** repository. **All v2 code lives on branch `dev`** (`origin/dev`, commit `b4f6a9a`). Local `main` still holds legacy v1 code — do not work there.

```bash
git fetch origin
git switch -c fix/marl-performance origin/dev
git submodule update --init --recursive   # third_party/HARL @099d29e, third_party/Multi-Agent-Transformer @be3ff49
```

Read `AGENTS.md` first and follow every rule (English only, YAML-driven experiments, lockstep docs, never report a metric you did not observe after your latest edit). The report under review is `docs/paper/dotshot_marl_report.{tex,pdf}` (Milestone 3).

Each finding below is tagged **[verified]** (reproduced by running code or by an exact code reading) or **[hypothesis]** (plausible, not proven — confirm before relying on it). Line numbers refer to `origin/dev` and the pinned submodules.

---

## 1. Evidence: Stage 3–4 learners are at random level, and a "suicide" policy beats them all

Direct rollouts of `DotShotEngine` (per-agent-mean return, same metric the benchmark reports; 6–20 episodes each):

| Stage | Policy | Return | Kills/ep | Core HP at end | Ep. length |
|---|---|---|---|---|---|
| 3 | random | −1.90 | 0.0 | 19% | 470 |
| 3 | no-op (stand still, never fire) | −1.79 … −1.90 | 0.0 | 21–32% | 440–466 |
| 3 | **charge: walk into the nearest enemy, never fire** | **−0.94** | 0.0 | 96% | 234 |
| 3 | scripted: shoot nearest, correct element, lead target | **+15.5** | 24.8 | 21% | 1473 |
| 3 | scripted: guard core, shoot enemy nearest the core | +14.9 | 18.5 | 70% | 1044 |
| 4 | random | −2.42 | 0.3 | 11% | 555 |
| 4 | charge (suicide) | −1.18 | 0.0 | 88% | 373 |
| 4 | scripted: shoot nearest | +24.4 | 40.3 | 29% | 1685 |

Milestone 3 (1M steps): Stage 3 best = OFFICIAL_MAT −1.62, HARL −2.06…−2.17; Stage 4 best = MAT −2.29, HARL_MAPPO −2.80. **Every Milestone 3 learner scores below a policy that walks into enemies to end the episode.** The rankings in the paper's Stage 3–4 tables are differences between near-random policies. No random, no-op, suicide, or heuristic baseline is reported for Stages 3–4, which hid this. [verified]

Stage 2: PPO elemental accuracy 36–40% in all runs ≈ the 33% of "always fire Water". The Stage 2 policy never learned element switching; its wins come from kiting. Stage 1's "100% elemental accuracy" is trivial (only Fire enemies exist). [verified]

---

## 2. Root causes

### 2.1 Algorithm and adapter bugs

**A. [verified, CRITICAL] HARL computes the PPO ratio as `r⁴` for MultiDiscrete actions.**
`ACTLayer.forward` sums the 4 branch log-probs into shape `(B, 1)` (`third_party/HARL/harl/models/base/act.py:70`; same in `evaluate_actions`, `:135`). The actor buffer allocates `action_log_probs` with shape `(T, E, 4)` because `get_shape_from_act_space(MultiDiscrete)` returns 4 (`on_policy_actor_buffer.py:56,64`), and `insert` broadcasts the `(E, 1)` sum into all 4 columns (`:96`). The update then computes `torch.prod(torch.exp(new_lp(B,1) − old_lp(B,4)), dim=-1)` (`mappo.py:64`, `happo.py:66`, `haa2c.py:65`, `hatrpo.py:77,149`) with `action_aggregation="prod"` (`src/benchmarks/official/harl_runner.py:370`) = `exp(4·Δlp)` = **r⁴**. I reproduced this with the real `ACTLayer`: `log(ratio_harl) / log(true_ratio) = 4.0` for every sample. Consequences: gradients are scaled ×4·r³, and the clip range on the true ratio shrinks from `[0.85, 1.15]` to about `[0.96, 1.035]`. All four HARL algorithms are affected. The HAPPO sequential `factor` (`on_policy_ha_runner.py:116`) is **not** affected (both terms come from `evaluate_actions`).

**B. [verified] HARL MultiDiscrete entropy builds a `(B, B)` matrix.**
`act.py:126-130` multiplies `entropy()` of shape `(B,)` by `active_masks` of shape `(B, 1)` → `(B, B)` per branch (memory ≈ B² floats × 4 branches; B = 6,400–12,800 here). Today, because no agent is ever masked (see C), the resulting scalar equals the sum over branches of the unmasked mean entropy, so the value is right and only memory/compute is wasted. **Once death masking is fixed (C), this code will include dead agents in the entropy term** — fix A, B and C together.

**C. [verified] Dead agents are never masked in HARL.**
`harl_dotshot_env.py:89` sets `dones` only when the whole episode ends. Agents that die mid-episode keep contributing all-zero-observation transitions to the policy and value losses (`active_masks` stays 1).

**D. [verified] HARL FP critic receives the same global state for every agent.**
`harl_dotshot_env.py:63-65` copies `env.state()` to all agents, and `harl_runner.py:309` uses `state_type="FP"` (one value per agent, trained on that agent's reward). With α = 0.65/0.70 the per-agent rewards differ, so part of each value target is unpredictable from the input. HAPPO/HATRPO's improvement guarantee also assumes one shared team reward.

**E. [verified] Optimisation budget is tiny and the LR never decays.**
`episode_length=200`, `n_rollout_threads=32` → 1M steps = 156 rollouts × `ppo_epoch=8` × `actor_num_mini_batch=1` ≈ **1,248 actor updates** (780 critic updates) at `lr=1.02e-4`, versus ≈ 78,000 for SB3 PPO on Stage 1. `use_linear_lr_decay` is off in HARL and MAT never calls `policy.lr_decay()`. Note: `num_mini_batch=1` and `ppo_epoch` 5 are upstream HARL/MAT defaults — they are not bugs in themselves; the problem is that combination together with `lr=1e-4` and a 1M budget. (That LR plateaus cause the late-training decline in the Stage 3–4 curves is a [hypothesis].)

**F. [verified] Stage 4 "shared MAPPO" is not shared.**
`bohb_tune.py:381` writes `share_param: algo == "mappo" and stage == 3`; `configs/official/stage4_harl_mappo.yaml` has `share_param: false`. All roles have identical obs/action spaces and the obs carries a role one-hot (`engine.py:714`), so sharing is valid. The paper's Stage 4 parameter-sharing conclusion refers to a run that never happened.

**G. [verified] MAT runs through a continuous Gaussian adapter on a discrete problem.**
`mat_dotshot_env.py:13-21` maps an 8-D Gaussian (std = `sigmoid(log_std)·0.5`, initially ≈ 0.37, never above 0.5; `transformer_act.py:53,77`) to movement thresholds, `atan2` aim and `argmax` weapon. Log-probs, entropy and exploration are those of the Gaussian, not of the real discrete choice. Also `mat_runner.py:77` sets `use_proper_time_limits=False` and the adapter (`mat_dotshot_env.py:71`) emits no `bad_transition`, so reaching `max_steps` (the win condition) is treated as terminal.

**H. [verified, design note — lower priority] MAT ignores `share_obs`.**
Upstream `ma_transformer.py:255-257,279-281,302-304` replaces `state` with zeros (`# state unused`); the critic attends over all agents' local observations instead. This is MAT's published design, not a DotShot bug. Enabling `encode_state=True` would require a patch (the zero state has a hard-coded width of 37). Treat as an optional ablation.

**I. [verified] MAT evaluation is biased.**
`mat_runner.py:148` stops at `max_eval_steps = max_steps` loop iterations over 4 threads, so it over-samples short (failed) episodes and can return as few as 4 episodes.

**J. [verified, minor] Smaller HARL runner issues.**
- `runner.warmup()` is never called (`harl_runner.py:472-477`): only step 0 of the first rollout uses all-zero obs. Negligible impact, trivial fix.
- HAA2C runs `a2c_epoch = ppo_epoch = 8` unclipped passes (`harl_runner.py:354`); upstream default is 5. Use the upstream value or tune it separately.
- HATRPO's KL uses HARL's `kl_approx` (unweighted `exp(q−p)−1−q+p` summed over the 42 concatenated log-probs). Upstream applies the same approximation to plain `Discrete` actions, so this is an upstream approximation rather than a DotShot bug; replacing it with the exact sum of per-branch categorical KLs is an optional improvement.

**K. [verified] BOHB tuning selected noise.**
Only HARL-MAPPO Stage 3 was tuned; its values were copied to HAPPO, HATRPO (which uses `kl_threshold`, not an actor LR), HAA2C and all Stage 4 configs. HARL trials ran with `max_steps` defaulting to 1500 (`harl_runner.py:292`) while final runs use 2000/2200. One seed per trial, pruning from 50k steps (~8 rollouts), and the objective is `best_return` **including the untrained step-0 evaluation** — MAT trials 0 and 1 both report exactly −0.982986, consistent with their step-0 score. All of it ran on the buggy pipeline (A–G).

**L. [verified] Evaluation statistics.**
`evaluate.py:compute_iqm` returns the plain mean when n < 4, so every 3-seed "IQM" is a mean and every "95% bootstrap CI" is resampled from 3 numbers. "Best Return" = max over evaluations on fixed eval seeds (selection bias). `evaluate.py:100` emits `-1e9` for `eval/synergy_combos.best` because HARL/MAT never log that key (25 occurrences in each of `docs/results/m3_stage{3,4}_official_eval.json`). 10 eval episodes per checkpoint is too few.

### 2.2 Environment, reward and task design

**M. [verified, CRITICAL] Early-termination ("suicide") reward trap.**
All agents dead → `terminated` (`engine.py:596-601`), which also stops all future core-damage penalties. For a policy that cannot kill yet, the per-step return is negative (core damage outweighs the +0.0015/+0.002 survival terms), so ending the episode early is better than staying. Measured: "charge into enemies, never fire" scores −0.94 (Stage 3) / −1.18 (Stage 4) vs no-op −1.79 / −2.43, and beats every Milestone 3 learner. Rough per-agent arithmetic (Stage 3, α = 0.65): dying costs ≈ −1.27; staying alive while the core dies costs ≈ −3.38 (Stage 4: −3.92). There is no win bonus for surviving `max_steps`. Whether specific Milestone 3 checkpoints actually converged to this behaviour is a [hypothesis] — the final checkpoints' low Core HP (1.4–11.8%) suggests most did *not*, so check episode lengths in the logs before claiming it.

**N. [verified] Agents cannot see enemy projectiles.**
`self.enemy_bullets` is absent from `get_local_obs` and `get_global_state` (`engine.py:692-846`), although Stages 2–4 require dodging. Vanguard `shield_active_timer` is also not observed.

**O. [verified] Aim shaping, weapon credit, fire throttle.**
- `engine.py:316-331` pays `0.25·dot²` every tick for pointing at the *current* (un-led) position of the nearest visible enemy, whether or not the agent fires — non-potential, farmable, ignores target leading.
- Correct-element reward only arrives on impact (20–40 ticks later), and the weapon head samples independently of the aim head; with `ent_coef` ≈ 0.003 and transfer from an "always Water" Stage 1 policy, the weapon head collapses.
- `engine.py:342-346` (`in_flight_same_target`) allows one bullet in flight per target: at 500 px that is ≈ 37 ticks between shots vs the 12-tick cooldown.
- Stages 3–4 train from random weights (no Stage 2 transfer) with a feed-forward policy under LOS partial observability.

**P. [verified] Difficulty is uncalibrated.** No scripted policy wins Stage 3–4 reliably: agents cannot see incoming fire, there is no respawn, and episodes end when all agents die (core-guard script keeps 70% core HP but dies at ~1040 ticks).

**Q. [verified] Paper text contradicts the code.**
- "A Stage 3 episode counts as a win only after the boss wave is cleared" — the code's win = reaching `max_steps` with core and ≥ 1 agent alive (`engine.py:601-604`).
- Stage 4 MAPPO described as shared (see F).
- "tunes each with Optuna BOHB" and "converged 1M-step benchmarks" (see K, §1).
- The M2 LOS ablation explains returns via "−0.04 pillar-impact misses", but bullets hitting obstacles are not penalised (`engine.py:385-388`).

---

## 3. Phased plan (execute strictly in order)

Do **not** launch long training runs until Phases 0–2 pass their gates. Keep `third_party/` unmodified; put wrappers, subclasses and patches in `src/benchmarks/official/`.

### Phase 0 — Baselines, diagnostics, sanity harness (no env semantics change)

1. Add `src/benchmarks/runners/run_marl_baselines.py` + YAML configs under `configs/` with `random`, `noop`, **`charge` (suicide)**, `heuristic_nearest`, `heuristic_core_guard` for Stages 3–4. Log the same schema as HARL/MAT: `eval/return_mean`, `eval/return_std`, `eval/win_rate`, `eval/core_hp_ratio`, `eval/elemental_accuracy`, `eval/kills_mean`, `eval/boss_kills_mean`, `eval/shield_breaks_mean`, `eval/synergy_combos`, `eval/episode_length`.
2. Expose per-component episode reward sums (team vs individual terms, core-damage penalty, death penalties, hit/kill rewards, aim shaping) and `episode_length` in `info`; log them in SB3, HARL, MAT and baseline runners. Fix `evaluate.py` so missing keys never produce `-1e9` and n < 4 is never labelled IQM.
3. Add a slow sanity test (`pytest -m slow`) for a short fixed HARL-MAPPO Stage 3 run (see gate in Phase 1).

### Phase 1 — Correctness and optimisation fixes (HARL, MAT, evaluation)

4. **Fix `r⁴` (A).** Simplest exact fix without touching `third_party/`: set `action_aggregation: "mean"` for MultiDiscrete in `harl_runner.py` (the 4 buffer columns are identical copies of the joint log-prob, so the mean of 4 identical ratios is the true joint ratio). Alternative: store per-branch log-probs `(B, 4)` in both buffer and `evaluate_actions` and keep `"prod"`. Make it configurable and add a unit test asserting `imp_weights == exp(new_joint_lp − old_joint_lp)` on `MultiDiscrete([3,3,32,4])` through the real buffer insert path, for MAPPO, HAPPO, HAA2C and HATRPO.
5. **Fix entropy (B) and death masking (C) together.**
   - Patch `ACTLayer.evaluate_actions` for MultiDiscrete (subclass or monkey-patch in `src/benchmarks/official/`): per branch `(entropy * active_masks.squeeze(-1)).sum() / active_masks.sum().clamp(min=1)`, then **sum over branches** (keeps the current `entropy_coef` semantics). Unit test: shape is scalar, no `(B, B)` tensor, masked rows excluded.
   - In `DotShotHARLEnv.step`, return `dones[i] = True` for agents that are dead while the episode continues; reset only when the episode ends. Verify against `OnPolicyHARunner.insert` / `OnPolicyMARunner.insert` that `active_masks[i] = 0` for dead agents and returns to 1 after reset. Unit test.
6. **Critic input and training reward (D).** Decide one design and document it (`docs/03-dotshot-simulator-spec.md`, ADR in `.agents/docs/DECISIONS.md`):
   - Default for all CTDE algorithms: **one shared team reward** `r = mean_i r_indiv,i + r_team` (individual combat terms stay in the signal, so aiming/kills are not drowned out) with `state_type: EP`. This satisfies HAPPO/HATRPO assumptions.
   - Per-agent mixed rewards (α) become an ablation only; when used, `state_type: FP` must get an **agent-specific state** `[global_state, agent_one_hot, local_obs_i]`.
   - Unit test: FP critic inputs differ across agents.
7. **Budget and LR schedule (E).** Expose `actor_num_mini_batch`, `critic_num_mini_batch` (HARL) and `num_mini_batch` (MAT) in YAML; defaults 4–8 (≈ 512–2048 samples per minibatch). Enable linear LR decay in HARL and call `policy.lr_decay(episode, updates)` in `mat_runner.py` when enabled. Start from `lr` 3e-4–5e-4, `entropy_coef` ≈ 0.01.
8. **Minor HARL fixes (J).** Call `runner.warmup()` before the loop; HAA2C `a2c_epoch` from YAML, default 5 (upstream); optional exact per-branch KL for HATRPO with a test against `torch.distributions.kl_divergence`.
9. **Stage 4 sharing (F).** `harl_mappo` = shared parameters in both stages (`bohb_tune.py:381`, `configs/official/stage4_harl_mappo.yaml`). If an unshared variant is wanted, add it as `harl_mappo_unshared`. Unit test: `actor[0] is actor[1] is actor[2]`.
10. **MAT (G, I).** Replace the continuous adapter with factored categorical heads `(3, 3, 32, 4)` in a thin subclass in `src/benchmarks/official/` (joint log-prob = sum over branches, entropy = sum over branches, autoregressive over agents as upstream). Enable `use_proper_time_limits` and pass `bad_masks` from `bad_transition`. Fix `_evaluate()` to run until exactly `n_eval_episodes` complete episodes. `encode_state` (H) only as an optional ablation.
11. **Configs and evaluation (K, L).** Pass `max_steps` explicitly from YAML everywhere. ≥ 32 eval episodes per checkpoint on seeds disjoint from training and from checkpoint selection; ≥ 5 training seeds in reported tables. Final metrics are primary; if Best is reported, select on a validation seed set and report held-out scores.

**Phase 1 gate:** fixed HARL-MAPPO on Stage 3 at 1M–2M steps (5 seeds) must beat **both `noop` and `charge`** beyond the 95% CI **and** reach ≥ 3 kills/episode with episode length above the `charge` baseline. Beating `noop` alone is not enough: until Phase 2 removes the trap, a learner can beat `noop` simply by dying faster.

### Phase 2 — Environment v2.2: observability, reward trap, calibration

Bump the simulator to **v2.2**; document the breaking obs/reward changes in `DECISIONS.md` and `docs/03-dotshot-simulator-spec.md`.

12. **Remove the suicide trap (M).** Recommended for Stages 3–4: respawn dead agents after 120–180 ticks with a death penalty; terminate only on `core_destroyed` or `max_steps`. If you keep all-dead termination, charge the remaining expected core loss as a terminal penalty. Add a win bonus on surviving `max_steps` (e.g. +10 raw × remaining Core HP ratio). Unit test on scripted rollouts: `charge` return < `noop`-but-fighting return < winning return.
13. **Observations (N).** Add the top-K (K = 4) nearest threatening enemy projectiles (relative position, velocity, time-to-closest-approach, active flag) to `get_local_obs`, active enemy projectiles to `get_global_state`, plus `shield_active_timer / 90` and a core-attacker direction signal. Update `OBS_DIM`, `state_dim`, tests and spec.
14. **Shaping and fire mechanics (O).**
    - Aim shaping toward the **lead point** of the prioritised threat (prefer core attackers when defending), paid only on firing ticks or in potential-based form `Φ(s') − γΦ(s)`, with a YAML annealing option.
    - Fire-time elemental credit: when a shot is fired and `_find_ray_target_uid` returns a target, add `+c` for the counter element and `−c` otherwise (c ≈ 0.05–0.08 raw).
    - Relax `in_flight_same_target` to 2–3 bullets per target (or remove it).
    - For Stage 1 → 2 transfer, reset or re-raise weapon-head entropy (or train Stage 1 with a mild multi-element curriculum).
15. **Calibrate difficulty (P).** Upgrade the scripted policy to dodge (using the new projectile obs), lead targets, pick the counter element, guard the core, and use Stage 4 abilities. Tune `spawn_cooldown`, `enemy_speed_mult`, enemy projectile damage and core contact damage so that the script wins ≥ 70% (Stage 3) / ≥ 60% (Stage 4) while `noop`, `random` and `charge` win 0%.
16. Re-run Stages 1–2 on v2.2: Stage 1 PPO ≥ 80% win rate, Stage 2 PPO elemental accuracy ≥ 80%.

### Phase 3 — Recurrence, warm start, clean tuning

17. Test `use_recurrent_policy: true` (GRU, `data_chunk_length` 10–20) vs MLP under `partial_obs: true`. Run 2M steps to confirm curves clearly above all baselines, then 3–5M+ for final numbers.
18. Warm-start Stage 3–4 actors via (a) short behaviour cloning from the calibrated script (works for HARL and MAT), or (b) verified weight transfer from a v2.2 Stage 2 policy (unit test: identical action distributions on the same obs; mind HARL's feature-normalisation LayerNorm and per-branch heads). Report scratch and warm-start as separate rows.
19. Re-run BOHB **only after Phases 1–2 pass**: per algorithm family (SB3 PPO, HARL-MAPPO, HARL-HAPPO, HARL-HATRPO tuning `kl_threshold`, MAT); ≥ 2 seeds per trial; same `max_steps` as final runs; max resource ≥ 1M steps; objective = mean **final** return on held-out eval seeds (never `best_return`, never including step 0).

### Phase 4 — Final benchmarks and paper

20. Run the 5-seed suite for Stages 1–4 with `random`, `noop`, `charge` and heuristic rows in every table.
21. Update `docs/paper/dotshot_marl_report.tex` (recompile the PDF), `docs/04-experiments-log.md`, `docs/03-dotshot-simulator-spec.md`, `.agents/docs/DECISIONS.md`: regenerate every table and figure from `docs/results/*.json`, correct the contradictions in Q, and archive the old Milestone 3 numbers as superseded, citing bugs A, C, D, F, G, K, M.

---

## 4. Acceptance criteria

- `pytest` passes, with new unit tests for: `r¹` ratio through the real buffer path (A); masked MultiDiscrete entropy without `(B, B)` (B); per-agent death masking (C); agent-specific FP critic input (D); Stage 4 MAPPO parameter sharing (F); discrete MAT heads, `bad_masks` on truncation and complete-episode evaluation (G, I); v2.2 projectile observations, fire-time elemental shaping and the reward ordering `charge < fight-and-lose < win` (M–O).
- Stage 3–4 baseline tables (`random`, `noop`, `charge`, heuristics) with ≥ 32 episodes each; calibrated heuristic wins ≥ 70% (Stage 3) / ≥ 60% (Stage 4).
- Phase 1 gate met (see above).
- After Phases 1–3 on v2.2: Stage 2 PPO elemental accuracy ≥ 80%; at least one Stage 3 algorithm reaches ≥ 50% of the calibrated heuristic's return and > 20% win rate with non-trivial kills and core HP; Stage 4 return clearly above all baselines.
- If a target is missed, stop and report curves, reward-component breakdowns and diagnostics instead of tuning blindly.

## 5. Working rules

- Work on `fix/marl-performance` from `origin/dev`; small verified commits per phase; follow `AGENTS.md` for pushing and docs lockstep.
- Do not modify `legacy_v1/`; do not edit `third_party/` in place.
- Never report a number that did not come from a run executed after your latest code change. Claims tagged [hypothesis] above must be confirmed or dropped, not repeated.
