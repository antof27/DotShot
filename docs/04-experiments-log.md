# 4. Experiments and Results

> [!NOTE]
> **What this chapter covers**
> 1. The hardware, the evaluation protocol and the scripted baselines.
> 2. The development experiments that turned DotShot into a learnable benchmark, and the decisions each one supported.
> 3. The hyperparameter search for every stage and learner.
> 4. The final 5-seed results.

Every number in this chapter comes from a run on simulator v2.2 with the code state stated next to it. Budgets are in simulator ticks; with the default decision interval of 8 ticks, 1M ticks is 125k decisions.

---

## 4.1 Hardware and Protocol

**Workstation.** NVIDIA RTX 3060 (12 GB), 8 physical cores, Windows 11, Python 3.11, PyTorch with CUDA. The pure-Python simulator bounds throughput, so the small MLP learners gain nothing from the GPU (50k-step benchmark, one process at a time, `docs/results/dev/device_benchmark.json`):

| Job | Device | Env steps/s | Peak VRAM (MiB) |
| :--- | :---: | ---: | ---: |
| HARL-MAPPO Stage 3 | CPU | 1,098 | 0 |
| HARL-MAPPO Stage 3 | CUDA | 754 | 33 |
| SB3 PPO Stage 1 | CPU | 1,844 | 0 |
| SB3 PPO Stage 1 | CUDA | 897 | 19 |

HARL and SB3 run on the CPU; MAT runs on CUDA. At most 6 training processes run at once (8 cores − 2), each GPU process is capped at 25% of the VRAM, and only the best and final checkpoints are kept.

**Evaluation.** One evaluator (`benchmarks/eval/marl_eval.py`) scores every learner and every baseline. Periodic evaluations play 16 episodes from the `val` seed set; after training, the final policy (never the best checkpoint) plays 32 episodes from the disjoint `test` set. Each configuration is trained with seeds 0–4. Across seeds we report the mean with a 95% percentile bootstrap CI (2,000 resamples) and the IQM. Returns are per-agent episode returns. Every evaluation also records the behaviour audit (`hold_frac`, `move_frac`, `shots_per_100`, `deaths`, `core_dist`) and every reward component.

---

## 4.2 Scripted Baselines

`configs/baselines/stage{3,4}_baselines.yaml`, 96 `test` episodes per policy, `action_repeat = 8`. Return: mean [95% bootstrap CI].

| Stage | Policy | Return [95% CI] | Win (%) | Kills / ep | Core HP (%) | Elem. acc. (%) | Move | Deaths / ep |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | random | −6.54 [−6.76, −6.30] | 0.0 | 0.1 | 0.0 | 33.2 | 0.89 | 1.62 |
| 3 | noop | −7.32 [−7.47, −7.16] | 0.0 | 0.0 | 0.0 | – | 0.00 | 1.83 |
| 3 | charge | −9.96 [−10.21, −9.72] | 0.0 | 0.0 | 0.0 | – | 0.92 | 3.02 |
| 3 | heuristic_nearest | 73.19 [69.20, 77.08] | 43.8 | 34.0 | 13.9 | 94.3 | 0.21 | 0.74 |
| 3 | heuristic_core_guard | 64.97 [60.91, 68.97] | 58.3 | 30.6 | 28.5 | 93.5 | 0.00 | 2.45 |
| 3 | **heuristic_calibrated** | **87.59** [83.97, 91.15] | **78.1** | 37.9 | 21.1 | 94.1 | 0.23 | 0.02 |
| 4 | random | −7.71 [−7.97, −7.48] | 0.0 | 0.2 | 0.0 | 34.3 | 0.89 | 2.17 |
| 4 | noop | −9.07 [−9.23, −8.92] | 0.0 | 0.0 | 0.0 | – | 0.00 | 2.73 |
| 4 | charge | −11.68 [−11.94, −11.41] | 0.0 | 0.0 | 0.0 | – | 0.85 | 4.43 |
| 4 | heuristic_nearest | 56.98 [53.66, 60.30] | 3.1 | 33.1 | 1.4 | 93.9 | 0.27 | 0.71 |
| 4 | heuristic_core_guard | 46.68 [42.74, 50.86] | 9.4 | 29.4 | 3.0 | 91.4 | 0.00 | 4.32 |
| 4 | **heuristic_calibrated** | **100.17** [96.89, 103.34] | **75.0** | 52.3 | 31.0 | 89.9 | 0.34 | 0.22 |

`charge` (walk into the nearest enemy, never fire) is the worst policy on both stages, as intended (§4.3). `heuristic_core_guard` stands at the Core and wins more Stage 3 episodes than `heuristic_nearest` but kills less. `heuristic_calibrated` is the reference for the difficulty calibration and the teacher for behaviour cloning.

Stage 1–2 baselines are listed with the final results (§4.6.1).

---

## 4.3 Withdrawn First Results and the Suicide Trap

The first Milestone 3 run reported Optuna-tuned results for all four stages, with every Stage 3–4 learner near random level. An audit found the integration defects listed in the simulator chapter and in `.agents/docs/DECISIONS.md` (ADR-005): among them, HARL clipped the fourth power of the PPO ratio, dead agents were never masked, Stage 4 "shared" MAPPO was unshared, and MAT acted through a continuous adapter. The tuning had covered only Stage 3 MAPPO, copied its values to everything else, used one seed per trial and scored the best checkpoint. Those results are withdrawn and not compared with anything here.

The environment had a deeper flaw. On v2.1, an episode on Stages 3–4 also ended when every agent was dead, which stopped the Core-damage penalties. Scripted baselines on 32 `test` episodes, original reward (`docs/results/dev/baselines_v21_stage{3,4}.json`):

| Stage | random | noop | charge | heuristic_nearest | heuristic_core_guard |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | −1.72 | −1.66 | **−1.05** | 23.07 | 15.93 |
| 4 | −2.42 | −2.30 | **−1.25** | 27.35 | 21.03 |

`charge` beat `random`, `noop` and every learner of the withdrawn tables (best −1.62 and −2.29). For a team that could not shoot yet, dying quickly was the best strategy.

---

## 4.4 How the Benchmark Became Learnable

These are development experiments: single runs or short multi-seed checks that justified each design decision. Configs are in `configs/ablations/`, raw records in `docs/results/dev/`.

### 4.4.1 Gate 1: fixed learners on the v2.1 rules (failed)

HARL-MAPPO with all learner fixes (true ratio, masked entropy, dead-agent masks, team reward with an EP critic, LR decay, shared actor), Stage 3 v2.1 rules, one decision per tick, 2M ticks, seeds 0–4, 32 `test` episodes per seed (`gate_v21.json`):

| Policy | Return mean [95% CI] | Kills / ep | Ep. length |
| :--- | :---: | :---: | :---: |
| HARL-MAPPO (5 seeds) | −2.37 [−2.74, −2.01] | 0.78 | 533 |
| noop | −4.31 [−4.70, −3.94] | 0.0 | 433 |
| charge | −3.04 [−3.31, −2.79] | 0.0 | 266 |

The learner beat `noop` and `charge` but missed the gate's minimum of 3 kills per episode. The reward breakdown showed why: aim shaping, then paid every tick for pointing at the current position of the nearest enemy, earned +1.28 per episode, 3.6 times the reward for correct hits (+0.35). Elemental accuracy stayed at chance and the policy entropy at 6.1–6.2 of a maximum of 7.05. The learner was paid for pointing, not for shooting well. This led to DotShot v2.2 (simulator chapter, §3.7–3.9).

### 4.4.2 Diagnostics on early v2.2 builds

Seed 0, 32 `test` episodes, one decision per tick. Build a: first v2.2 build with a symmetric ±0.6 element credit, and a HARL adapter that ended the episode whenever all agents were dead at once. Build b: both fixed, crosshair observation, +0.6 / −0.3 credit. Build c: +1.5 / −0.75 credit (the final rule).

| Build | Learner and change | Return | Kills / ep | Elem. acc. (%) | Win (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| a | HARL-MAPPO, gate config | −4.23 | 0.2 | 32 | 0 |
| a | HARL-MAPPO, `entropy_coef` 0.0025 | −3.80 | 0.5 | 41 | 0 |
| b | HARL-MAPPO, gate config | −3.35 | 0.5 | 28 | 0 |
| b | HARL-MAPPO, 16 actor / 8 critic minibatches, 10 epochs | −4.80 | 0.2 | 34 | 0 |
| b | SB3 PPO Stage 1 | 21.41 | 12.4 | 100 | 71.9 |
| b | SB3 PPO Stage 2, standard settings | 3.95 | 1.8 | 47.6 | 71.9 |
| c | SB3 PPO Stage 2, standard settings | 6.19 | 3.6 | 62.3 | 6.2 |

What the behaviour probes showed:
- **Build a.** The HARL policy held fire on 40% of its actions and hit with the wrong element three times as often as with the right one. Under the symmetric credit an uncertain shot had negative expected value, so not shooting was safer. The adapter bug also brought the suicide trap back inside HARL training.
- **Build b.** The Stage 2 PPO policy fired Water on 74% of its shots whatever the crosshair showed.
- **Build c.** Its choice now depended on the target (Grass against Water 62% of the time, Water against Fire 61%), though Fire against Grass was still rare (11%).

More gradient steps per sample did not help HARL-MAPPO, so the slow learning came from credit assignment for the element choice, not from the optimiser.

### 4.4.3 Gate 2: v2.2 at one decision per tick (failed)

The gate-1 learner on the final v2.2 rules, 2M ticks, seeds 0–4 (`gate_v22_k1.json`): return −1.64 [−1.98, −1.23], 1.84 kills per episode, elemental accuracy 39–57% per seed, 0% wins. The gate failed again on kills, but validation kills rose on every seed and flattened only as the decaying learning rate approached zero. The learner was improving slowly rather than failing.

### 4.4.4 Decision interval

A shot leaves only every 12–16 ticks, so at one decision per tick the aim and weapon branches mattered on about one tick in twelve. Holding each action for k ticks (ADR-007), seed 0 unless noted, 32 `test` episodes:

| Learner | k | Ticks | Win (%) | Kills / ep | Elem. acc. (%) | Return | Wall time |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | ---: |
| SB3 PPO Stage 1 | 1 | 1M | 71.9 | 12.4 | 100 | 21.41 | – |
| SB3 PPO Stage 1 | 8 | 2M | **100.0** | 14.8 | 100 | 31.93 | – |
| SB3 PPO Stage 2 | 1 | 1M | 6.2 | 3.6 | 62.3 | 6.19 | 892 s |
| SB3 PPO Stage 2 | 4 | 2M | 62.5 | 4.8 | 68.2 | 11.28 | 767 s |
| SB3 PPO Stage 2 | 8 | 2M | **96.9** | 4.7 | 61.6 | 11.56 | 586 s |
| HARL-MAPPO Stage 3 | 1 | 2M | 0.0 | 1.84 (5 seeds) | 39–57 | −1.64 | 2,222 s |
| HARL-MAPPO Stage 3 | 8 | 2M | 0.0 | 1.0 | 44.4 | −2.06 | 749 s |
| HARL-MAPPO Stage 3 | 8 | 6M | 0.0 | 2.2 / 2.3 (2 seeds) | 60.8 / 58.1 | +0.85 / +0.73 | ≈ 2,250 s |

At equal wall time HARL gets three times more experience at k = 8. Every run with the behaviour audit (k = 4 and k = 8) passed it, moving 78–99% of the time and holding fire 1–14%. The scratch MARL learners still lost the Core in every episode: they stayed about 400 px from it on average, while the calibrated script holds a 240 px ring.

### 4.4.5 What makes Stage 3 learnable

HARL-MAPPO on Stage 3, seed 0, k = 8, gate hyperparameters unless noted, 32 `test` episodes (or the latest 16-episode `val` evaluation where marked):

| Variant | Ticks | Return | Win (%) | Kills / ep | Elem. acc. (%) | Core HP (%) | Move | Notes |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Scratch MLP 128 | 6M | 0.85 | 0 | 2.2 | 60.8 | 0 | 0.86 | |
| **Scratch MLP 128, 24M-tick LR schedule** | 24M | **53.05** | **25.0** | 24.6 | 90.1 | 5.1 | 0.93 | `val` return 3.3 at 6M, 29.7 at 12M, 48.5 at 20M, 63.1 at 24M |
| Scratch GRU 128 | 6M | 5.23 | 0 | 4.3 | 61.5 | 0 | 0.93 | |
| Scratch MLP 128 + guard potential | 6M | 2.29 | 0 | 2.9 | 58.3 | 0 | 0.89 | potential-based shaping |
| Scratch, 16 minibatches × 10 epochs, lr 3e-4 | 3.7M (`val`) | −1.00 | 0 | 1.4 | 53.2 | 0 | 0.93 | stopped: worse |
| BC (40 episodes, 3 epochs) + RL | 2M | 1.77 | 0 | 3.0 | 68.5 | 0 | 0.46 | clone never moved at step 0 (`IDLE`) |
| **BC (200 episodes, 30 epochs), clone only** | 0 | 52.76 (`val`) | 37.5 | 24.5 | 82.3 | 14.1 | 0.32 | no RL |
| **BC + RL, 10 critic-only updates first** | 2M | **68.69** | **50.0** | 30.8 | 86.7 | 13.5 | 0.37 | |
| reference: `heuristic_calibrated` | – | 87.59 | 78.1 | 37.9 | 94.1 | 21.1 | 0.23 | 96 `test` episodes |

Findings:
- **The behaviour audit caught a broken warm start.** Cloned from too little data, the movement heads always chose "stay", so the policy never moved. With 200 episodes and 30 epochs, held-out branch accuracy reaches 83% (moves), 82% (aim) and 96% (weapon), and the clone plays by itself.
- **RL improves on the clone instead of erasing it** when the critic is warmed up first: 52.8 → 68.7 return and 37.5% → 50% wins in 2M ticks.
- **From scratch, learning is slow but does not plateau.** With a schedule sized for 24M ticks, the scratch learner learns to defend the Core (25% wins, 90% elemental accuracy) with a normal behaviour profile (moving 93%, holding fire 13%, 0.4 deaths per episode). Scratch runs need budgets of the order of 16–24M ticks; warm-started runs converge within a few million.

Stage 2 PPO at 1.2M ticks (`val`): from scratch the policy reaches 58.2% elemental accuracy and 2.8 kills; started from the Stage 1 policy with a reset weapon head it reaches **83.1%** and 9.2 kills.

These results fixed the final protocol: scratch and warm-started runs as separate rows, and scratch Stage 3–4 runs at much longer budgets than the tuning trials.

---

## 4.5 Hyperparameter Search

Protocol (`benchmarks/official/bohb_tune.py`, ADR-009): 12 studies (Stage 1 and 2 PPO; Stages 3 and 4 × MAPPO, HAPPO, HATRPO, HAA2C, MAT), 24 trials each, `TPESampler(multivariate=True, seed=42)` and `HyperbandPruner` between 1M and 4M ticks (reduction factor 3). A trial reports every 0.5M ticks on 16 `val` episodes from seed 100; trials that reach 4M ticks also train seed 101, and the objective is the mean final-policy return of both seeds on 32 `val` episodes. Searched: learning rates, entropy coefficient, clip range or HATRPO KL threshold, epochs, minibatches, γ, GAE λ, hidden size, MLP or GRU (scratch only), MAT depth, width and heads, and the warm start. Each Stage 4 study starts from the best Stage 3 configuration of the same learner. Logs: `runs/optuna/`; summary: `docs/results/optuna_summary.json`.

Two things to keep in mind when reading the search results:
- **Scratch trials on Stages 3–4 rank on a weak signal.** At 4M ticks a scratch learner has barely left the random level (§4.4.5), so its tuned values are only a starting point for the long final runs.
- **Stage 2 transfer trials are effectively single-seed.** `PPO.load` restored the seed stored in the Stage 1 checkpoint, so both seeds of a transfer trial produced the same run (trial 5 scored 44.67 on both). The runner now reseeds after loading (`tests/test_transfer_seed.py`); the fix landed while the Stage 2 study was running, and the final runs use it.

<!-- TUNING-RESULTS: best trial per study and warm start, added when all studies finish. -->

---

## 4.6 Final Results

All rows: 5 seeds (0–4), final policy, 32 `test` episodes per seed. Return: mean over seeds [95% bootstrap CI]; other columns: mean over seeds. Summaries: `docs/results/final_stage*.json`; curves: `docs/assets/figures/final_stage12_curves.png`.

### 4.6.1 Stages 1 and 2 (PPO, 4M ticks)

| Stage | Run | Return [95% CI] | IQM | Win (%) | Kills / ep | Right element (%) | Deaths / ep | Move | Hold |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | PPO from scratch | 43.15 [39.85, 45.69] | 43.90 | 100.0 | 20.4 | 100.0 | 0.00 | 0.95 | 0.09 |
| 2 | PPO from scratch | 22.46 [21.35, 23.29] | 22.78 | 93.1 | 8.6 | 77.8 | 0.07 | 0.86 | 0.05 |
| 2 | PPO from the Stage 1 policy | **44.17** [42.46, 46.34] | 43.50 | 78.1 | **18.7** | **91.6** | 0.22 | 0.98 | 0.06 |

Scripted baselines on the same stages (one run each, 96 `test` episodes; `configs/baselines/stage{1,2}_{random,heuristic}.yaml`): Stage 1 `random` −2.50 (0% wins), `heuristic` 46.04 (96.9% wins, 21.1 kills); Stage 2 `random` −2.25 (0% wins), `heuristic` 25.32 (0% wins, 11.2 kills, 98.0% right element, 873-tick episodes). The single-agent heuristic leads its target and picks the counter element but never dodges: on Stage 1, where enemies do not shoot, it slightly out-scores PPO; on Stage 2 it dies in every episode.


- **Stage 1 is solved.** Every seed wins all 32 test episodes without dying, with 20 kills per episode. Since Stage 1 has only Fire enemies, 100% elemental accuracy is expected once the agent fires Water.
- **Transfer doubles what the Stage 2 agent achieves**: twice the kills (18.7 vs 8.6), twice the return, and 91.6% instead of 77.8% right elements. The weapon head starts uniform after the reset, and the Stage 1 trunk already knows how to aim and move, so the agent only has to learn which element to load.
- **Transfer wins fewer episodes (78.1% vs 93.1%).** The transferred agent is more aggressive: it moves 98% of the time, fights more enemies at once and dies 0.22 times per episode instead of 0.07. Two of its five seeds win 56–66% of episodes, the other three 84–94%. The return rewards kills more than survival, so PPO trades some survival for kills. The scratch agent wins more often because it engages less.
- **No sign of reward hacking.** Aim shaping and the element credit together make up about a third of the return (Stage 2 transfer: 2.6 + 12.2 of 44.2). Both scale with useful shots: the credit is zero in expectation for a random weapon, and correct hits (+15.3) and kills (+9.4) grow in the same proportion. The agents fire 8.2 shots per 100 ticks, essentially the Striker's maximum of 8.3, and hold fire only 5–9% of the time.

### 4.6.2 Stage 3 (two Strikers)

Warm-started rows: actor cloned from `heuristic_calibrated`, 10 critic-only updates, then 4M ticks of RL with the tuned configuration. Scratch rows: standard hyperparameters (`configs/official/s3_*_scratch_std.yaml`), 16M ticks; only the two learners that led the pilot were scaled (below).

| Run | Return [95% CI] | IQM | Win (%) | Kills / ep | Right element (%) | Core HP (%) | Deaths / ep | Move |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HAA2C, cloned start | **84.87** [82.48, 87.26] | 84.99 | 63.1 | 37.5 | 91.4 | 21.4 | 1.1 | 0.76 |
| MAT, cloned start | 84.31 [82.02, 87.08] | 83.97 | 63.1 | 35.9 | 90.0 | 17.8 | 0.4 | 0.58 |
| MAPPO, cloned start | 81.96 [79.35, 84.99] | 81.54 | 61.3 | 36.0 | 90.1 | 19.5 | 0.4 | 0.38 |
| HATRPO, cloned start | 79.23 [76.78, 81.77] | 79.08 | 61.3 | 35.2 | 89.9 | 21.4 | 0.5 | 0.29 |
| HAPPO, cloned start | 79.01 [77.39, 80.54] | 79.06 | 56.9 | 34.4 | 90.6 | 17.9 | 0.3 | 0.29 |
| MAPPO, scratch (16M ticks) | 22.68 [16.91, 28.83] | 21.41 | 1.2 | 12.8 | 82.8 | 0.3 | 0.5 | 0.94 |
| MAT, scratch (16M ticks) | 20.11 [18.65, 21.78] | 20.07 | 0.0 | 11.6 | 82.6 | 0.0 | 0.3 | 0.95 |
| *heuristic_calibrated* (96 ep.) | 87.59 [83.97, 91.15] | 92.19 | 78.1 | 37.9 | 94.1 | 21.1 | 0.02 | 0.23 |

Scratch pilot (2 seeds, 8M ticks, mean final `val` return, the selection criterion): MAT 5.03, MAPPO 4.18, HAA2C 0.74, HAPPO −0.03, HATRPO −1.56 (`runs/pilot/stage3/selection.json`).

- **Cloned starts end close to the script but below it.** All five learners improve on the clone (validation return 55–62 at the start of RL, 80–87 after 4M ticks) and land within 9 points of the script's return. HAA2C and MAT are ahead of HAPPO with non-overlapping intervals; the other differences are within noise. Every learner wins fewer episodes than the script (57–63% against 78%), with the same Core HP left at the end, so the gap is in how often the Core survives, not in how well the agents fight.
- **HAA2C buys its return with risk.** It moves 76% of the time against 29–58% for the others and dies 1.1 times per episode against 0.3–0.5. Without PPO's clipping or a trust region, its policy drifted further from the cautious clone.
- **From scratch, shared parameters learn first.** In the pilot the two learners with one network for both agents (MAPPO, and MAT's shared transformer) were the only ones clearly above zero at 8M ticks; HAPPO, HATRPO and HAA2C train one actor per agent and see half the data each. At 16M ticks MAPPO and MAT learned to aim and to choose the element (83% right) but not to hold the Core: it falls in nearly every episode. The 24M-tick development run (§4.4.5) reached a test return of 53, so the scratch curve keeps rising well past this budget.
- **Shaping stays a minority of the return.** Aim shaping plus element credit contribute 17–18 of the 79–85 points of the cloned learners (about 20%), and the win bonus is small (0.4), because the Core rarely ends with much HP.


### 4.6.3 Stage 4 (Vanguard, Striker, Support)

Same protocol as Stage 3.

| Run | Return [95% CI] | IQM | Win (%) | Kills / ep | Right element (%) | Core HP (%) | Deaths / ep | Move |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MAPPO, cloned start | **107.90** [106.69, 109.11] | 108.04 | **88.1** | 56.1 | 90.1 | **49.1** | 0.1 | 0.70 |
| HAA2C, cloned start | 103.95 [102.75, 105.16] | 104.02 | 85.6 | 55.0 | 88.6 | 48.5 | 0.2 | 0.73 |
| MAT, cloned start | 100.93 [99.21, 102.93] | 100.70 | 70.0 | 52.9 | 87.5 | 30.1 | 0.0 | 0.34 |
| HATRPO, cloned start | 99.40 [97.07, 102.04] | 98.84 | 73.8 | 52.8 | 87.5 | 36.0 | 0.1 | 0.38 |
| HAPPO, cloned start | 91.63 [88.62, 95.52] | 90.92 | 63.7 | 49.2 | 85.2 | 30.9 | 0.2 | 0.42 |
| MAPPO, scratch (16M ticks) | 34.32 [31.38, 37.58] | 33.96 | 5.6 | 22.8 | 70.7 | 1.0 | 0.2 | 0.91 |
| MAT, scratch (16M ticks) | 32.23 [25.75, 38.37] | 33.01 | 6.2 | 22.2 | 71.4 | 2.0 | 0.2 | 0.87 |
| *heuristic_calibrated* (96 ep.) | 100.17 [96.89, 103.34] | 103.68 | 75.0 | 52.3 | 89.9 | 31.0 | 0.2 | 0.34 |

Scratch pilot (2 seeds, 8M ticks, mean final `val` return): MAPPO 4.01, MAT 2.92, HAPPO 0.84, HAA2C −1.18, HATRPO −1.68 (`runs/pilot/stage4/selection.json`).

- **Two learners beat the script.** MAPPO and HAA2C end above the scripted reference on every column that matters: return (intervals do not overlap), wins (88% and 86% against 75%) and Core HP left (49% against 31%). MAT and HATRPO match the script; HAPPO stays below it.
- **On the heterogeneous stage the shared actor wins.** HAPPO and HATRPO were designed for teams of different agents, yet MAPPO, with one actor for all three roles, finishes first both from the cloned start and in the scratch pilot, and HAPPO finishes last among the cloned starts. The class one-hot in the observation is enough for a shared network to behave differently per role, and sharing triples the data each actor parameter sees.
- **From scratch, Stage 4 learns further than Stage 3.** After 16M ticks MAPPO and MAT reach 32–34 return and win their first episodes (about 6%), against 20–23 and almost no wins on Stage 3. The Support's healing and the Vanguard's damage reduction keep a struggling team alive longer, which gives it more experience of the later waves.
- **The best policies changed how the team moves.** MAPPO and HAA2C move 70–73% of the time against 34% for the script and the other learners, while dying no more often.


---

## 4.7 Reproducing

```powershell
# Baselines
.\.venv\Scripts\python.exe -m benchmarks.runners.run_marl_baselines --config configs\baselines\stage3_baselines.yaml

# One study (journal storage under runs\optuna), then export the tuned configs
.\.venv\Scripts\python.exe -m benchmarks.official.bohb_tune --study s3_harl_mappo --device cpu
.\.venv\Scripts\python.exe -m benchmarks.official.bohb_tune --export-only

# Final runs, 5 seeds; Stage 2 transfer starts from the Stage 1 run with the same seed
.\.venv\Scripts\python.exe scripts\run_jobs.py --config configs\official\s1_sb3_ppo_scratch.yaml --seeds 0-4 --out runs\final\stage1 --workers 5
.\.venv\Scripts\python.exe scripts\run_jobs.py --config configs\official\s2_sb3_ppo_transfer.yaml --seeds 0-4 --out runs\final\stage2 --workers 5 --set "init_model_path=runs/final/stage1/s1_sb3_ppo_scratch/seed_{seed}/checkpoints/final.zip"

# Tables and curves
.\.venv\Scripts\python.exe scripts\report_final.py --stage 3 --runs runs\final\stage3 --baselines runs\baselines\stage3 --out docs\results\final_stage3.json --figure docs\assets\figures\final_stage3_curves.png
```
