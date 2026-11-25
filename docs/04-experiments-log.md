# 4. Living Experiments and Benchmark Log

This chapter records every empirical benchmark run on **DotShot Game v2** in chronological milestone order. All numbers come directly from logged `metrics.jsonl` trajectories and automated `src/benchmarks/eval/evaluate.py` summaries (with Milestone 2 and Milestone 3 JSON exports archived under [docs/results/](results/)).

---

## 4.1 Reproducibility and Statistical Protocol

- **Workstation Hardware**: Windows 11/10, **NVIDIA GeForce RTX 3060 (12 GB VRAM)**, PyTorch `2.6.0+cu124`, Python `3.11.18`.
- **Configuration & Seeding**: Every run executes from a versioned YAML file in `configs/` with deterministic seeding (`seed_everything`) across `random`, `numpy`, `torch`, and environment instances.
- **Statistical Reporting Standard**: Following **Agarwal et al. (2021)** [@agarwal2021rliable] and **Gorsane et al. (2022)** [@gorsane2022marleval], we report **Interquartile Mean (IQM)** with **95% bootstrap confidence intervals** ($B = 2{,}000$ resamples) alongside mean win rate, Core survival ratio, elemental accuracy, and kills per episode.

---

## 4.2 Milestone 1: Initial Single-Agent & Cooperative Smoke Benchmarks

Milestone 1 validates the end-to-end training, `rliable` evaluation, and 300 DPI plotting pipelines across single-agent Stages 1–2 and a 40,000-step cooperative smoke benchmark on Stage 3. Every summary table below reports the exact output emitted by `src/benchmarks/eval/evaluate.py` (`2,000` bootstrap resamples).

### 4.2.1 Stage 1 Single-Agent Fundamentals (60,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 21.63 [20.19, 22.57] | 21.63 [20.19, 22.57] | 97.9% | 100.0% | 100.0% | 21.1 |
| `PPO` | 3 | -4.47 [-4.95, -4.00] | -3.46 [-4.00, -3.15] | 0.0% | 100.0% | 34.7% | 0.1 |
| `RANDOM` | 3 | -3.04 [-3.08, -2.99] | -3.04 [-3.08, -2.99] | 0.0% | 100.0% | 32.6% | 0.0 |

![Stage 1 Learning Curves](assets/figures/stage1_learning_curves.png)

**Quantitative & Qualitative Analysis**:
- **Scripted Anchor**: `HEURISTIC` reaches `21.63 [20.19, 22.57]` episodic return, `97.9%` win rate, `100.0%` elemental accuracy, and `21.1` kills per episode. Because Stage 1 spawns only `Fire` enemies and four symmetric pillars, linear velocity target leading paired with fixed `Water` shots nearly solves the stage.
- **Short-Budget PPO Underfitting**: Over `60,000` timesteps (`3` seeds), SB3 `PPO` (`@schulman2017ppo`) ends at `-4.47 [-4.95, -4.00]` (`0.0%` wins, `0.1` kills/episode), trailing uniform `RANDOM` (`-3.04 [-3.08, -2.99]`). Elemental accuracy climbs from `0.0%` at step `0` to `34.7%` (near the `33.3%` uniform 3-weapon baseline) as the policy begins triggering `Water` shots, yet `60k` steps is too short to coordinate the `32`-bin aim head with kiting. Repeated misses and contact damage pull PPO slightly below random movement.

---

### 4.2.2 Stage 2 Elemental Mastery (80,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 9.70 [9.19, 10.58] | 9.70 [9.19, 10.58] | 0.0% | 100.0% | 99.7% | 10.9 |
| `PPO` | 3 | -3.48 [-3.89, -3.25] | -2.98 [-3.30, -2.73] | 0.0% | 100.0% | 17.8% | 0.0 |
| `RANDOM` | 3 | -2.72 [-2.83, -2.64] | -2.72 [-2.83, -2.64] | 0.0% | 100.0% | 31.6% | 0.0 |

![Stage 2 Learning Curves](assets/figures/stage2_learning_curves.png)

**Quantitative & Qualitative Analysis**:
- **Difficulty Jump on the Heuristic**: Introducing all three element types, incoming enemy projectiles, and neutral allies cuts `HEURISTIC` return by more than half, from `21.63` to `9.70 [9.19, 10.58]` (`0.0%` win rate, `10.9` kills/episode, `99.7%` elemental accuracy). Because the scripted baseline only kites radially from the nearest enemy without dodging incoming bullets or routing around cover, mixed enemy fire wears down its `100` HP before wave completion.
- **PPO vs. Three-Way Switching**: At `80,000` steps, `PPO` reaches `-3.48 [-3.89, -3.25]` (`-2.98` best) with `17.8%` elemental accuracy and `0.0` kills/episode, below `RANDOM` (`-2.72`, `31.6%` elemental accuracy). Learning simultaneous target leading, three-element switching, projectile evasion, and neutral-ally restraint requires longer horizon budgets (`300k–1M` steps) or curriculum transfer from a converged Stage 1 policy.

---

### 4.2.3 Stage 3 Cooperative Core Defense Smoke Benchmark (40,000 steps on RTX 3060 CUDA, 2 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MAT` | 2 | -2.16 [-2.26, -2.06] | -2.06 [-2.26, -1.86] | 0.0% | 21.2% | 31.7% | 0.1 |
| `QMIX` | 2 | -2.19 [-2.33, -2.04] | -2.02 [-2.04, -1.99] | 0.0% | 22.7% | 52.1% | 0.1 |
| `MAPPO` | 2 | -2.30 [-2.40, -2.21] | -2.20 [-2.21, -2.20] | 0.0% | 29.3% | 32.2% | 0.1 |
| `IPPO` | 2 | -2.52 [-2.53, -2.50] | -2.36 [-2.53, -2.20] | 0.0% | 7.0% | 30.1% | 0.1 |
| `HAPPO` | 2 | -2.78 [-3.12, -2.44] | -2.20 [-2.29, -2.12] | 0.0% | 2.2% | 16.8% | 0.1 |

![Stage 3 MARL Comparison](assets/figures/stage3_marl_smoke_comparison.png)

> [!TIP]
> **Recorded Evaluation Replay**: Watch the Stage 3 policy rollout captured via `scripts/enjoy.py --record`: [Stage 3 MARL Gameplay Replay (MP4)](assets/videos/dotshot_stage3_demo.mp4).

**Quantitative & Qualitative Comparison (40k-Step Smoke Regime)**:

Because `40,000` environment steps on `2` seeds represent a short verification run rather than converged training, all five MARL algorithms stay at `0.0%` win rate and `0.1` kills/episode. Even in this early window, clear behavioral differences separate value factorization, shared-parameter actor-critics, sequential transformers, and independent or heterogeneous trust-region updates:

1. **`MAT` (`-2.16` Final Return, `21.2%` Core HP, `31.7%` Elem Acc)**: Multi-Agent Transformer (`@wen2022mat`) achieves the highest final return IQM (`-2.16 [-2.26, -2.06]`) and rebounds steadily after `20k` steps. Cross-agent attention over both Strikers' `129`-dim observations reduces wasted overlap near the Core during early exploration.
2. **`Branching-QMIX` (`-2.19` Final Return, `-2.02` Best Return, `52.1%` Elem Acc)**: Combining QMIX monotonic mixing (`@rashid2018qmix`) with Branching Dueling Q-heads (`@tavakoli2018branching`) produces the best peak return (`-2.02 [-2.04, -1.99]`) and the clearest learning signal on **elemental accuracy (`52.1%`)**, climbing sharply between `20k` and `40k` steps. Off-policy TD backups from the replay buffer directly reinforce the 4-choice weapon branch whenever counter-element hits occur, avoiding the high-variance policy-gradient credit noise seen in on-policy runs.
3. **`MAPPO` (`-2.30` Final Return, `29.3%` Core HP, `32.2%` Elem Acc)**: Shared-parameter MAPPO (`@yu2022mappo`) achieves the highest **Core survival (`29.3%`)** at `40k` steps, rising steeply from `7%` at step `20k`. Conditioning the centralized critic on `env.state()` stabilizes early Core-defense positioning because both Stage 3 agents are homogeneous `Striker` units that benefit from doubled gradient throughput via parameter sharing.
4. **`IPPO` (`-2.52` Final Return, `7.0%` Core HP, `30.1%` Elem Acc)**: Independent PPO (`@dewitt2020ippo`) flattens near `-2.52` while Core survival drops steadily from `24%` to `7.0%`. Without centralized state access or cross-agent conditioning, each independent critic tracks a moving target under LOS partial observability.
5. **`HAPPO` (`-2.78` Final Return, `2.2%` Core HP, `16.8%` Elem Acc)**: Sequential heterogeneous updates (`@kuba2022happo`) perform worst in the `40k`-step smoke window (`-2.78 [-3.12, -2.44]`). Unlike shared `MAPPO` or `MAT`, `HAPPO` maintains separate actor parameters per agent without weight sharing, halving per-network sample reuse on a 2-agent homogeneous team. Milestone 2 (`200k+` steps across `5` seeds on Stages 3 and 4) will test whether `HAPPO` catches up on Stage 3 and pulls ahead of shared `MAPPO` when Stage 4 introduces asymmetric `Vanguard`, `Striker`, and `Support` roles.

---

## 4.3 Milestone 2: Converged Single-Agent, 5-Seed MARL Sweeps, and Controlled Ablations

Milestone 2 applies **Reward Shaping v2.1** (smooth `(best_dot ** 2) * 0.25` aim gradient, removal of hold-fire farming, flat `-0.04` miss penalty, and boosted hit/kill terms) together with `DummyMARLEnv` (`vec_type="dummy"`, `2,349 env_steps/sec` on Windows) and pre-cached LOS geometry (`24,388 steps/sec` single-env). This section records converged 200k-step single-agent runs, 5-seed Stage 3 and Stage 4 MARL sweeps on RTX 3060 CUDA, and controlled 3-seed ablations over $\alpha$ and Line-of-Sight (LOS) masking.

### 4.3.1 Stage 1 & Stage 2 Converged Single-Agent (v2.1, 200,000 steps, 3 seeds)

#### Stage 1 Single-Agent Fundamentals (v2.1, 200,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 37.68 [35.34, 39.21] | 37.68 [35.34, 39.21] | 97.9% | 100.0% | 100.0% | 21.1 |
| `PPO` | 3 | 3.31 [1.72, 6.41] | 3.55 [1.81, 6.41] | 38.9% | 100.0% | 89.5% | 2.3 |
| `A2C` | 3 | -1.34 [-1.88, -1.00] | -1.21 | 0.0% | 100.0% | 74.9% | 0.4 |
| `RANDOM` | 3 | -2.29 [-2.36, -2.18] | -2.29 [-2.36, -2.18] | 0.0% | 100.0% | 28.6% | 0.0 |

![Stage 1 Converged Learning Curves](assets/figures/m2_stage1_converged.png)

#### Stage 2 Elemental Mastery (v2.1, 200,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 18.84 [18.07, 20.32] | 18.84 [18.07, 20.32] | 0.0% | 100.0% | 99.7% | 10.9 |
| `PPO` | 3 | -1.16 [-1.59, -0.92] | -1.03 [-1.19, -0.92] | 0.0% | 100.0% | 31.5% | 0.2 |
| `A2C` | 3 | -1.58 [-1.97, -1.07] | -1.26 | 0.0% | 100.0% | 22.0% | 0.2 |
| `RANDOM` | 3 | -1.99 [-2.02, -1.93] | -1.99 [-2.02, -1.93] | 0.0% | 100.0% | 32.5% | 0.0 |

![Stage 2 Converged Learning Curves](assets/figures/m2_stage2_converged.png)

**Quantitative & Qualitative Analysis**:
- **Stage 1 Breakthrough Under v2.1**: Replacing the binary `>0.90` aim cliff with smooth `(best_dot ** 2) * 0.25` shaping lets `PPO` [@schulman2017ppo] move past the Milestone 1 underfitting regime (`-4.47` return, `0.0%` wins) to `3.31 [1.72, 6.41]` final IQM (`3.55 [1.81, 6.41]` best), `38.9%` win rate, `89.5%` elemental accuracy, and `2.3` kills per episode. `A2C` also learns element selection (`74.9%` accuracy) and improves over `RANDOM` (`-1.34 [-1.88, -1.00]` vs `-2.29 [-2.36, -2.18]`), though synchronous actor-critic updates trail PPO's clipped surrogate (`0.0%` win rate, `0.4` kills).
- **Why Stage 2 Remains Hard at 200k Steps**: Even with smooth aim gradients, Stage 2 `PPO` plateaus at `-1.16 [-1.59, -0.92]` (`31.5%` elemental accuracy, `0.2` kills) and `A2C` reaches `-1.58 [-1.97, -1.07]` (`22.0%` accuracy, `0.2` kills), ahead of `RANDOM` (`-1.99`) yet far behind `HEURISTIC` (`18.84 [18.07, 20.32]`). Conditioning the `4`-choice weapon head on all three enemy types while simultaneously dodging incoming fire and sparing neutral allies causes multi-head gradient conflict when trained from scratch; this directly motivates Stage 1 $\to$ Stage 2 weight transfer in Milestone 3.

---

### 4.3.2 Stage 3 Homogeneous Co-op (5 seeds, 60,000 env steps, CUDA)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `QMIX` | 5 | -1.72 [-1.97, -1.42] | -1.64 [-1.78, -1.37] | 0.0% | 32.0% | 27.4% | 0.2 |
| `HAPPO` | 5 | -1.75 [-2.37, -1.45] | -1.65 [-1.87, -1.45] | 0.0% | 10.1% | 39.7% | 0.1 |
| `MAPPO` | 5 | -1.84 [-2.45, -1.56] | -1.61 [-1.69, -1.47] | 0.0% | 16.2% | 42.2% | 0.2 |
| `MAT` | 5 | -1.99 [-2.29, -1.80] | -1.87 [-2.05, -1.77] | 0.0% | 17.0% | 40.9% | 0.1 |
| `IPPO` | 5 | -2.02 [-2.23, -1.59] | -1.67 [-1.96, -1.45] | 0.0% | 10.6% | 39.6% | 0.2 |

![Stage 3 5-Seed MARL Comparison](assets/figures/m2_stage3_5seed_marl.png)

**Quantitative & Qualitative Analysis**:
- **Value Factorization Leads Core Defense**: Across `5` seeds at `60,000` environment steps, `Branching-QMIX` [@rashid2018qmix; @tavakoli2018branching] ranks first on final return (`-1.72 [-1.97, -1.42]`) and doubles every on-policy method on **Core survival (`32.0%` vs `10.1%–17.0%`)**. Off-policy replay reuses rare wave-survival and boss-defense transitions more efficiently than `60k`-step on-policy rollouts, keeping the two `Striker` units anchored near the Central Core.
- **Actor-Critic Peak vs Variance**: `MAPPO` [@yu2022mappo] achieves the highest peak return IQM (`-1.61 [-1.69, -1.47]`) and top elemental accuracy (`42.2%`), while `HAPPO` [@kuba2022happo] closes its Milestone 1 deficit to reach `-1.75 [-2.37, -1.45]` (`39.7%` elemental accuracy). `MAT` [@wen2022mat] stabilizes with a tight CI (`-1.99 [-2.29, -1.80]`, `40.9%` elemental accuracy), whereas `IPPO` [@dewitt2020ippo] stays last (`-2.02 [-2.23, -1.59]`, `10.6%` Core HP) because decentralized critics cannot disambiguate teammate failures under LOS cover.

---

### 4.3.3 Stage 4 Heterogeneous Squad (Vanguard, Striker, Support; 5 seeds, 60,000 env steps, CUDA)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ROMAT` | 5 | -2.27 [-2.47, -2.16] | -2.18 [-2.29, -2.07] | 0.0% | 10.8% | 31.1% | 0.1 |
| `MAPPO` | 5 | -2.39 [-2.92, -1.95] | -2.03 [-2.43, -1.88] | 0.0% | 9.4% | 31.3% | 0.2 |
| `HAPPO` | 5 | -2.47 [-2.85, -2.17] | -2.19 [-2.51, -1.85] | 0.0% | 6.5% | 32.9% | 0.1 |
| `IPPO` | 5 | -2.48 [-2.97, -1.97] | -2.06 [-2.49, -1.93] | 0.0% | 7.6% | 24.4% | 0.1 |

![Stage 4 5-Seed Heterogeneous Squad Comparison](assets/figures/m2_stage4_heterogeneous_5seed.png)

> [!TIP]
> **Recorded Stage 4 Rollout**: Inspect the heterogeneous `Vanguard`, `Striker`, and `Support` coordination replay: [Stage 4 Heterogeneous Squad Replay (MP4)](assets/videos/dotshot_stage4_heterogeneous.mp4).

**Quantitative & Qualitative Analysis**:
- **Role-Aware Attention Cuts Cross-Seed Variance**: Adding learnable per-agent role embeddings in `RoMAT` produces the best final return IQM (`-2.27 [-2.47, -2.16]`), highest **Core survival (`10.8%`)**, and the **tightest 95% CI width (`0.31`)** across all four Stage 4 baselines. By conditioning self-attention queries on explicit `Vanguard`, `Striker`, and `Support` identities, `RoMAT` avoids gradient interference across incompatible class speeds, HP pools, and `a[3] == 0` tactical abilities.
- **Parameter-Sharing Instability on Asymmetric Squads**: `MAPPO` records a higher best return (`-2.03 [-2.43, -1.88]`) and `0.2` kills/episode on favorable seeds, yet suffers a `0.97`-wide final CI (`[-2.92, -1.95]`), while `IPPO` spans `1.00` (`[-2.97, -1.97]`) with only `24.4%` elemental accuracy. `HAPPO` avoids parameter sharing across roles (`32.9%` elemental accuracy), though `60k` steps splits rollout experience three ways across separate actor networks, leaving Core survival at `6.5%`.

---

### 4.3.4 Controlled Credit-Assignment ($\alpha$) & LOS Occlusion Ablations (3 seeds, 50,000 steps, CUDA)

#### Reward Mixing Parameter $\alpha$ Ablation (`Stage 3 MAPPO`, 50,000 steps, 3 seeds)

| Condition | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ALPHA_0_00` | 3 | -1.05 [-1.62, -0.00] | -0.63 [-1.39, -0.00] | 0.0% | 17.2% | 34.5% | 0.2 |
| `ALPHA_0_25` | 3 | -1.04 [-1.28, -0.79] | -1.04 | 0.0% | 3.1% | 31.1% | 0.3 |
| `ALPHA_0_50` | 3 | -1.68 [-1.84, -1.56] | -1.61 | 0.0% | 13.7% | 36.7% | 0.2 |
| `ALPHA_0_75` | 3 | -2.10 [-2.15, -2.05] | -1.94 | 0.0% | 16.0% | 20.0% | 0.1 |
| `ALPHA_1_00` | 3 | -2.46 [-2.92, -1.83] | -1.89 | 0.0% | 19.6% | 28.9% | 0.0 |

#### Line-of-Sight (LOS) Partial-Observability Ablation (`Stage 3 MAPPO`, 50,000 steps, 3 seeds)

| Condition | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `LOS_OFF` | 3 | -2.15 [-2.28, -2.03] | -1.80 [-1.88, -1.72] | 0.0% | 10.2% | 39.7% | 0.2 |
| `LOS_ON` | 3 | -1.92 [-2.15, -1.76] | -1.77 [-1.84, -1.70] | 0.0% | 12.3% | 29.1% | 0.1 |

![Alpha and LOS Controlled Ablations](assets/figures/m2_ablations_alpha_los.png)

**Quantitative & Qualitative Analysis**:
- **The Credit-Assignment / Core-Defense Tradeoff ($\alpha$)**: Shifting from pure individual shaping (`ALPHA_0_00`: `-1.05 [-1.62, -0.00]`) to pure team reward (`ALPHA_1_00`: `-2.46 [-2.92, -1.83]`) creates a monotonic drop in return and kill rate (`0.2–0.3` kills/ep down to `0.0`), showing classic multi-agent lazy-agent credit dilution when $r_{\text{team}}^t$ dominates without individual feedback. Conversely, `ALPHA_0_25` achieves the highest kills (`0.3`) and best final return (`-1.04 [-1.28, -0.79]`) while neglecting the Core (`3.1%` Core HP), whereas `ALPHA_1_00` preserves the most Core HP (`19.6%`) by hugging the center passively. Intermediate mixing at `ALPHA_0_50` balances both objectives (`13.7%` Core HP, `36.7%` elemental accuracy, `0.2` kills), matching the default `0.45`–`0.50` operating point.
- **LOS Occlusion Sharpens Local Threat Focus**: Disabling cover raycasting (`LOS_OFF`, full `760 px` omniscient enemy slots) improves **elemental accuracy (`39.7%` vs `29.1%`)** and kills (`0.2` vs `0.1`), because agents see incoming elements behind pillars before rounding corners. Yet `LOS_ON` yields higher final return (`-1.92 [-2.15, -1.76]` vs `-2.15 [-2.28, -2.03]`) and Core survival (`12.3%` vs `10.2%`): masking occluded targets removes unshootable behind-wall enemies from the top-5 hostile slots, preventing agents from wasting fire against cover blocks.

> [!NOTE]
> **Archived Milestone 2 Metric Summaries (`docs/results/`)**: [Stage 1](results/m2_stage1_eval.json) · [Stage 2](results/m2_stage2_eval.json) · [Stage 3](results/m2_stage3_eval.json) · [Stage 4](results/m2_stage4_eval.json) · [Alpha Ablation](results/m2_alpha_eval.json) · [LOS Ablation](results/m2_los_eval.json)

---

## 4.4 Milestone 3: Official SOTA Integration, Optuna BOHB, and 1M-Step Converged Benchmarks

Milestone 3 replaces the local headline MARL implementations with pinned upstream code. `third_party/HARL` [@zhong2024harl] supplies MAPPO, HAPPO, HATRPO, and HAA2C on DotShot's native `MultiDiscrete([3, 3, 32, 4])` action space. `third_party/Multi-Agent-Transformer` [@wen2022mat] supplies MAT through an 8-D `Box` action adapter in `src/benchmarks/official/mat_dotshot_env.py`, which maps continuous outputs back to the four factored heads. Stages 1 and 2 keep Stable-Baselines3 PPO [@schulman2017ppo]. Every learned row in Sections 4.4.2 through 4.4.5 comes from a 1,000,000-step run on 3 seeds, with IQM and 95% bootstrap CIs from `docs/results/m3_stage{1..4}_official_eval.json`; `HEURISTIC` and `RANDOM` are scripted baselines evaluated on the same seeds.

### 4.4.1 Optuna BOHB Hyperparameter Search

Each study ran 16 trials with `TPESampler(multivariate=True, seed=42)` and `HyperbandPruner(min_resource=50_000, max_resource=350_000, reduction_factor=3)`, following BOHB [@falkner2018bohb] inside Optuna [@akiba2019optuna]. Trials report intermediate objectives every 50,000 environment steps.

| Study | Best objective | Best trial | Complete / pruned | Top parameter importances |
| :--- | :---: | :---: | :---: | :--- |
| `sb3_ppo_stage1` | `14.57` | 14 | 6 / 10 | `ent_coef` 0.530, `lr` 0.433 |
| `sb3_ppo_stage2` | `9.54` | 4 (`warm_start=True` from Stage 1 trial 14) | 5 / 11 | `ent_coef` 0.897, `lr` 0.083 |
| `official_mat_stage3` | `-0.777` | 3 (`n_block=2`, `n_embd=64`, `n_head=2`, `lr=3.63e-4`, `ppo_epoch=8`) | 4 / 12 | `entropy_coef` 0.555, `lr` 0.368 |
| `harl_mappo_stage3` | `-2.799` | 10 | 5 / 11 | `lr` 0.637, `critic_lr` 0.232 |

![Optuna BOHB Tuning and Parameter Importance](assets/figures/m3_bohb_tuning.png)

BOHB objectives come from the 350,000-step trial ceiling; the converged tables below use 1,000,000 steps. Entropy regularization dominates both single-agent studies: `ent_coef` carries 0.530 of the Stage 1 importance and 0.897 at Stage 2. MAT splits importance between `entropy_coef` (0.555) and `lr` (0.368), while HARL MAPPO leans on the actor and critic learning rates (`lr` 0.637, `critic_lr` 0.232).

The `harl_mappo_stage3` value is a raw 2-agent team-sum objective. The same policy scores `-1.40` per agent. `src/benchmarks/official/harl_runner.py` was aligned to per-agent mean reward before every 1M-step Stage 3 and Stage 4 run, so the converged tables below compare HARL and MAT on the same reward scale.

### 4.4.2 Stage 1 Single-Agent Fundamentals (1,000,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 37.68 [35.34, 39.21] | 37.68 [35.34, 39.21] | 97.9% | 100.0% | 100.0% | 21.1 |
| `PPO` | 3 | 14.95 [7.94, 18.93] | 17.66 [14.86, 19.20] | 50.0% | 100.0% | 100.0% | 8.7 |
| `RANDOM` | 3 | -2.20 [-2.22, -2.18] | -2.20 [-2.22, -2.18] | 0.0% | 100.0% | 42.1% | 0.0 |

![Stage 1 Converged Official PPO](assets/figures/m3_stage1_official_converged.png)

PPO climbs to `14.95 [7.94, 18.93]` final return IQM with a `50.0%` win rate, `100.0%` elemental accuracy, and `8.7` kills per episode. The 200k-step Milestone 2 run reached `3.31 [1.72, 6.41]` and `38.9%` wins on the same stage. The scripted `HEURISTIC` stays ahead at `37.68 [35.34, 39.21]` and `21.1` kills per episode.

### 4.4.3 Stage 2 Elemental Mastery with Transfer (1,000,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HEURISTIC` | 3 | 18.84 [18.07, 20.32] | 18.84 [18.07, 20.32] | 0.0% | 100.0% | 99.7% | 10.9 |
| `PPO` | 3 | 2.81 [0.85, 5.64] | 4.00 [2.65, 6.30] | 13.9% | 100.0% | 39.8% | 1.6 |
| `PPO_TRANSFER` | 3 | 8.91 [7.89, 9.44] | 10.12 [8.44, 12.01] | 66.7% | 100.0% | 36.1% | 3.5 |
| `RANDOM` | 3 | -1.99 [-2.03, -1.95] | -1.99 [-2.03, -1.95] | 0.0% | 100.0% | 27.6% | 0.0 |

![Stage 2 Converged Official PPO Transfer](assets/figures/m3_stage2_official_converged.png)

`PPO_TRANSFER` starts from the converged Stage 1 checkpoint (`runs/m3_official/stage1/ppo/seed_1/checkpoints/best.zip`) using the Stage 2 BOHB `warm_start=True` configuration. It reaches `8.91 [7.89, 9.44]` final return IQM, a `66.7%` win rate, and `3.5` kills per episode. Scratch `PPO` records `2.81 [0.85, 5.64]` and `13.9%` wins, and `RANDOM` stays at `-1.99 [-2.03, -1.95]`. The un-evasive `HEURISTIC` collects `18.84 [18.07, 20.32]` but wins `0.0%` of episodes because it does not dodge incoming fire.

### 4.4.4 Stage 3 Homogeneous Cooperative Defense (1,000,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HARL_HAA2C` | 3 | -2.06 [-2.12, -1.96] | -1.50 [-1.81, -1.09] | 0.0% | 7.9% | 45.1% | 0.2 |
| `HARL_HAPPO` | 3 | -2.17 [-2.38, -1.90] | -1.82 [-2.00, -1.62] | 0.0% | 9.9% | 41.9% | 0.2 |
| `HARL_HATRPO` | 3 | -2.15 [-2.22, -2.11] | -1.90 [-1.98, -1.81] | 0.0% | 1.5% | 42.2% | 0.3 |
| `HARL_MAPPO` | 3 | -2.12 [-2.43, -1.83] | -1.66 [-1.76, -1.47] | 0.0% | 11.8% | 37.5% | 0.2 |
| `OFFICIAL_MAT` | 3 | -1.62 [-1.67, -1.56] | -1.29 [-1.43, -1.20] | 0.0% | 1.4% | 34.9% | 0.9 |

![Stage 3 Converged Official HARL and MAT](assets/figures/m3_stage3_official_converged.png)

`OFFICIAL_MAT` leads final return (`-1.62 [-1.67, -1.56]`), best return (`-1.29 [-1.43, -1.20]`), and kills (`0.9` per episode). `HARL_MAPPO` keeps the most Core HP (`11.8%`), and `HARL_HAA2C` reaches the highest elemental accuracy (`45.1%`). All five algorithms stay at `0.0%` win rate on this wave, so final return and Core retention separate them.

> [!TIP]
> Stage 3 best replay: [m3_official_stage3_best.mp4](assets/videos/m3_official_stage3_best.mp4) (`357` frames, `Core HP: 76.7%`). Evaluation summary: [m3_stage3_official_eval.json](results/m3_stage3_official_eval.json).

### 4.4.5 Stage 4 Heterogeneous Squad (Vanguard, Striker, Support; 1,000,000 steps, 3 seeds)

| Algorithm | Seeds | Final Return IQM [95% CI] | Best Return IQM [95% CI] | Win Rate (%) | Core HP (%) | Elem Acc (%) | Kills / Ep |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `HARL_HAA2C` | 3 | -2.50 [-2.54, -2.45] | -1.99 [-2.13, -1.74] | 0.0% | 3.4% | 34.4% | 0.1 |
| `HARL_HAPPO` | 3 | -2.54 [-2.94, -2.22] | -2.32 [-2.46, -2.22] | 0.0% | 2.1% | 31.9% | 0.4 |
| `HARL_HATRPO` | 3 | -2.53 [-2.68, -2.44] | -2.15 [-2.48, -1.98] | 0.0% | 1.8% | 33.5% | 0.7 |
| `HARL_MAPPO` | 3 | -2.80 [-2.86, -2.74] | -2.38 [-2.57, -2.22] | 0.0% | 1.7% | 31.8% | 0.2 |
| `OFFICIAL_MAT` | 3 | -2.29 [-2.34, -2.19] | -1.61 [-1.73, -1.44] | 0.0% | 4.7% | 33.1% | 0.9 |

![Stage 4 Converged Official HARL and MAT](assets/figures/m3_stage4_official_converged.png)

`OFFICIAL_MAT` leads final return (`-2.29 [-2.34, -2.19]`), best return (`-1.61 [-1.73, -1.44]`), Core HP (`4.7%`), and kills (`0.9` per episode). The heterogeneous HARL algorithms beat shared-parameter `HARL_MAPPO` (`-2.80`) on the asymmetric squad: `HARL_HAA2C` `-2.50`, `HARL_HATRPO` `-2.53`, and `HARL_HAPPO` `-2.54`.

> [!TIP]
> Stage 4 best replay: [m3_official_stage4_best.mp4](assets/videos/m3_official_stage4_best.mp4) (`450` frames, `Score: 300`, `Core HP: 86.7%`). Evaluation summary: [m3_stage4_official_eval.json](results/m3_stage4_official_eval.json).

### 4.4.6 Reproduction Commands and Archived Summaries

```powershell
# Full official 1M-step x 3-seed suite with IQM export
.\.venv\Scripts\python.exe scripts\run_m3_official_suite.py --part all --seeds 0 1 2 --steps 1000000 --device cuda

# Optuna BOHB studies (16 trials each, seed 42)
.\.venv\Scripts\python.exe -m benchmarks.official.bohb_tune --study all --trials 16 --device cuda --db runs\optuna_bohb_mat.db --stage1-best-ckpt runs\bohb\sb3_ppo_stage1\trial_14\checkpoints\best.zip

# Replay official checkpoints (interactive, or add --record for MP4)
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --official-mat runs\m3_official\stage3\official_mat\seed_0\checkpoints\best
.\.venv\Scripts\python.exe scripts\enjoy.py --stage 3 --official-harl runs\m3_official\stage3\harl_mappo\seed_0\checkpoints\best
```

Archived Milestone 3 summaries under `docs/results/`: [Stage 1](results/m3_stage1_official_eval.json), [Stage 2](results/m3_stage2_official_eval.json), [Stage 3](results/m3_stage3_official_eval.json), [Stage 4](results/m3_stage4_official_eval.json), and [BOHB](results/m3_bohb_summary.json).

---

## 4.5 Milestone 3 Performance Fix (branch `fix/marl-performance`, in progress)

This section follows the plan in `PROMPT_fix_marl_performance_complete.md`. Every number below comes from a run executed on this branch after the code change it measures. Milestone 3 numbers in §4.4 are superseded by this section; §4.5.7 lists the bugs that invalidate them.

### 4.5.1 Hardware Budget and Device Choice

Measured on the RTX 3060 workstation (8 physical cores) with 50,000-step runs, one process at a time (`runs/device_benchmark.json`):

| Job | Device | Wall time (s) | Env steps/s | Peak VRAM (MiB) |
| :--- | :---: | ---: | ---: | ---: |
| HARL-MAPPO Stage 3 (32 threads) | CPU | 45.6 | 1,098 | 0 |
| HARL-MAPPO Stage 3 (32 threads) | CUDA | 66.3 | 754 | 32.6 |
| SB3 PPO Stage 1 (8 envs) | CPU | 27.1 | 1,844 | 0 |
| SB3 PPO Stage 1 (8 envs) | CUDA | 55.7 | 897 | 19.1 |

The pure-Python simulator bounds throughput, and the 128×128 MLPs gain nothing from the GPU, so all HARL and SB3 runs use the CPU. At most 6 worker processes run at once (8 physical cores − 2).

### 4.5.2 Phase 0: Scripted Baselines on the Milestone 3 Environment (v2.1)

`src/benchmarks/runners/run_marl_baselines.py`, `configs/baselines/stage{3,4}_baselines_v21.yaml`, 32 `test` episodes each, original per-agent α-mixed reward. Return is the per-agent mean episode return, the metric Milestone 3 reported. CI = 95% percentile bootstrap of the mean over episodes.

| Stage | Policy | Return [95% CI] | Win (%) | Kills / ep | Core HP (%) | Ep. length |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | random | −1.72 [−1.92, −1.52] | 0.0 | 0.0 | 34.4 | 442 |
| 3 | noop | −1.66 [−1.83, −1.50] | 0.0 | 0.0 | 33.9 | 433 |
| 3 | charge (walk into enemies, never fire) | **−1.05** [−1.15, −0.96] | 0.0 | 0.0 | 87.5 | 266 |
| 3 | heuristic_nearest | 23.07 [20.17, 25.99] | 3.1 | 28.2 | 11.3 | 1468 |
| 3 | heuristic_core_guard | 15.93 [14.81, 17.05] | 6.2 | 20.8 | 56.2 | 1204 |
| 4 | random | −2.42 [−2.64, −2.21] | 0.0 | 0.2 | 7.4 | 552 |
| 4 | noop | −2.30 [−2.48, −2.10] | 0.0 | 0.0 | 19.7 | 557 |
| 4 | charge | **−1.25** [−1.36, −1.15] | 0.0 | 0.0 | 84.7 | 387 |
| 4 | heuristic_nearest | 27.35 [24.35, 30.31] | 12.5 | 36.1 | 3.1 | 1516 |
| 4 | heuristic_core_guard | 21.03 [18.64, 23.50] | 6.2 | 28.5 | 47.9 | 1300 |

The suicide policy `charge` beats `random` and `noop` on both stages, and it also beats every Milestone 3 learner (Stage 3 best −1.62, Stage 4 best −2.29). This confirms finding M of the fix plan. Raw per-episode records: `runs/baselines_v21/stage{3,4}/*.json`.

### 4.5.3 Phase 1 Gate, Attempt 1 (FAILED on kills)

Configuration `configs/phase1/stage3_harl_mappo_gate.yaml`: HARL-MAPPO with the Phase 1 fixes (true PPO ratio, masked entropy, dead-agent masks, shared team reward with an EP critic, 4 actor and 4 critic minibatches, linear LR decay from 5e-4, `entropy_coef` 0.01, shared actor), Stage 3 v2.1 dynamics, 2,000,000 steps, seeds 0–4, CPU, 38 min per seed. Each seed's final policy is scored on 32 `test` episodes. The baselines were re-evaluated under the same team reward (`runs/baselines_phase1/stage3`).

| Policy | n | Return mean [95% CI] | Kills / ep | Ep. length | Elem. acc. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| HARL-MAPPO (fixed) | 5 seeds | −2.37 [−2.74, −2.01] | 0.78 | 533 | 25–46 (per seed) |
| noop | 32 episodes | −4.31 [−4.70, −3.94] | 0.0 | 433 | – |
| charge | 32 episodes | −3.04 [−3.31, −2.79] | 0.0 | 266 | – |
| heuristic_nearest | 32 episodes | 52.07 | 28.2 | 1468 | – |

Gate checks (`runs/phase1/gate1_report.json`): beats `noop` beyond the CI: yes. Beats `charge` beyond the CI: yes. Episode length above `charge`: yes. Kills ≥ 3: **no (0.78)**. The gate fails.

Diagnosis from the seed-0 validation reward breakdown at 1.23M steps: Core damage (−4.00) and Core destruction (−1.20) dominate the return. The per-tick aim shaping pays +1.28 per episode, about 3.6 times the correct-hit reward (+0.35), for pointing at the *current* position of the nearest enemy whether or not the agent fires. Elemental accuracy stays at chance. The summed policy entropy is 6.1–6.2 out of a maximum of 7.05 (ln 3 + ln 3 + ln 32 + ln 4), so the aim and weapon heads remain close to uniform. These symptoms match findings N and O, which DotShot v2.2 addresses, rather than an optimisation fault. The retry therefore runs on v2.2 (§4.5.7).

### 4.5.4 Phase 2: DotShot v2.2 Calibration

Calibration loop: one parameter changed at a time, the scripted policies evaluated on 32 `val` episodes per setting (the first Stage 3 probe used 16), logged in `runs/calibration/log.jsonl`. "Defaults" are the v2.2 rules of ADR-014 with the v2.1 difficulty parameters.

| Stage | Setting | Policy | Win (%) | Return | Kills / ep | Core HP (%) | Ep. length |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | defaults | heuristic_calibrated | 78.1 | 85.72 | 41.4 | 22.9 | 1969 |
| 3 | defaults (16 ep.) | noop / charge / random | 0 / 0 / 0 | −7.35 / −10.71 / −6.68 | 0.0 / 0.0 / 0.1 | 0 | 494 / 483 / 503 |
| 4 | defaults | heuristic_calibrated | 100.0 | 105.65 | 57.6 | 84.0 | 2200 |
| 4 | defaults | noop / charge / random | 0 / 0 / 0 | −9.20 / −12.69 / −8.26 | 0.0 / 0.0 / 0.2 | 0 | 594 / 565 / 593 |
| 4 | `spawn_cooldown` 30 → 22 | heuristic_calibrated | 96.9 | 106.48 | 59.3 | 76.9 | 2197 |
| 4 | `enemy_bullet_damage` 10 → 14 | heuristic_calibrated | 96.9 | 101.45 | 56.1 | 60.1 | 2178 |
| 4 | `core_contact_mult` 0.045 → 0.07 | heuristic_calibrated | 100.0 | 105.61 | 57.6 | 83.3 | 2200 |
| 4 | `enemy_speed_mult` 1.18 → 1.35 | heuristic_calibrated | 96.9 | 108.42 | 59.8 | 80.8 | 2197 |
| 4 | `enemy_bullet_damage` → 17 | heuristic_calibrated | 90.6 | 98.58 | 55.4 | 39.2 | 2152 |
| 4 | `enemy_bullet_damage` → 18.5 | heuristic_calibrated | 71.9 | 94.63 | 53.9 | 26.2 | 2095 |
| 4 | `enemy_bullet_damage` → 20 | heuristic_calibrated | 50.0 | 88.48 | 50.9 | 18.9 | 1988 |
| 4 | `enemy_bullet_damage` → 26 | heuristic_calibrated | 21.9 | 65.52 | 39.3 | 6.2 | 1554 |
| 4 | **`enemy_bullet_damage` → 17 (96 episodes, chosen)** | heuristic_calibrated | **80.2** | 98.72 | 55.8 | 31.4 | 2139 |
| 4 | `enemy_bullet_damage` → 17.5 (96 episodes) | heuristic_calibrated | 71.9 | 97.22 | 55.1 | 28.7 | 2115 |
| 3 | defaults (96 episodes) | heuristic_calibrated | 83.3 | 85.53 | 41.3 | 25.0 | 1972 |

A first choice of 18.5 (71.9% on 32 `val` episodes) won only 53.1% of the 32 `test` episodes. With n = 32 the binomial standard error of a win rate near 65% is about 8 points, so the value had been tuned on a noisy estimate. The setting was re-measured on 96 `val` episodes, and 17.0 was chosen (80.2%), leaving margin above the 60% target. Stage 3 keeps the v2.1 difficulty parameters: the calibrated script wins 83.3% of 96 `val` episodes (target ≥ 70%). On Stage 4 with the v2.1 parameters the reference script won every episode with 84% Core HP, so Stage 4 enemy projectiles now deal 17 damage (bosses 1.4×). `noop`, `random` and `charge` win 0% on both stages, and `charge` is now the worst policy instead of the best.

### 4.5.5 v2.2 Baselines on the `test` Split

`configs/baselines/stage{3,4}_baselines.yaml`, 32 `test` episodes per policy, final v2.2 settings, team reward (`runs/baselines_v22/`).

| Stage | Policy | Return [95% CI] | Win (%) | Kills / ep | Core HP (%) | Ep. length |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 3 | random | −6.62 [−6.96, −6.30] | 0.0 | 0.0 | 0.0 | 494 |
| 3 | noop | −7.19 [−7.42, −6.96] | 0.0 | 0.0 | 0.0 | 483 |
| 3 | charge | −10.32 [−10.70, −9.92] | 0.0 | 0.0 | 0.0 | 474 |
| 3 | heuristic_nearest | 87.77 [82.58, 92.59] | 65.6 | 40.4 | 15.9 | 1870 |
| 3 | heuristic_core_guard | 72.83 [66.09, 78.59] | 78.1 | 34.2 | 35.6 | 1853 |
| 3 | heuristic_calibrated | 96.83 [93.41, 100.17] | 71.9 | 41.5 | 20.2 | 1960 |
| 4 | random | −7.70 [−8.11, −7.31] | 0.0 | 0.4 | 0.0 | 541 |
| 4 | noop | −9.02 [−9.21, −8.85] | 0.0 | 0.0 | 0.0 | 542 |
| 4 | charge | −11.58 [−11.93, −11.22] | 0.0 | 0.0 | 0.0 | 509 |
| 4 | heuristic_nearest | 66.92 [61.38, 72.36] | 6.2 | 38.4 | 2.3 | 1526 |
| 4 | heuristic_core_guard | 57.30 [49.39, 65.17] | 15.6 | 34.4 | 4.7 | 1526 |
| 4 | heuristic_calibrated | 108.03 [103.18, 112.22] | 68.8 | 55.7 | 27.9 | 2100 |

Both calibration targets hold on the held-out seeds: 71.9% (Stage 3, ≥ 70%) and 68.8% (Stage 4, ≥ 60%). On Stage 3, `heuristic_core_guard` wins more episodes (78.1%) than the calibrated script but scores a lower return because it kills less.

### 4.5.6 Diagnostics Between Gate Attempts (single seed, 1M steps)

Each row is one seed (seed 0) scored on 32 `test` episodes. The environment changed between groups. Build a is the first v2.2 build. It had a symmetric `±0.6` fire-time credit, and its HARL adapter ended the episode whenever every agent was dead at once, which restored the suicide trap inside HARL training. Build b fixes both, adds the crosshair observation, and uses a `+0.6 / −0.3` credit. Build c raises the credit to `+1.5 / −0.75`.

| Build | Learner and change | Return | Kills / ep | Elem. acc. (%) | Win (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| a | HARL-MAPPO, gate config | −4.23 | 0.2 | 32 | 0 |
| a | HARL-MAPPO, `entropy_coef` 0.0025 | −3.80 | 0.5 | 41 | 0 |
| b | HARL-MAPPO, gate config | −3.35 | 0.5 | 28 | 0 |
| b | HARL-MAPPO, 16 actor / 8 critic minibatches, 10 epochs | −4.80 | 0.2 | 34 | 0 |
| b | same, `entropy_coef` 0.003 | −4.33 | 0.1 | 37 | 0 |
| b | SB3 PPO Stage 1, Milestone 3 hyperparameters | 21.41 | 12.4 | 100 | 71.9 |
| b | SB3 PPO Stage 2, Milestone 3 hyperparameters | 2.62 | 1.6 | 31.9 | 71.9 |
| b | SB3 PPO Stage 2, standard PPO (lr 3e-4, 10 epochs, `ent_coef` 0.01) | 3.95 | 1.8 | 47.6 | 71.9 |
| c | SB3 PPO Stage 2, standard PPO | 6.19 | 3.6 | 62.3 | 6.2 |

Behaviour probes explain the three environment changes. In build a, the HARL policy (6 `val` episodes) held fire on 40% of its actions and hit with the wrong element three times as often as with the right one (39 vs 13 hits). In build b, the Stage 2 PPO policy fired Water on 74% of steps whatever the crosshair showed, which is right only against Fire enemies. In build c, its choice depends on the target: against Water it picks Grass 62% of the time, against Fire it picks Water 61%, but against Grass it still picks Fire only 11% of the time. More gradient steps per sample did not help HARL-MAPPO, so the slow learning comes from the element-choice credit assignment, not from the optimiser budget.

### 4.5.7 Phase 1 Gate, Attempt 2 (FAILED on kills)

Configuration `configs/phase1/stage3_harl_mappo_gate2.yaml`, the attempt 1 learner unchanged, final DotShot v2.2 (build c), 2,000,000 steps, seeds 0–4, 38 min per seed. Baselines: §4.5.5. Report: `docs/results/fix/phase1_gate_attempt2.json`.

| Policy | n | Return mean [95% CI] | Kills / ep | Ep. length | Elem. acc. (%) | Win (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HARL-MAPPO (fixed) | 5 seeds | −1.64 [−1.98, −1.23] | 1.84 | 571 | 39–57 (per seed) | 0.0 |
| noop | 32 episodes | −7.19 [−7.42, −6.96] | 0.0 | 483 | – | 0.0 |
| charge | 32 episodes | −10.32 [−10.70, −9.92] | 0.0 | 474 | – | 0.0 |
| heuristic_calibrated | 32 episodes | 96.83 [93.41, 100.17] | 41.5 | 1960 | – | 71.9 |

Gate checks: beats `noop` and `charge` beyond the CI, episode length above `charge`, but **kills 1.84 < 3**. The gate fails a second time. Validation kills rise on every seed from 0.0–0.2 at step 0 to 1.1–1.8 at 2M; they flatten over the last 0.4M steps while the linearly decaying learning rate approaches zero. The final policies still let the Core fall in every episode (Core damage −4.00 and Core destruction −1.20 on seed 4, against +0.91 for correct hits and +0.48 for kills), and the summed policy entropy stays at 6.1–6.2 of 7.05. Per `PROMPT_pc_full_run.md` §7 the run stops here for a decision on how to proceed.

### 4.5.8 Decision Interval (Action Repeat) A/B and Behaviour Audit

Rationale and decision: ADR-015. Budgets are in simulator ticks; with action repeat k an agent takes one decision every k ticks. Seed 0 unless noted, scored on 32 `test` episodes. "Hold" is the share of live-agent ticks spent holding fire, "Move" the share spent moving; both come from the new behaviour audit and stayed in a healthy range in every run (no idle, non-firing or suicidal policy).

| Learner | k | Ticks | Win (%) | Kills / ep | Elem. acc. (%) | Return | Hold | Move | Wall time |
| :--- | :---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: |
| SB3 PPO Stage 1 | 1 | 1M | 71.9 | 12.4 | 100 | 21.41 | – | – | – |
| SB3 PPO Stage 1 | 8 | 2M | **100.0** | 14.8 | 100 | 31.93 | 0.12 | 0.96 | – |
| SB3 PPO Stage 2 (standard PPO) | 1 | 1M | 6.2 | 3.6 | 62.3 | 6.19 | – | – | 892 s |
| SB3 PPO Stage 2 (standard PPO) | 4 | 2M | 62.5 | 4.8 | 68.2 | 11.28 | 0.07 | 0.96 | 767 s |
| SB3 PPO Stage 2 (standard PPO) | 8 | 2M | **96.9** | 4.7 | 61.6 | 11.56 | 0.03 | 0.99 | 586 s |
| HARL-MAPPO Stage 3 (gate config) | 1 | 2M | 0.0 | 1.84 (5 seeds) | 39–57 | −1.64 | – | – | 2,222 s |
| HARL-MAPPO Stage 3 (gate config) | 4 | 2M | 0.0 | 0.6–0.9 (2 seeds) | 42–55 | −1.91 / −2.64 | 0.04–0.14 | 0.79–0.99 | 979 s |
| HARL-MAPPO Stage 3 (gate config) | 8 | 2M | 0.0 | 1.0 | 44.4 | −2.06 | 0.02 | 0.99 | 749 s |
| HARL-MAPPO Stage 3 (gate config) | 8 | 6M | 0.0 | 2.2 / 2.3 (2 seeds) | 60.8 / 58.1 | **+0.85 / +0.73** | 0.01–0.02 | 0.78–0.86 | ≈ 2,250 s |

At equal wall time, k = 8 gives HARL three times more experience. Within the 2,250 s one 2M-tick gate seed took at k = 1, the k = 8 learner reaches 6M ticks and a positive return. The scratch MARL learners still let the Core fall in every episode: their mean distance to the Core is about 400 px, while the reference script holds a 240 px guard ring. The next round therefore tests a longer budget, a potential-based guard shaping, update intensity, a recurrent actor, and a behaviour-cloning warm start.

### 4.5.9 What Makes Stage 3 Learnable: Budget, Recurrence, Warm Starts (single seed, k = 8)

HARL-MAPPO on Stage 3, seed 0, gate hyperparameters unless noted, 32 `test` episodes (or the latest 16-episode `val` evaluation where marked). Ticks = 8 × decisions.

| Variant | Ticks | Return | Win (%) | Kills / ep | Elem. acc. (%) | Core HP (%) | Move | Notes |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Scratch MLP 128 (gate config) | 6M | 0.85 | 0 | 2.2 | 60.8 | 0 | 0.86 | §4.5.8 |
| **Scratch MLP 128, 24M-tick schedule** | 24M | **53.05** | **25.0** | 24.6 | 90.1 | 5.1 | 0.93 | `val` return 3.3 at 6M, 29.7 at 12M, 48.5 at 20M, 63.1 at 24M |
| Scratch GRU 128 | 6M | 5.23 | 0 | 4.3 | 61.5 | 0 | 0.93 | |
| Scratch MLP 128 + guard potential (`c = 10`) | 6M | 2.29 | 0 | 2.9 | 58.3 | 0 | 0.89 | potential-based, ADR-015 |
| Scratch MLP 128, 16 minibatches × 10 epochs, lr 3e-4 | 3.7M (`val`) | −1.00 | 0 | 1.4 | 53.2 | 0 | 0.93 | stopped: worse |
| Scratch MLP 256 | 3.3M (`val`) | 1.10 | 0 | 2.2 | 71.8 | 0 | 0.95 | running |
| BC (40 episodes, 3 epochs) + RL | 2M | 1.77 | 0 | 3.0 | 68.5 | 0 | 0.46 | clone frozen at step 0 (`IDLE`) |
| **BC (200 episodes, 30 epochs, MLP 256), clone only** | 0 | 52.76 (`val`) | 37.5 | 24.5 | 82.3 | 14.1 | 0.32 | no RL |
| **BC + RL (10 critic-only warm-up updates)** | 2M | **68.69** | **50.0** | 30.8 | 86.7 | 13.5 | 0.37 | lr 5e-4, `entropy_coef` 0.01 |
| reference: `heuristic_calibrated` | – | 87.59 | 78.1 | 37.9 | – | 21.1 | – | 96 `test` episodes |

Findings:
- The behaviour audit caught the first warm start. Cloned from too little data, the deterministic movement heads always chose "stay", so the policy never moved (`move_frac` 0.00, 2.1 deaths per episode). With 200 episodes and 30 epochs, held-out branch accuracy reaches 83% (moves), 82% (aim) and 96% (weapon), and the clone plays the game by itself.
- RL fine-tuning after a critic-only warm-up improves on the clone instead of erasing it (return 52.8 → 68.7, win 37.5% → 50%). A single seed reaches 78% of the reference script's return with a 50% win rate, which already meets the Stage 3 acceptance bar (≥ 50% of the script's return and > 20% wins).
- From scratch, learning is slow but steady and does not plateau within 24M ticks. With a learning-rate schedule sized for 24M ticks, the scratch MLP learns to defend the Core: 53.05 `test` return (61% of the reference script), 25% wins, 24.6 kills and 90% elemental accuracy, with a normal behaviour audit (13% hold fire, 93% moving, 0.4 deaths per episode). Shorter scratch runs (6M ticks) never get that far. Final scratch runs therefore need budgets of the order of 16–24M ticks, while warm-started runs converge within a few million.

Stage 2 PPO, seed 0, standard PPO settings, `val` evaluations at 1.2M ticks: from scratch the policy reaches 58.2% elemental accuracy and 2.8 kills. Started from the Stage 1 policy with its weapon head reset to uniform (`reset_weapon_head`), it reaches **83.1%** and 9.2 kills. The Stage 1 → 2 curriculum transfer is what unlocks the element-choice target.

### 4.5.10 Phase 3 Plan: Optuna Studies, Compute Estimate and Run Order

Protocol (`src/benchmarks/official/bohb_tune.py`): one study per stage and learner (12 studies), 24 complete-or-pruned trials each, `TPESampler(multivariate=True, seed=42)` with `HyperbandPruner(min_resource=125k, max_resource=500k decisions, reduction_factor=3)`, which is 1M–4M ticks at k = 8. Pruning reports every 62.5k decisions (0.5M ticks) on 16 `val` episodes from seed 100. Trials that reach the maximum also train seed 101. The objective is the mean over both seeds of the final policy's return on 32 `val` episodes (never the best checkpoint, never step 0). Searched: actor and critic learning rates, entropy coefficient, clip range or HATRPO `kl_threshold`, PPO/A2C/critic epochs, minibatch counts, γ, GAE λ, hidden size (128/256), MLP vs GRU with chunk length (scratch only), and scratch vs BC warm start (Stages 3–4) or scratch vs Stage 1 transfer (Stage 2). MAT also searches blocks, embedding width and heads. Storage is Optuna's journal file under `runs/optuna/`, because Windows application control on the workstation blocks the SQLAlchemy extensions that SQLite storage needs.

Estimate, from measured k = 8 throughput under 6-way load (HARL Stage 3 ≈ 25 min per 4M ticks, Stage 4 ≈ 1.5×, MAT ≈ 2×, SB3 ≈ 15 min per 4M ticks), assuming about 1.1 full single-seed runs per trial after pruning: Stage 3 HARL 4 × 11 h, Stage 3 MAT 22 h, Stage 4 HARL 4 × 17 h, Stage 4 MAT 33 h, SB3 2 × 3.5 h, about 180 core-hours in total, or roughly 30 h on 6 workers. That is under the 72 h limit, so no cuts are needed.

Run order (parallel lanes via `scripts/optuna_lane.sh`; each Stage 4 study starts with the best Stage 3 parameters of the same learner as trial 0):
1. `s1_sb3_ppo` → `s2_sb3_ppo` → `s3_mat` (second worker) → `s4_harl_haa2c`
2. `s3_harl_mappo` → `s4_harl_mappo`
3. `s3_harl_happo` → `s4_harl_happo`
4. `s3_harl_hatrpo` → `s4_harl_hatrpo`
5. `s3_mat` (CUDA) → `s4_mat` (CUDA)
6. `s3_harl_haa2c` → `s4_mat` (second worker), started when the long scratch probe of §4.5.9 finishes

Progress notes (2026-11-25, 07:00, about 15 h in):
- Tuning is slower than estimated: HyperBand prunes fewer trials than assumed, so the projection is about 45 h, still under the 72 h limit. No trial has crashed, and the GPU stays below 40 °C.
- Every Stage 3 study has a BC-warm-started trial between 78 and 86 val return (calibrated script: 87.6 on `test`). The behaviour audit shows no degenerate policy in those trials. Move fraction is 0.2–0.6, fire rate 7.6–8.0 shots per 100 ticks, elemental accuracy ≥ 89 %, and deaths ≤ 0.6 per episode. Scratch trials stay near zero at 4M ticks (best −1.8 for MAPPO), as §4.5.9 predicted, so scratch hyperparameters are chosen on a weak signal.
- Bug found in Stage 2 transfer: `PPO.load` reseeds torch and the environments with the seed stored in the Stage 1 checkpoint. Both seeds of a transfer trial therefore produced the same run to the last digit; trial 5 scored 44.67 on both. The runner now reseeds after loading, and `tests/test_transfer_seed.py` covers it. The running `s2_sb3_ppo` study keeps the old code because the module is already imported, so its transfer trials are effectively single-seed. The final 5-seed runs use the fixed code.
