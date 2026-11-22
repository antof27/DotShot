---
name: marl-benchmark-design
description: >-
  Design Dec-POMDP PettingZoo ParallelEnv environments, implement CTDE MARL
  algorithms (IPPO, Branching-VDN/QMIX, MAPPO, HAPPO/HATRPO, MAT/PMAT/RoMAT),
  avoid factored-action and credit-assignment pitfalls, and aggregate multi-seed
  benchmarks using rliable IQM and bootstrap confidence intervals.
---

# MARL Benchmark Design & CTDE Implementation Guide

Use this skill when building `dotshot/envs/multi_agent.py`, PyTorch MARL networks/trainers under `benchmarks/marl/`, and statistical evaluators under `benchmarks/eval/`.

---

## 1. Dec-POMDP & PettingZoo `ParallelEnv` Contract

Model DotShot Stages 3 and 4 formally as a Dec-POMDP $\langle \mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$:

- **Strict Information Separation**:
  - **Execution-time local observation $o_i^t$**: Only egocentric telemetry, line-of-sight gated enemies (within vision/LOS), Core vector, local LIDAR, and teammate relative positions. Never leak unseen off-screen enemy state or global hidden timers into $o_i^t$.
  - **Training-time privileged global state $s^t$** (`env.state()`): Full noise-free arena snapshot (all agents' states + classes, Core HP, all active enemies + boss shield timers, wave counters). Consumed solely by centralized critics and mixing networks during training.
- **PettingZoo API Compliance**:
  - Implement `possible_agents`, `agents`, `observation_space(agent)`, `action_space(agent)`, `reset(seed=...)`, `step(actions)`, and `state()`.
  - Pass `pettingzoo.test.parallel_api_test(env, num_cycles=200)` in automated tests.
  - Handle dead agents cleanly: either keep fixed agent slots with an explicit `alive_mask` (zeroing rewards/actions when dead) or follow PettingZoo termination conventions consistently across buffers and advantage calculations.

---

## 2. Factored `MultiDiscrete` Action Architecture

DotShot's action per agent is `MultiDiscrete([3, 3, 32, 4])` (`move_x`, `move_y`, `aim_bin`, `weapon`).

- **Never flatten to a single $3 \times 3 \times 32 \times 4 = 1{,}152$ or $2{,}304$ Categorical head.** Flattening destroys directional locality (aim bin 15 vs 16) and explodes variance during exploration.
- **Actor Factorization**: Output four categorical logits from a shared trunk:
  $$\log \pi_i(a_i \mid o_i) = \sum_{k=1}^{4} \log \pi_i^{(k)}(a_i^{(k)} \mid o_i), \qquad H(\pi_i(\cdot \mid o_i)) = \sum_{k=1}^{4} H(\pi_i^{(k)}(\cdot \mid o_i))$$
- **Value-Based Factorization (`VDN` / `QMIX`)**: Use a **Branching Dueling Q-Network** (Tavakoli et al., 2018): shared state representation $V_i(o_i)$ plus per-branch advantages $A_i^{(k)}(o_i, a_i^{(k)})$, yielding individual agent utility $Q_i(o_i, a_i) = V_i(o_i) + \frac{1}{4}\sum_{k=1}^{4} \big(A_i^{(k)}(o_i, a_i^{(k)}) - \bar{A}_i^{(k)}\big)$ passed into the VDN sum or QMIX hypernetwork.

---

## 3. Algorithm Invariants & Known Pitfalls

| Algorithm | Core Invariant to Verify | Common Implementation Bug |
| :--- | :--- | :--- |
| **IPPO** | Decentralized critic $V_{\phi_i}(o_i)$ sees only local $o_i$ | Accidentally passing global state $s^t$ to IPPO critic |
| **VDN / QMIX** | Individual-Global-Max (IGM): $\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0$ via `torch.abs(w)` on mixing hypernetwork weights + target networks | Allowing negative hypernetwork weights or updating on shuffled single transitions instead of episode/trajectory sequences |
| **MAPPO** | Centralized critic $V_\phi(s^t)$ + ValueNorm (PopArt/running mean-std) + death/active mask on loss (Yu et al., 2022) | Omitting value normalization or sharing actor weights across incompatible heterogeneous classes without role IDs |
| **HAPPO / HATRPO** | Separate policy parameters per agent; random permutation $i_{1:n}$ per update; sequential importance factor $M^{i_{1:m}} = \frac{\pi^{i_m}_{\text{new}}}{\pi^{i_m}_{\text{old}}} M^{i_{1:m-1}}$ (Kuba et al., 2022; Zhong et al., 2024) | Updating agents in parallel using the same stale $A_{\text{old}}$ without multiplying the sequential policy ratio factor $M$ |
| **MAT / PMAT / RoMAT** | Causal mask in Transformer decoder so agent $i_m$ conditions on $o_{1:n}$ and previous actions $a_{i_1:i_{m-1}}$; linear complexity in $n$ (Wen et al., 2022) | Leaking future agent actions in decoder cross-attention or forgetting to detach/include Plackett-Luce ordering log-probs cleanly |

---

## 4. Statistical Evaluation Protocol (`rliable`)

Follow Agarwal et al. (2021) and Gorsane et al. (2022):
- Run at least **3 seeds** for fast ablations/smoke tests and **5+ seeds** for benchmark comparisons, evaluated over $N_{\text{eval}} \ge 32$ deterministic episodes per checkpoint.
- Report **Interquartile Mean (IQM)**, **Mean**, **Median**, and **95% stratified bootstrap confidence intervals** across seeds alongside sample-efficiency curves (Return, Core Survival %, Elemental Accuracy %, Synergy Combo Count vs environment steps).
- Never claim superiority from a single seed or overlapping wide CIs; report overlap explicitly.
