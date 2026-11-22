# 1. Evolution: From Single-Agent RL to Multi-Agent Systems

> [!NOTE]
> **Learning Objectives**
> By the end of this chapter, you will be able to:
> 1. Formulate single-agent control using MDPs, Bellman optimality, the Policy Gradient Theorem, GAE, TRPO, and PPO.
> 2. Extend single-agent MDPs into Stochastic (Markov) Games and Cooperative Dec-POMDPs.
> 3. Explain mathematically why naive independent learners break down under non-stationarity, multi-agent credit assignment, partial observability, and combinatorial joint action spaces.
> 4. Map each theoretical failure mode onto concrete combat dynamics in **DotShot Game v2**.

---

## 1.1 Intuition: Why One Shooter Cannot Learn Alone in a Team Fight

Picture a single agent inside DotShot Stage 1. The agent moves around four stone pillars, leads a fast Fire enemy with its aim vector, and fires a Water bolt. From the agent's perspective, the environment obeys fixed physical laws: given position, velocity, and enemy state, the next state depends only on the agent's action and fixed simulator dynamics.

Now drop a second agent into Stage 3 to defend the Central Energy Core. Agent 0 moves left to draw a Water boss away from the Core while firing Grass bolts to break its shield. At the exact same tick, Agent 1 shifts its policy and abandons the left flank. The reward Agent 0 receives for holding its ground changes even though the arena physics did not change at all.

The environment ceased to be stationary from Agent 0's viewpoint. That single shift breaks the core mathematical guarantee behind single-agent Reinforcement Learning and motivates the entire Multi-Agent RL stack.

---

## 1.2 Single-Agent Foundations: MDP, Value Functions, and Policy Gradients

A single-agent sequential decision task is modeled as a **Markov Decision Process (MDP)** [@sutton2018rl]:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

- $\mathcal{S}$: state space with state $s_t \in \mathcal{S}$ at step $t$.
- $\mathcal{A}$: action space with action $a_t \in \mathcal{A}$.
- $\mathcal{P}(s_{t+1} \mid s_t, a_t)$: stationary Markov transition kernel.
- $\mathcal{R}(s_t, a_t)$: scalar reward function producing $r_t$.
- $\gamma \in [0, 1)$: discount factor trading immediate rewards against future returns.

An agent follows a stochastic policy $\pi_\theta(a_t \mid s_t)$ parameterized by $\theta$ to maximize expected discounted return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$. Under policy $\pi$, the state-value and action-value functions satisfy the **Bellman expectation equations** [@sutton2018rl]:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}\!\left[ Q^\pi(s, a) \right], \qquad Q^\pi(s, a) = \mathcal{R}(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}(\cdot \mid s, a)}\!\left[ V^\pi(s') \right]$$

For large continuous or high-dimensional discrete spaces (such as DotShot's `129`-dim observation vector and `MultiDiscrete([3, 3, 32, 4])` action space), tabular dynamic programming is intractable. Policy-gradient methods optimize $J(\theta)$ directly using the **Policy Gradient Theorem** [@sutton1999policy]:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\pi, \, a \sim \pi_\theta}\!\left[ \nabla_\theta \log \pi_\theta(a \mid s) \, A^\pi(s, a) \right]$$

where $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ is the **advantage function**. Subtracting the state baseline $V^\pi(s)$ leaves the gradient expectation unchanged while cutting estimator variance sharply.

### Variance Reduction and Trust-Region Updates: GAE, TRPO, and PPO

In practice, $A^\pi(s_t, a_t)$ is unknown and must be estimated from finite rollouts. **Generalized Advantage Estimation (GAE)** [@schulman2016gae] blends 1-step temporal-difference (TD) residuals $\delta_t^V = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ across horizons using parameter $\lambda \in [0, 1]$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

Setting $\lambda = 0$ gives low-variance, high-bias TD(0); $\lambda = 1$ gives high-variance Monte Carlo returns. Standard values ($\gamma = 0.99, \lambda = 0.95$) balance credit propagation across combat waves against variance from stochastic spawns.

Even with clean advantage estimates, naive policy-gradient steps can push $\theta$ too far, collapsing policy performance because the data distribution shifts faster than the critic can adapt. **Trust Region Policy Optimization (TRPO)** [@schulman2015trpo] constrains each update inside a KL-divergence ball using a second-order Fisher-vector conjugate-gradient step. **Proximal Policy Optimization (PPO)** [@schulman2017ppo] achieves comparable stability using only first-order optimization by clipping the importance-sampling ratio $\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \!\left[ \min\!\left( \rho_t(\theta) \hat{A}_t, \; \text{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$

> [!TIP]
> **DotShot Connection (Stages 1 and 2)**
> In DotShot Stages 1 and 2, a single Striker interacts with fixed enemy rules and stationary cover geometry. Standard PPO (`SB3 PPO`) works reliably here because $\mathcal{P}(s' \mid s, a)$ remains stationary throughout training, and dense shaping obeys reward-invariance principles [@ng1999shaping].

---

## 1.3 Stochastic Games and Cooperative Dec-POMDPs

When $n$ agents act simultaneously, single-agent MDPs generalize to **Stochastic (Markov) Games** [@shapley1953stochastic; @littman1994markov]. In cooperative settings where agents share a team goal under partial observability, the standard model is a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)** [@bernstein2002complexity; @oliehoek2016decpomdp]:

$$\mathcal{G} = \langle \mathcal{I}, \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

- $\mathcal{I} = \{1, \dots, n\}$: finite set of $n$ agents.
- $\mathcal{S}$: true global environment state $s_t \in \mathcal{S}$.
- $\mathcal{A} = \prod_{i=1}^n \mathcal{A}_i$: joint action space with joint action $\mathbf{a}_t = (a_t^1, \dots, a_t^n) \in \mathcal{A}$.
- $\mathcal{P}(s_{t+1} \mid s_t, \mathbf{a}_t)$: transition probability conditioned on the **joint** action $\mathbf{a}_t$.
- $\mathcal{R}(s_t, \mathbf{a}_t)$: reward function (shared team reward or mixed individual/team signal).
- $\Omega = \prod_{i=1}^n \Omega_i$: joint observation space, where agent $i$ receives local observation $o_t^i \sim \mathcal{O}(\cdot \mid s_t, i)$.

```mermaid
flowchart TD
    S["Global Arena State s_t"] -->|Partial LOS Filter| O1["Agent 1 Obs o_t^1"]
    S -->|Partial LOS Filter| O2["Agent 2 Obs o_t^2"]
    O1 -->|Decentralized Policy pi_1| A1["Action a_t^1"]
    O2 -->|Decentralized Policy pi_2| A2["Action a_t^2"]
    A1 --> Joint["Joint Action a_t = (a_t^1, a_t^2)"]
    A2 --> Joint
    Joint -->|Transition P(s' | s, a)| NextS["Next State s_{t+1} + Team Reward R"]
    NextS --> S
```

Solving a finite-horizon Dec-POMDP exactly is **NEXP-complete** [@bernstein2002complexity]. Unlike a single-agent POMDP, agents cannot maintain a shared belief over $\mathcal{S}$ at execution time without full communication because each agent observes only its own action-observation history $\tau_t^i = (o_0^i, a_0^i, \dots, o_t^i)$.

---

## 1.4 Four Pathologies That Break Naive Independent Learners

A tempting baseline is **Independent Q-Learning (IQL)** or **Independent PPO (IPPO)** [@dewitt2020ippo]: give each agent $i$ its own critic and actor conditioned solely on $o_t^i$, treating teammates as part of the environment. Four structural pathologies limit this approach.

### 1. Non-Stationarity
From agent $i$'s perspective, the effective transition kernel is:

$$\mathcal{P}_{\text{eff}}^i(s_{t+1} \mid s_t, a_t^i) = \sum_{\mathbf{a}_t^{-i}} \mathcal{P}(s_{t+1} \mid s_t, a_t^i, \mathbf{a}_t^{-i}) \prod_{j \neq i} \pi_{\theta_j}(a_t^j \mid \tau_t^j)$$

As teammates update $\theta_j$, $\mathcal{P}_{\text{eff}}^i$ drifts continuously. Off-policy replay buffers storing old transitions $(o_t^i, a_t^i, r_t, o_{t+1}^i)$ become stale because $\mathbf{a}_t^{-i}$ came from outdated teammate policies. Even on-policy critics experience higher target variance when teammates change tactics simultaneously.

### 2. Multi-Agent Credit Assignment
Suppose in DotShot Stage 3, the Striker breaks a Fire Boss's shield with Water ammo and kills it, yielding a `+8.0` team reward, while a passive teammate hides behind a pillar doing nothing. A shared scalar team reward credits both agents equally. Without counterfactual reasoning, value decomposition, or decomposed individual/team shaping, the lazy agent reinforces hiding while the active agent receives noisy gradients whenever teammates misfire.

### 3. Partial Observability and Relative Overgeneralization
Under Line-of-Sight occlusion around DotShot's cover blocks, Agent 1 may not see an enemy flanking the Central Core behind a pillar. Worse, under **relative overgeneralization**, a coordinated action pair (for example, Vanguard drawing boss aggro while Striker peeks for a rear shot) pays off only if both agents commit together. If early exploration makes the teammate unreliable, independent learners retreat to safe, suboptimal solo camping policies.

### 4. Combinatorial Joint Action Scaling
Each DotShot agent selects from `MultiDiscrete([3, 3, 32, 4])`, which has $3 \times 3 \times 32 \times 4 = 1{,}152$ combinations per agent (and $2{,}304$ in v1's `64`-bin space). Across $n = 3$ agents in Stage 4, the joint action space grows exponentially:

$$|\mathcal{A}| = \prod_{i=1}^n |\mathcal{A}_i| = 1{,}152^3 \approx 1.53 \times 10^9$$

No centralized joint policy $\pi(a^1, \dots, a^n \mid s)$ over flattened discrete indices can explore $1.53 \times 10^9$ actions directly. Overcoming this bottleneck requires **factored action representations** and **autoregressive or decomposed multi-agent architectures**, examined in [Chapter 2](02-sota-marl-taxonomy.md).

---

## 1.5 Common Pitfalls When Moving from Single-Agent to MARL

> [!WARNING]
> **Pitfall 1: Treating Timeouts as Terminal Deaths**
> In survival defense tasks like DotShot Stages 3 and 4, reaching `max_steps` with the Core alive sets `truncated = True` and `terminated = False`. Zeroing out $V(s_T)$ on truncation teaches agents that surviving to the final wave has zero future value, distorting GAE advantages near episode end.

> [!WARNING]
> **Pitfall 2: Flattening MultiDiscrete Actions**
> Legacy DotShot v1 encoded `[move_x, move_y, aim_bin, weapon]` into a single integer in $[0, 2303]$. Aim bin `15` and aim bin `16` shared zero parameters in the output layer, forcing the network to relearn geometry separately for every movement and weapon combination. Factoring the policy into four independent heads ($\log \pi_i = \sum_{k=1}^4 \log \pi_i^{(k)}$) cuts output weights from $2{,}304$ logits to $3 + 3 + 32 + 4 = 42$ logits per agent.

---

## 1.6 Quick Self-Check

1. **Why does subtracting a state-dependent baseline $b(s)$ reduce Policy Gradient variance without introducing bias?**
   <details>
   <summary>Show Answer</summary>
   Because $\mathbb{E}_{a \sim \pi_\theta(\cdot \mid s)}[\nabla_\theta \log \pi_\theta(a \mid s) \, b(s)] = b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s) \nabla_\theta 1 = 0$. Any baseline independent of $a$ cancels in expectation while lowering the variance of the Monte Carlo return multiplier.
   </details>

2. **In a Dec-POMDP with $n=3$ agents, why can we train with a centralized critic $V(s_t)$ even though execution must remain decentralized?**
   <details>
   <summary>Show Answer</summary>
   During execution, only the actor $\pi_i(a_t^i \mid o_t^i)$ is deployed; it reads solely local observation $o_t^i$. The critic $V(s_t)$ is discarded at test time and used only during training to compute lower-variance advantages $\hat{A}_t$ from the true global state $s_t$.
   </details>

3. **Which two DotShot Stage 4 mechanisms directly test multi-agent credit assignment and heterogeneity?**
   <details>
   <summary>Show Answer</summary>
   (a) Elemental-shielded Bosses require counter-element shield shattering followed by focus-fire combo kills; (b) `Vanguard` (shield tank), `Striker` (DPS), and `Support` (repair pulse + vulnerability mark) perform distinct roles whose benefits accrue partly to teammates rather than solely to the actor triggering them.
   </details>
