# DotShot Game v2: Multi-Agent RL Research Handbook

This handbook documents **DotShot Game v2**, a tactical 2D cooperative shooter simulator paired with a reproducible single-agent and multi-agent Reinforcement Learning (MARL) benchmark suite. Theory, simulator engineering, and experimental results update together after every milestone.

---

## Handbook Structure

| Chapter | Focus | Key Topics Covered |
| :--- | :--- | :--- |
| [1. Evolution from Single-Agent RL to MARL](01-evolution-rl-to-marl.md) | Foundations & Pathologies | MDPs, Bellman equations, Policy Gradient Theorem, GAE, TRPO, PPO, Stochastic Games, Dec-POMDPs, non-stationarity, credit assignment, joint action scaling |
| [2. State of the Art in Cooperative MARL](02-sota-marl-taxonomy.md) | CTDE Algorithms & Proofs | CTDE, IGM, VDN, QMIX, QTRAN, QPLEX, MADDPG, COMA, MA-POCA, IPPO, MAPPO, HAPPO, HATRPO, HARL/HAML, MAT/PMAT/RoMAT, *rliable* statistics |
| [3. DotShot Game v2 Simulator Spec](03-dotshot-simulator-spec.md) | Environment & Engineering | `1440×960` cover/LOS geometry, 4-stage curriculum, `MultiDiscrete([3,3,32,4])` factorization, `129`-dim observation schema, v1-to-v2 post-mortem |
| [4. Living Experiments & Benchmark Log](04-experiments-log.md) | Empirical Benchmarks | Hardware & seed protocol, Stage 1–2 single-agent baselines vs PPO, Stage 3 cooperative MARL benchmarks, IQM + 95% bootstrap CIs, replays |

---

## Reproducibility Quickstart

All handbook experiments run inside the Windows Python 3.11 CUDA environment (`RL/RL-game1/.venv`) driven by versioned YAML files under `configs/`.

```powershell
# Serve this handbook locally
python -m http.server 8080 --directory docs

# Run the automated citation, link, and prose quality gate
.\.venv\Scripts\python.exe scripts/validate_docs.py

# Run full simulator & MARL test suite
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Every inline citation tag (`[@key]`) in these chapters resolves directly to verified primary entries in [references.bib](references.bib).
