# DotShot: Multi-Agent RL Handbook

This handbook documents DotShot, a 2D arena shooter with a four-stage curriculum for single- and multi-agent reinforcement learning, together with the learners trained on it and the results. The theory, the simulator and the experiments are kept in step with the code.

| Chapter | Contents |
| :--- | :--- |
| [1. From Single-Agent RL to MARL](01-evolution-rl-to-marl.md) | MDPs, policy gradients, GAE, TRPO and PPO; stochastic games and Dec-POMDPs; why independent learners struggle; common implementation pitfalls |
| [2. Cooperative MARL Algorithms](02-sota-marl-taxonomy.md) | CTDE; value factorisation (VDN, QMIX, QTRAN, QPLEX); centralised critics (MADDPG, COMA, MA-POCA); MAPPO; HAPPO, HATRPO and HARL; MAT; evaluation standards |
| [3. DotShot Simulator Specification](03-dotshot-simulator-spec.md) | The four stages, entities, the `MultiDiscrete([3, 3, 32, 4])` action space, the 166-value observation, the global state, every reward term, calibration and the behaviour audit |
| [4. Experiments and Results](04-experiments-log.md) | Hardware, protocol and baselines; how the benchmark became learnable; hyperparameter search; final 5-seed results |

The [report](paper/dotshot_marl_report.pdf) presents the same material as a single paper.

## Reproducing

All commands run from the repository root in the project environment (`.venv`, Python 3.11):

```powershell
# Serve this handbook locally
python -m http.server 8080 --directory docs

# Check citations and links
.\.venv\Scripts\python.exe scripts\validate_docs.py

# Run the test suite
.\.venv\Scripts\python.exe -m pytest
```

Citations of the form `[@key]` resolve to verified entries in [references.bib](references.bib).
