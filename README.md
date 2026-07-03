# DotShot RL

A 2D top-down cooperative shooter environment built with Gymnasium and Pygame for training reinforcement learning agents. The agents control their movement and aiming vectors (twin-stick layout) and must dynamically switch weapons to counter incoming enemies, optionally coordinating with an ally.

---

## Element Matchups
- **Water Gun** (blue projectile) counters **Fire Enemy** (red, fast)
- **Fire Gun** (red projectile) counters **Grass Enemy** (green, slow)
- **Grass Gun** (green projectile) counters **Water Enemy** (blue, medium)

*Note: Firing the incorrect weapon deals 0 damage and incurs a reward penalty.*

---

## Key Environment Mechanics

### 1. Interception Aiming (Target Leading)
Enemies move continuously toward the agent. To aim accurately, the agent's observation space includes the **velocity vector ($v_x, v_y$)** of the 3 closest enemies. This allows the policy to learn to calculate interception angles and lead moving targets.

### 2. Bullet Discipline (In-Flight Capping)
To prevent the agent from spamming bullets and wasting ammunition, the environment tracks active bullets. If a bullet is already in flight toward a target, subsequent trigger pulls at that same target are automatically suppressed (without consuming the weapon cooldown).

### 3. Dynamic Potential-Based Reward Shaping (DPBRS)
To boost exploration without distorting the true reward landscape, the environment wraps training with a dynamic potential-based curiosity module. It automatically tracks the historical best and worst rolling average rewards (`running_min_rew`, `running_max_rew`) to dynamically fade out exploration rewards as the agent masters the task.

### 4. Observation Space
Each agent receives a **70-dimensional** observation space:
- **Agent state (9 features)**: Position, velocity, health, cooldown, and last fired weapon.
- **3 Closest Hostile Enemies (39 features)**: Relative position, angle, distance, health, type (one-hot), correct weapon hint, and **velocity ($v_x, v_y$)**.
- **2 Closest Allies (22 features)**: Relative position, angle, distance, health, and active/inactive status.

---

## Getting Started

### 1. Installation
Activate the virtual environment:
```bash
source .venv/bin/activate
```
Verify the environment registration and dimensions:
```bash
python3 test_env.py
```

### 2. Manual Play (Play it Yourself)
To test the game mechanics, element switching, and control layout yourself:
```bash
python3 enjoy.py --manual
```
**Controls**:
- **Movement**: `W`, `A`, `S`, `D` / Arrow keys
- **Aiming**: Mouse cursor
- **Weapons**: `1` (Water), `2` (Grass), `3` (Fire)
- **Shoot**: Left Click / `Space`

---

## Standard Single-Agent Training & Tuning (Stable Baselines 3)

### 1. Hyperparameter Tuning (`tune.py`)
Search for optimal PPO hyperparameters using Optuna (BOHB with Hyperband pruning):
```bash
python3 tune.py [options]
```
#### Options:
- `--trials <int>`: Total trials to run (default: `50`)
- `--steps-per-trial <int>`: Max steps per trial (default: `300000`)
- `--num-envs <int>`: Number of parallel environments per trial (default: `8`)
- `--vec-env {dummy,subproc}`: Use `subproc` to parallelize environment stepping across CPU cores (default: `dummy`)
- `--n-jobs <int>`: Number of parallel Optuna trials to run simultaneously (default: `1`)
- `--storage <str>`: SQLite database storage URL to coordinate distributed trials (default: `sqlite:///optuna_study.db`)
- `--curiosity`: Enable the Intrinsic Curiosity Module (ICM)
- `--device {cpu,cuda,auto}`: Device to run PyTorch calculations on (default: `cpu`)

*Tip: Because tuning coordinates through a SQLite database, you can run multiple independent `python3 tune.py` instances in parallel (e.g. on different GPUs or shell terminals), and they will coordinate and share the search space.*

---

### 2. Agent Training (`train.py`)
Train the agent using PPO and GAE:
```bash
python3 train.py [options]
```
#### Options:
- `--steps <int>`: Total training timesteps (default: `1000000`)
- `--num-envs <int>`: Number of parallel environments (default: `8`)
- `--vec-env {dummy,subproc}`: Use `subproc` to run environments in parallel child processes for a **4x to 8x speedup**.
- `--torch-threads <int>`: Set PyTorch CPU thread limit to prevent core thrashing (default: `1`)
- `--device {cpu,cuda,auto}`: Device to run PyTorch models on (default: `cpu`)
- `--curiosity`: Train with Intrinsic Curiosity Module (ICM) exploration rewards
- `--env-id <str>`: Gymnasium level: `DotShot-Level1-v0` (easy), `DotShot-Level2-v0` (medium), `DotShot-Level3-v0` (full 2-agent co-op)
- `--resume <path>`: Resume training from a saved checkpoint ZIP

---

### 3. Checkpoint Weight Surgery (`surgery.py`)
Convert a trained Level 2 single-agent model (`.zip`) into an initialized Level 3 multi-agent model (`.zip`) so you don't have to train Level 3 from scratch:
```bash
python3 surgery.py --source ./models/ppo_element_shooter_final.zip --target ./models/ppo_level3_init.zip
```

---

## Multi-Agent Reinforcement Learning (MARL) Framework

For advanced, sequence-modeling cooperative play, the repository includes a custom PyTorch MARL framework implementing **Multi-Agent Transformers** under the `marl/` directory.

### 1. Supported Algorithms
*   **MAT (Multi-Agent Transformer)**: Autoregressively sequences action choices between cooperating agents using a fixed decision order (`[0, 1]`).
*   **PMAT (Prioritized Multi-Agent Transformer)**: Dynamically decides the optimal decision order step-by-step using a Plackett-Luce sampling mechanism.
*   **RoMAT (Role-based Multi-Agent Transformer)**: Concatenates a learnable role embedding (`Initiator`, `Support`) to each agent's observations, allowing role specialization.

### 2. Train MARL Algorithms (`marl/train_marl.py`)
Train a custom Transformer-based policy using PPO:
```bash
python3 marl/train_marl.py [options]
```
#### Options:
- `--algo {mat,pmat,romat}`: Select the MARL algorithm to train (default: `pmat`)
- `--env-id <str>`: Gymnasium environment level (default: `DotShot-Level3-v0`)
- `--steps <int>`: Total training steps (default: `1000000`)
- `--num-envs <int>`: Number of parallel environments (default: `8`)
- `--vec-env {dummy,subproc}`: Vectorized environment type (default: `subproc`)
- `--resume <path>`: Path to a custom MARL checkpoint (`.pt`) to resume training
- `--device {cpu,cuda,auto}`: Device to train models on (default: `cpu`)

---

### 3. MARL Checkpoint Surgery (`marl/surgery_marl.py`)
Bootstrap Level 3 MARL training using a pre-trained Level 2 MARL checkpoint by performing weight surgery on the Centralized Critic input layer:
```bash
python3 marl/surgery_marl.py --source ./models_marl/pmat_final.pt --target ./models_marl/pmat_level3_init.pt
```

---

## Evaluation & Visualization (`enjoy.py`)

Watch any trained model (Stable Baselines 3 or custom PyTorch MARL) play the game:
```bash
python3 enjoy.py --env-id DotShot-Level3-v0 --model <path_to_checkpoint>
```
*Note: If the model path ends with `.zip`, it will automatically load via Stable Baselines 3. If it ends with `.pt`, it will load via our custom PyTorch MARL policy runner, ensuring seamless compatibility.*

---

## Recording Gameplay Videos (`record_video.py`)
Record high-resolution agent gameplay videos directly to MP4:
```bash
python3 record_video.py [options]
```
#### Options:
- `--model <path>`: Path to the trained model (default: `./models/ppo_element_shooter_final`)
- `--env-id <str>`: Gym env ID (default: `DotShot-Level1-v0`)
- `--output <path>`: Output video filename (default: `agent_gameplay.mp4`)
- `--steps <int>`: Maximum steps to record (default: `3600`)
- `--fps <int>`: Video frame rate (default: `60`)
- `--device {cpu,cuda,auto}`: PyTorch device to run model on (default: `cpu`)
