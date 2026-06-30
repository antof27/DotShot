# Element Shooter RL

A 2D top-down shooter environment built with Gymnasium and Pygame for training reinforcement learning agents. The agent controls its movement and aiming vectors (twin-stick layout) and must dynamically switch weapons to counter incoming enemies.

---

## Element Matchups
- **Water Gun** (blue projectile) counters **Fire Enemy** (red, fast)
- **Fire Gun** (red projectile) counters **Grass Enemy** (green, slow)
- **Grass Gun** (green projectile) counters **Water Enemy** (blue, medium)

*Note: Firing the incorrect weapon deals 0 damage and incurs a reward penalty.*

---

## Key Environment Mechanics

### 1. Interception Aiming (Target Leading)
Enemies move continuously toward the agent. To aim accurately, the agent's observation space includes the **velocity vector ($v_x, v_y$)** of the 3 closest enemies. This allows the PPO policy to learn to calculate interception angles and lead moving targets.

### 2. Bullet Discipline (In-Flight Capping)
To prevent the agent from spamming bullets and wasting ammunition, the environment tracks active bullets. If a bullet is already in flight toward a target, subsequent trigger pulls at that same target are automatically suppressed (without consuming the weapon cooldown).

### 3. Observation Space
Each agent receives a **70-dimensional** observation space containing:
- **Agent state (9 features)**: Position, velocity, health, cooldown, and last fired weapon.
- **3 Closest Hostile Enemies (39 features)**: Relative position, angle, distance, health, type (one-hot), correct weapon hint, and **velocity ($v_x, v_y$)**.
- **2 Closest Allies (22 features)**: Relative position, angle, distance, health, and status.

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

## Training & Tuning

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
- `--curiosity`: Train with Intrinsic Curiosity Module (ICM) exploration rewards
- `--env-id <str>`: Gymnasium level: `ElementShooter-v0` (easy), `ElementShooter-v1` (medium), `ElementShooter-v2` (full 2-agent co-op)
- `--resume <path>`: Resume training from a saved checkpoint ZIP

Monitor training real-time in TensorBoard:
```bash
tensorboard --logdir=./tb_log
```

---

### 3. Evaluate & Watch the Agent (`enjoy.py`)
To watch a trained policy play:
```bash
python3 enjoy.py --env-id ElementShooter-v0
```

---

### 4. Record MP4 Gameplay Videos (`record_video.py`)
Record high-resolution agent gameplay videos directly to MP4:
```bash
python3 record_video.py [options]
```
#### Options:
- `--model <path>`: Path to the trained model `.zip` (default: `./models/ppo_element_shooter_final`)
- `--env-id <str>`: Gym env ID (default: `ElementShooter-v0`)
- `--output <path>`: Output video filename (default: `agent_gameplay.mp4`)
- `--steps <int>`: Maximum steps to record (default: `3600`)
- `--fps <int>`: Video frame rate (default: `60`)
- `--no-deterministic`: Use stochastic action selection instead of deterministic
