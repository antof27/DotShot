# Element Shooter - Reinforcement Learning Project

A custom 2D top-down continuous shooter game designed for training reinforcement learning (RL) agents. The agent must control movement and aiming vectors (twin-stick control) while switching between countering elements to shoot and kill incoming enemies before taking damage.

## Element Matchups
- **Water Gun** (blue projectile) counters **Fire Enemy** (red, moves fast).
- **Fire Gun** (red projectile) counters **Grass Enemy** (green, moves slow).
- **Grass Gun** (green projectile) counters **Water Enemy** (blue, moves medium).
- **Wind Gun** (white projectile) counters **Flying Enemy** (purple, moves very fast).

Shooting the wrong weapon deals 0 damage and incurs a reward penalty.

---

## Installation & Setup

1. **Activate the Virtual Environment**:
   ```bash
   source .venv/bin/activate
   ```
   *(If not already activated, dependencies are installed inside the `.venv` directory).*

2. **Verify Environment Setup**:
   Run the sanity check script to verify Gymnasium registration, observations, and text descriptions:
   ```bash
   python3 test_env.py
   ```

---

## How to Play

### 1. Manual Play Mode (Test the Game Yourself)
Before training an agent, you can play the game manually to experience the controls, element switching, and mechanics:
```bash
python3 enjoy.py --manual
```
**Controls**:
- **Movement**: `W`, `A`, `S`, `D` (or Arrow keys)
- **Aiming**: Move the mouse cursor
- **Select Element Weapon**:
  - `1` : Water Gun (counters Fire)
  - `2` : Grass Gun (counters Water)
  - `3` : Fire Gun (counters Grass)
  - `4` : Wind Gun (counters Flying)
- **Shooting**: Click **Left Mouse Button** (or press `Space`)

---

### 2. Hyperparameter Tuning (Optuna)
To search for the optimal PPO hyperparameters (learning rate, entropy coefficient for exploration, GAE lambda, gamma, policy network size) using Optuna:
```bash
python3 tune.py
```
This runs 12 trials of 25,000 steps each. The best hyperparameters are automatically saved to `./models/best_hyperparams.json`.

---

### 3. Agent Training (PPO with GAE)
To train the reinforcement learning agent (PPO with GAE) using either the default parameters or the tuned parameters found by Optuna:
```bash
python3 train.py --steps 100000
```
This will train the agent for 100,000 steps, save checkpoint models, save the final model to `./models/ppo_element_shooter_final`, and output Tensorboard logs to `./tb_log`.

You can monitor training in real-time using TensorBoard:
```bash
tensorboard --logdir=./tb_log
```

---

### 4. Watch the Trained Agent Play
Once training is complete, watch your trained RL agent play the game:
```bash
python3 enjoy.py
```
It loads the final trained model from `./models/ppo_element_shooter_final` and runs the game loop with Pygame visual rendering.
