import os
import re
import sys
import json
import urllib.request
import subprocess
import argparse

# Default System Prompt to guide the LLM
SYSTEM_PROMPT = """
You are an expert Reinforcement Learning reward design agent. 
Your goal is to write a reward function in Python for an Element Shooter game.
The reward function is defined as `def calculate_reward(env, metrics):` and must return a float.

The game is about a player agent shooting hostile elemental enemies with the counter weapon elements, while avoiding getting hit by enemies and enemy projectiles, avoiding corner-camping, and avoiding hitting/killing glowing white friendly "Ally" entities (they walk randomly across the screen).

The `metrics` parameter is a dictionary containing EXACTLY these keys:
- "steps_survived": int, number of steps survived in the current episode (max 3600).
- "difficulty": float, current difficulty scaling factor (increases over time).
- "agent_healths": list of floats, health of each agent (each max 100.0).
- "num_agents": int, number of agents in the environment (1 or 2).
- "enemies_count": int, count of currently active enemies.
- "bullets_count": int, count of player bullets on screen.
- "enemy_bullets_count": int, count of enemy bullets on screen.
- "score": int, game score (increases when killing enemies).
- "alignment_reward": float, default weapon alignment reward computed during step (based on whether the player aims at the closest enemy when shooting).
- "missed_bullets_count": int, count of player bullets that missed and went off-screen during this step.
- "enemy_hit_correct": int, count of correct weapon hits on hostiles.
- "enemy_killed": int, count of hostiles killed this step.
- "ally_hit": int, count of friendly allies shot by player bullets.
- "ally_killed": int, count of friendly allies killed.
- "damage_taken": float, total contact damage taken from enemies.
- "eb_damage_taken": float, total projectile damage taken from enemy bullets.
- "edge_penalty": float, penalty value calculated if the agent camps near the screen edges.
- "survival_bonus": float, default step survival bonus (0.05).
- "any_dead": bool, True if any agent health reached 0.

Your task:
Write a Python function `def calculate_reward(env, metrics):` that outputs a float reward.
You can use env attributes (e.g., `env.max_health`, `env.width`, etc.) if needed, but it is recommended to design the reward solely using the `metrics` dictionary.

CRITICAL INSTRUCTIONS:
1. Return ONLY the code inside a standard markdown code block: ```python ... ```
2. Do NOT write anything else before or after the code block.
3. Make sure the code is syntactically valid and has no external dependencies outside standard Python and numpy (imported inside your block if needed).
4. Do NOT import gymnasium, stable_baselines3, or pygame inside the function.
5. WARNING: Do NOT use any keys in the `metrics` dictionary other than the ones listed above. For example, do NOT assume keys like 'ally_count', 'weapon_alignment', or 'hostile_kills' exist! They do not! Use only the exact strings in the list above (e.g., 'ally_hit', 'ally_killed', 'alignment_reward', 'enemy_killed').
6. Do NOT put spaces in variable names (e.g., use `CORRECT_HIT_REWARD` instead of `CORRECT Hit_REWARD`). This is a syntax error in Python.
"""

def check_ollama_status(model_name):
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode())
            models = [m.get("name") for m in data.get("models", [])]
            if model_name in models:
                return True
            # Check for name prefix mismatch (e.g. llama3.2 vs llama3.2:latest)
            for m in models:
                if m.startswith(model_name) or model_name.startswith(m):
                    return True
            print(f"Warning: Model '{model_name}' not found in local Ollama tags. Available: {models}")
            return len(models) > 0
    except Exception as e:
        print(f"Error checking Ollama status: {e}")
        return False

def query_ollama(model, system_prompt, user_prompt):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.8
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode())
            return res["message"]["content"]
    except Exception as e:
        print(f"Error querying Ollama API: {e}")
        return None

def extract_python_code(response_text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If not found inside code blocks, try to find lines that define the function
    if "def calculate_reward" in response_text:
        lines = response_text.split("\n")
        start_idx = -1
        for i, line in enumerate(lines):
            if "def calculate_reward" in line:
                start_idx = i
                break
        if start_idx != -1:
            return "\n".join(lines[start_idx:])
    return response_text.strip()

def run_eval_training(steps, num_envs, env_id):
    # Run train.py as a subprocess to train and evaluate
    python_bin = sys.executable
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    if os.path.exists(venv_python):
        python_bin = venv_python
        
    cmd = [
        python_bin, "train.py",
        "--steps", str(steps),
        "--num-envs", str(num_envs),
        "--env-id", env_id
    ]
    print(f"\n[EVALUATION] Starting short training: {' '.join(cmd)}")
    
    # We set stdout/stderr to PIPE to capture outputs
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr

def parse_metrics(stdout):
    ep_rew_mean = None
    ep_len_mean = None
    
    # Stable baselines tabular logger: |    ep_rew_mean          |    -4.52     |
    # Find last occurrences of metrics in logs
    rew_matches = re.findall(r"ep_rew_mean\s*\|\s*([e\d.+\-]+)", stdout)
    len_matches = re.findall(r"ep_len_mean\s*\|\s*([e\d.+\-]+)", stdout)
    
    if rew_matches:
        try:
            ep_rew_mean = float(rew_matches[-1])
        except ValueError:
            pass
    if len_matches:
        try:
            ep_len_mean = float(len_matches[-1])
        except ValueError:
            pass
            
    return ep_rew_mean, ep_len_mean

def main():
    parser = argparse.ArgumentParser(description="Eureka Reward function optimizer for Element Shooter")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to use")
    parser.add_argument("--iterations", type=int, default=3, help="Number of feedback loop iterations")
    parser.add_argument("--steps", type=int, default=60000, help="Steps for each training evaluation")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments to run")
    parser.add_argument("--env-id", type=str, default="ElementShooter-v0", help="Env ID to optimize on")
    args = parser.parse_args()
    
    # Verify Ollama
    if not check_ollama_status(args.model):
        print(f"Aborting: Ollama must be running and have the '{args.model}' model available.")
        sys.exit(1)
        
    print(f"Starting Eureka Reward optimization using Ollama model '{args.model}'...")
    
    # Initial Prompt
    initial_user_prompt = """
Write the initial version of `calculate_reward(env, metrics)`.
For this initial version, you should write a clean, balanced reward function:
- Penalize damage taken (-1.1 contact, -1.5 bullets) and death (-10.0).
- Penalize corner camping (-0.08 edge_penalty).
- Penalize shooting friendly Allies heavily (-15.0 for hit, -25.0 for death).
- Reward correct weapon hit (+3.0) and hostile kill (+10.0).
- Reward aiming alignment correctly (+1.8 for alignment > 0.7, plus +2.6 for alignment > 0.85).
- Penalize bullet misses (-2.0 per miss).
- Include survival bonus (+0.05).

Please design this initial baseline reward function exactly as described, inside the markdown block.
"""

    best_code = None
    best_rew = -float("inf")
    best_len = 0.0
    best_iter = -1
    
    history = []
    
    reward_file_path = "env/reward_function.py"
    backup_file_path = "env/reward_function_backup.py"
    
    # Backup original reward function if it exists
    if os.path.exists(reward_file_path):
        with open(reward_file_path, "r") as f:
            original_code = f.read()
        with open(backup_file_path, "w") as f:
            f.write(original_code)
        print(f"Backed up original reward function to {backup_file_path}")
    else:
        original_code = None
        
    try:
        current_prompt = initial_user_prompt
        
        for iteration in range(args.iterations):
            print(f"\n==================================================")
            print(f"             ITERATION {iteration + 1} / {args.iterations}")
            print(f"==================================================")
            
            # 1. Query LLM for code with syntax-correction retries
            code = None
            for retry in range(4):
                if retry > 0:
                    print(f"Prompting LLM '{args.model}' for syntax correction (try {retry + 1}/4)...")
                else:
                    print(f"Prompting LLM '{args.model}'...")
                    
                response = query_ollama(args.model, SYSTEM_PROMPT, current_prompt)
                if not response:
                    print("Failed to get response from Ollama. Retrying...")
                    continue
                    
                code = extract_python_code(response)
                print(f"Generated Reward Function Code (try {retry + 1}):")
                print("-" * 50)
                print(code)
                print("-" * 50)
                
                # Validate generated code syntax
                try:
                    compile(code, reward_file_path, "exec")
                    print("Code compilation successful!")
                    break # Syntax is valid, exit the retry loop
                except SyntaxError as syntax_err:
                    print(f"Syntax Error detected in generated code: {syntax_err}")
                    if retry == 3:
                        print("Failed to generate syntactically valid code after 4 tries.")
                        code = None
                        break
                    # Update prompt for the next try in the retry loop
                    current_prompt = f"""
The reward function code you wrote failed Python compilation with a SyntaxError:
{syntax_err}

Here is the code you generated:
```python
{code}
```

Please fix the syntax error. Make sure there are no spaces in variable names and the syntax is fully valid. Return ONLY the corrected code inside the ```python markdown block. Keep the function signature `def calculate_reward(env, metrics):`.
"""
            
            if code is None:
                print("Skipping iteration due to repeated syntax errors.")
                continue
            
            # 2. Write code to env/reward_function.py
            with open(reward_file_path, "w") as f:
                f.write(code)
                
            # 3. Launch subprocess training
            return_code, stdout, stderr = run_eval_training(args.steps, args.num_envs, args.env_id)
            
            if return_code != 0:
                print(f"Training failed! Return code: {return_code}")
                # Parse python Traceback error if any from stderr
                error_msg = stderr if stderr else stdout
                print("Error logs:")
                print(error_msg[-800:])  # Print last 800 chars of error
                
                # Feedback loop: send traceback to LLM
                current_prompt = f"The reward function code caused a runtime crash during training with the following traceback/error log:\n\n{error_msg[-1200:]}\n\nPlease analyze the traceback, fix the bug in the code, and return ONLY the corrected code inside the ```python markdown block."
                continue
                
            # 4. Parse training metrics
            ep_rew, ep_len = parse_metrics(stdout)
            
            if ep_rew is None or ep_len is None:
                print("Could not parse episode rewards/lengths from rollout logs. Training stdout might be empty or formatted differently.")
                print("Here is the training output:")
                print(stdout[-1500:])
                current_prompt = "The training completed, but no tabular logs were found in the output. Please verify that the calculate_reward function behaves normally, doesn't return NaN, and return a corrected code block."
                continue
                
            print(f"Iteration {iteration + 1} Performance:")
            print(f"  - Average Episode Reward: {ep_rew:.2f}")
            print(f"  - Average Episode Length: {ep_len:.1f} steps")
            
            history.append({
                "iteration": iteration + 1,
                "reward": ep_rew,
                "length": ep_len,
                "code": code
            })
            
            # 5. Check if it's the best code
            # We prioritize longer survival time (ep_len) and higher score (ep_rew)
            # A longer episode is a stronger signal of success.
            is_better = False
            if ep_len > best_len:
                is_better = True
            elif abs(ep_len - best_len) < 50.0 and ep_rew > best_rew:
                is_better = True
                
            if is_better:
                best_rew = ep_rew
                best_len = ep_len
                best_code = code
                best_iter = iteration + 1
                print("New Best Reward Function Found!")
                # Save best to env/reward_function_best.py
                with open("env/reward_function_best.py", "w") as f:
                    f.write(code)
            
            # 6. Build feedback prompt for the next iteration
            current_prompt = f"""
We evaluated version {iteration + 1} of your reward function. 
Performance results:
- Average steps survived: {ep_len:.1f} steps (maximum possible is 3600.0)
- Average episode reward: {ep_rew:.2f}

Here is the reward function you generated for reference:
```python
{code}
```

How can we modify this function to improve performance further?
Ideas to optimize:
- If the agent is dying too fast (average survival steps is low), increase the penalties for taking damage (`damage_taken` or `eb_damage_taken`) or boost survival bonus slightly.
- If it survives long but gets low scores, reward hostiles killed (`enemy_killed`) and correct weapons hits (`enemy_hit_correct`) more, or shape aiming alignment rewards (`alignment_reward`) to be more dense.
- Ensure allies hit (`ally_hit`) and killed (`ally_killed`) are heavily penalized to prevent collateral damage.
- Keep the reward scale balanced so training remains stable.

Please optimize the reward function code and return ONLY the full revised python function block inside the ```python markdown block.
"""
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        
    print("\n==================================================")
    print("             OPTIMIZATION SUMMARY")
    print("==================================================")
    for run in history:
        print(f"Iteration {run['iteration']}: Avg Reward = {run['reward']:.2f}, Avg Length = {run['length']:.1f} steps")
        
    if best_code:
        print(f"\nBest Reward Function was found in Iteration {best_iter} (Avg Length: {best_len:.1f} steps, Avg Reward: {best_rew:.2f}).")
        # Save the best code to env/reward_function.py
        with open(reward_file_path, "w") as f:
            f.write(best_code)
        print(f"Saved best reward function to {reward_file_path}")
    else:
        print("\nNo valid candidate reward function completed successfully.")
        # Restore backup if available
        if original_code:
            with open(reward_file_path, "w") as f:
                f.write(original_code)
            print(f"Restored original reward function from backup.")
            
    # Cleanup temporary backups
    if os.path.exists(backup_file_path):
        os.remove(backup_file_path)

if __name__ == "__main__":
    main()
