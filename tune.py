import os
import json
import argparse
import optuna
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import env  # Registers ElementShooter-v0

# Optimize PyTorch CPU performance by limiting thread count (prevents core thrashing)
# This is dynamic now, but defaults to 1 for vector env efficiency
torch.set_num_threads(1)

class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial's performance to Optuna,
    supporting early stopping (pruning).
    """
    def __init__(self, eval_env, trial, eval_freq, n_eval_episodes=5, *args, **kwargs):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            *args,
            **kwargs
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        
        # Check if evaluation was run in the parent class
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Report last mean reward to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            # Check if the trial should be pruned
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Stops training early
                
        return continue_training

def make_env_fn(env_id, rank, seed=0, wrapper_class=None, wrapper_kwargs=None):
    def _init():
        import env
        import gymnasium as gym
        from stable_baselines3.common.monitor import Monitor
        game_env = gym.make(env_id)
        game_env.reset(seed=seed + rank)
        # Wrap with Monitor to support SB3 statistics logging (resolves user warning)
        game_env = Monitor(game_env)
        if wrapper_class is not None:
            kwargs = wrapper_kwargs if wrapper_kwargs is not None else {}
            game_env = wrapper_class(game_env, **kwargs)
        return game_env
    return _init

def create_vec_env(env_id, num_envs, vec_env_type="dummy", wrapper_class=None, wrapper_kwargs=None, seed=0):
    if vec_env_type == "subproc":
        from stable_baselines3.common.vec_env import SubprocVecEnv
        return SubprocVecEnv([
            make_env_fn(env_id, i, seed=seed, wrapper_class=wrapper_class, wrapper_kwargs=wrapper_kwargs)
            for i in range(num_envs)
        ])
    else:
        from stable_baselines3.common.env_util import make_vec_env
        return make_vec_env(
            env_id,
            n_envs=num_envs,
            wrapper_class=wrapper_class,
            wrapper_kwargs=wrapper_kwargs,
            seed=seed
        )

def objective(trial, curiosity=False, env_id="ElementShooter-v0", steps_per_trial=300000, num_envs=8, eval_freq_steps=20000, vec_env_type="dummy"):
    """Optuna objective function for hyperparameter tuning using BOHB."""
    
    # Optimize PyTorch CPU performance by limiting thread count (prevents core thrashing in subprocesses)
    torch.set_num_threads(1)
    
    # Common hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    net_arch_width = trial.suggest_categorical("net_arch_width", [256, 512, 1024])
    
    policy_kwargs = dict(
        net_arch=dict(pi=[net_arch_width, net_arch_width], vf=[net_arch_width, net_arch_width])
    )
    
    wrapper_class = None
    wrapper_kwargs = None
    if curiosity:
        from env.curiosity import CuriosityWrapper
        wrapper_class = CuriosityWrapper
        
        # Suggest curiosity-specific parameters
        eta = trial.suggest_float("eta", 1e-4, 0.5, log=True)
        beta = trial.suggest_float("beta", 0.05, 0.5)
        lr_icm = trial.suggest_float("lr_icm", 1e-5, 1e-3, log=True)
        R_min = trial.suggest_float("R_min", -30.0, -10.0)
        R_max = trial.suggest_float("R_max", -9.0, -2.0)
        
        wrapper_kwargs = {
            "eta": eta,
            "beta": beta,
            "lr": lr_icm,
            "R_min": R_min,
            "R_max": R_max
        }
        
    game_env = create_vec_env(
        env_id, 
        num_envs=num_envs, 
        vec_env_type=vec_env_type, 
        wrapper_class=wrapper_class, 
        wrapper_kwargs=wrapper_kwargs
    )
    
    # PPO specific parameters
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 5e-4, 2e-2, log=True)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # Create a clean, single-env evaluation environment (no curiosity wrapper)
    # to measure pure task performance, matching the vec_env_type to avoid warnings
    eval_env = create_vec_env(env_id, num_envs=1, vec_env_type=vec_env_type)
    
    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        game_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        verbose=0
    )
    
    eval_freq = max(1, eval_freq_steps // num_envs)
    
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
        verbose=0
    )
    
    callbacks = [eval_callback]
    if curiosity:
        from env.curiosity import CuriosityCallback
        import numpy as np
        obs_dim = int(np.prod(game_env.observation_space.shape))
        action_dim = int(np.prod(game_env.action_space.shape))
        curiosity_callback = CuriosityCallback(
            obs_dim=obs_dim,
            action_dim=action_dim,
            eta=eta,
            beta=beta,
            lr=lr_icm
        )
        callbacks.append(curiosity_callback)
    
    try:
        model.learn(total_timesteps=steps_per_trial, callback=callbacks)
    except optuna.exceptions.TrialPruned:
        eval_env.close()
        game_env.close()
        raise
    
    eval_env.close()
    game_env.close()
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
        
    return eval_callback.last_mean_reward

def main():
    parser = argparse.ArgumentParser(description="Tune PPO hyperparameters using Optuna (BOHB)")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials to run")
    parser.add_argument("--steps-per-trial", type=int, default=300000, help="Max steps per trial")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments per trial")
    parser.add_argument("--eval-freq", type=int, default=20000, help="Frequency of evaluation steps")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel Optuna trials (use -1 for all CPUs)")
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"], help="Vector env type (dummy or subproc)")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db", help="Optuna storage URI (supports SQLite database)")
    parser.add_argument("--curiosity", action="store_true", help="Enable Intrinsic Curiosity Module (ICM) exploration rewards")
    parser.add_argument("--env-id", type=str, default="ElementShooter-v0", help="Gym env ID (curriculum: v0=easy, v1=medium, v2=full)")
    args = parser.parse_args()
    
    # BOHB Setup: TPESampler (Bayesian) + HyperbandPruner (early stopping)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3,
        max_resource=10,
        reduction_factor=3
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="ppo_element_shooter_tuning",
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    print(f"Starting BOHB PPO hyperparameter tuning ({args.trials} trials) on env '{args.env_id}'...")
    print(f"Curiosity (ICM + Dynamic PBRS) wrapper tuning: {args.curiosity}")
    print(f"Config: {args.steps_per_trial} steps per trial, evaluating every {args.eval_freq} steps.")
    print(f"Parallelism: {args.n_jobs} parallel trials, {args.num_envs} vector environments per trial ({args.vec_env}).")
    print(f"Optuna Database: {args.storage}\n")
    
    study.optimize(
        lambda trial: objective(
            trial, 
            curiosity=args.curiosity, 
            env_id=args.env_id,
            steps_per_trial=args.steps_per_trial,
            num_envs=args.num_envs,
            eval_freq_steps=args.eval_freq,
            vec_env_type=args.vec_env
        ),
        n_trials=args.trials,
        n_jobs=args.n_jobs,
        show_progress_bar=(args.n_jobs == 1) # Progress bar is disabled for multi-job tuning
    )
    
    # Print results
    best_trial = study.best_trial
    print(f"\n=== Best Trial ===")
    print(f"Mean Reward: {best_trial.value:.2f}")
    print(f"Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    # Save best hyperparams
    os.makedirs("./models", exist_ok=True)
    output_path = "./models/best_ppo_hyperparams.json"
    with open(output_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"\nBest hyperparameters saved to {output_path}")

if __name__ == "__main__":
    main()
