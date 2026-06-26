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

def objective(trial, curiosity=False, env_id="ElementShooter-v0"):
    """Optuna objective function for hyperparameter tuning using BOHB."""
    from stable_baselines3.common.env_util import make_vec_env
    
    # Common hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    net_arch_width = trial.suggest_categorical("net_arch_width", [256, 512, 1024])
    
    policy_kwargs = dict(
        net_arch=dict(pi=[net_arch_width, net_arch_width], vf=[net_arch_width, net_arch_width])
    )
    
    if curiosity:
        from env.curiosity import CuriosityWrapper
        
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
        game_env = make_vec_env(
            env_id, 
            n_envs=8,
            wrapper_class=CuriosityWrapper,
            wrapper_kwargs=wrapper_kwargs
        )
    else:
        game_env = make_vec_env(env_id, n_envs=8)
    
    # PPO specific parameters
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 5e-4, 2e-2, log=True)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 1.0])
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    
    # Create a clean, single-env evaluation environment (no curiosity wrapper)
    # to measure pure task performance
    eval_env = make_vec_env(env_id, n_envs=1)
    
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
    
    # 100K training steps, evaluate every 10K steps (1250 updates for 8 envs)
    total_timesteps = 100000
    eval_freq_steps = 10000
    n_envs = 8
    eval_freq = eval_freq_steps // n_envs
    
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
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
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
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run")
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
        sampler=sampler,
        pruner=pruner
    )
    
    print(f"Starting BOHB PPO hyperparameter tuning ({args.trials} trials) on env '{args.env_id}'...")
    print(f"Curiosity (ICM + Dynamic PBRS) wrapper tuning: {args.curiosity}")
    print("Each trial trains for up to 100K steps, evaluating every 10K steps over 5 episodes.\n")
    
    study.optimize(
        lambda trial: objective(trial, curiosity=args.curiosity, env_id=args.env_id),
        n_trials=args.trials,
        show_progress_bar=True
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
