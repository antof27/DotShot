import os
import json
import argparse
import optuna
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import env  # Registers ElementShooter-v0 and ElementShooter-Discrete-v0

def objective(trial, algo):
    """Optuna objective function for hyperparameter tuning."""
    
    # Common hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    net_arch_width = trial.suggest_categorical("net_arch_width", [128, 256, 512])
    
    policy_kwargs = dict(
        net_arch=dict(pi=[net_arch_width, net_arch_width], qf=[net_arch_width, net_arch_width])
    )
    
    # Choose environment based on algorithm
    env_id = "ElementShooter-Discrete-v0" if algo == "ppo" else "ElementShooter-v0"
    game_env = gym.make(env_id)
    
    if algo == "sac":
        tau = trial.suggest_float("tau", 0.001, 0.02)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        buffer_size = trial.suggest_categorical("buffer_size", [100000, 300000, 500000])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
        
        # Initialize SAC model
        model = SAC(
            "MlpPolicy",
            game_env,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            buffer_size=buffer_size,
            train_freq=train_freq,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            verbose=0
        )
    else:
        # PPO specific parameters
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 1e-1, log=True)
        
        # Initialize PPO model
        model = PPO(
            "MlpPolicy",
            game_env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=0
        )
    
    # Short training run for evaluation
    model.learn(total_timesteps=50000)
    
    # Evaluate over 5 episodes
    mean_reward, std_reward = evaluate_policy(model, game_env, n_eval_episodes=5, deterministic=True)
    
    game_env.close()
    
    return mean_reward

def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters using Optuna")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"], help="Algorithm to tune (sac or ppo)")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials to run")
    args = parser.parse_args()
    
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{args.algo}_element_shooter_tuning",
        sampler=optuna.samplers.TPESampler(seed=42)  # Tree-Structured Parzen Estimator (Bayesian)
    )
    
    print(f"Starting Optuna {args.algo.upper()} hyperparameter tuning ({args.trials} trials)...")
    print("Each trial trains for 50K steps and evaluates over 5 episodes.\n")
    
    study.optimize(lambda trial: objective(trial, args.algo), n_trials=args.trials, show_progress_bar=True)
    
    # Print results
    best_trial = study.best_trial
    print(f"\n=== Best Trial ===")
    print(f"Mean Reward: {best_trial.value:.2f}")
    print(f"Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
        
    # Save best hyperparams
    os.makedirs("./models", exist_ok=True)
    output_path = f"./models/best_{args.algo}_hyperparams.json"
    with open(output_path, "w") as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"\nBest hyperparameters saved to {output_path}")

if __name__ == "__main__":
    main()
