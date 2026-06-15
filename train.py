import os
import json
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import env  # Registers ElementShooter-v0 and ElementShooter-Discrete-v0

def load_best_hyperparams(algo="sac", json_path=None):
    if json_path is None:
        json_path = f"./models/best_{algo}_hyperparams.json"
    if os.path.exists(json_path):
        print(f"Loading tuned hyperparameters from {json_path}...")
        with open(json_path, "r") as f:
            return json.load(f)
    print(f"No tuned hyperparameters found. Using default {algo.upper()} parameters.")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Train a SAC or PPO agent on the Element Shooter environment")
    parser.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"], help="Algorithm to use (sac or ppo)")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--tb-log", type=str, default="./tb_log", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./models", help="Model saving directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint model (.zip) to resume training from")
    parser.add_argument("--curiosity", action="store_true", help="Enable Intrinsic Curiosity Module (ICM) exploration rewards")
    args = parser.parse_args()
    
    # 1. Setup folders
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Create environment
    print(f"Initializing environment for {args.algo.upper()}...")
    env_id = "ElementShooter-Discrete-v0" if args.algo == "ppo" else "ElementShooter-v0"
    game_env = gym.make(env_id)
    if args.curiosity:
        from env.curiosity import CuriosityWrapper
        print("Wrapping environment with Intrinsic Curiosity Module (ICM)...")
        game_env = CuriosityWrapper(game_env, eta=0.1)
        
    # 3. Load checkpoint or initialize new model
    AlgoClass = PPO if args.algo == "ppo" else SAC
    prefix = f"{args.algo}_element_shooter"
    
    if args.resume:
        if not os.path.exists(args.resume) and not args.resume.endswith(".zip"):
            checkpoint_path = args.resume + ".zip"
        else:
            checkpoint_path = args.resume
            
        print(f"Resuming training from checkpoint: {checkpoint_path}...")
        
        # Load tuned parameters to override during load
        tuned_params = load_best_hyperparams(args.algo)
        custom_objects = {}
        if tuned_params:
            print("Overriding hyperparameters with new tuned parameters:")
            keys_to_override = ["learning_rate", "gamma", "batch_size"]
            if args.algo == "sac":
                keys_to_override.extend(["tau", "train_freq"])
            elif args.algo == "ppo":
                keys_to_override.extend(["n_steps", "ent_coef"])
                
            for key in keys_to_override:
                if key in tuned_params:
                    custom_objects[key] = tuned_params[key]
                    print(f"  {key}: {tuned_params[key]}")
            print("Note: Network architecture (net_arch) cannot be changed when resuming a checkpoint.")
            
        model = AlgoClass.load(checkpoint_path, env=game_env, custom_objects=custom_objects)
        # Ensure tensorboard logging path is preserved
        model.tensorboard_log = args.tb_log
    else:
        # Load tuned parameters or use defaults
        tuned_params = load_best_hyperparams(args.algo)
        
        learning_rate = tuned_params.get("learning_rate", 3e-4)
        gamma = tuned_params.get("gamma", 0.99)
        batch_size = tuned_params.get("batch_size", 64 if args.algo == "ppo" else 256)
        net_arch_width = tuned_params.get("net_arch_width", 256)
        
        if args.algo == "sac":
            policy_kwargs = dict(
                net_arch=dict(pi=[net_arch_width, net_arch_width], qf=[net_arch_width, net_arch_width])
            )
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[net_arch_width, net_arch_width], vf=[net_arch_width, net_arch_width])
            )
        
        if args.algo == "sac":
            tau = tuned_params.get("tau", 0.005)          # Soft update coefficient
            buffer_size = tuned_params.get("buffer_size", 300000)  # Replay buffer
            learning_starts = tuned_params.get("learning_starts", 1000)  # Steps before learning begins
            train_freq = tuned_params.get("train_freq", 1)  # Update every N steps
            ent_coef = tuned_params.get("ent_coef", "auto")  # SAC auto-tunes entropy
            
            print("\n--- SAC Training Hyperparameters ---")
            print(f"Learning Rate:    {learning_rate:.2e}")
            print(f"Gamma:            {gamma:.4f}")
            print(f"Tau:              {tau:.4f}")
            print(f"Buffer Size:      {buffer_size}")
            print(f"Batch Size:       {batch_size}")
            print(f"Learning Starts:  {learning_starts}")
            print(f"Train Freq:       {train_freq}")
            print(f"Entropy Coef:     {ent_coef}")
            print(f"Network Width:    {net_arch_width}")
            print("------------------------------------\n")
            
            model = SAC(
                "MlpPolicy",
                game_env,
                learning_rate=learning_rate,
                gamma=gamma,
                tau=tau,
                buffer_size=buffer_size,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_freq=train_freq,
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs,
                tensorboard_log=args.tb_log,
                verbose=1
            )
        else:
            # PPO specific hyperparameters
            n_steps = tuned_params.get("n_steps", 2048)  # Steps to run per update
            ent_coef = tuned_params.get("ent_coef", 0.01) # Entropy coefficient for discrete exploration
            
            print("\n--- PPO Training Hyperparameters ---")
            print(f"Learning Rate:    {learning_rate:.2e}")
            print(f"Gamma:            {gamma:.4f}")
            print(f"N Steps:          {n_steps}")
            print(f"Batch Size:       {batch_size}")
            print(f"Entropy Coef:     {ent_coef}")
            print(f"Network Width:    {net_arch_width}")
            print("------------------------------------\n")
            
            model = PPO(
                "MlpPolicy",
                game_env,
                learning_rate=learning_rate,
                gamma=gamma,
                n_steps=n_steps,
                batch_size=batch_size,
                ent_coef=ent_coef,
                policy_kwargs=policy_kwargs,
                tensorboard_log=args.tb_log,
                verbose=1
            )
    
    # 5. Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=args.save_dir,
        name_prefix=prefix
    )
    
    # 6. Learn
    print(f"Starting {args.algo.upper()} training for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps, 
        callback=checkpoint_callback,
        reset_num_timesteps=False if args.resume else True
    )
    
    # 7. Save Final Model
    final_model_path = os.path.join(args.save_dir, f"{prefix}_final")
    model.save(final_model_path)
    print(f"Training finished! Final model saved to {final_model_path}")
    
    game_env.close()

if __name__ == "__main__":
    main()
