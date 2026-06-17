import os
import json
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import env  # Registers ElementShooter-v0

# Optimize PyTorch CPU performance by limiting thread count (prevents core thrashing)
torch.set_num_threads(1)

def load_best_hyperparams(json_path=None):
    if json_path is None:
        json_path = "./models/best_ppo_hyperparams.json"
    if os.path.exists(json_path):
        print(f"Loading tuned hyperparameters from {json_path}...")
        with open(json_path, "r") as f:
            return json.load(f)
    print("No tuned hyperparameters found. Using default PPO parameters.")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Element Shooter environment")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--tb-log", type=str, default="./tb_log", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./models", help="Model saving directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint model (.zip) to resume training from")
    parser.add_argument("--curiosity", action="store_true", help="Enable Intrinsic Curiosity Module (ICM) exploration rewards")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments to run (default: 8)")
    parser.add_argument("--env-id", type=str, default="ElementShooter-v0", help="Gym env ID (curriculum: v0=easy, v1=medium, v2=full)")
    args = parser.parse_args()
    
    # 1. Setup folders
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Create environment
    from stable_baselines3.common.env_util import make_vec_env
    env_id = args.env_id
    print(f"Initializing {args.num_envs} parallel '{env_id}' environments...")
    
    if args.curiosity:
        from env.curiosity import CuriosityWrapper
        print("Wrapping environments with Intrinsic Curiosity Module (ICM)...")
        game_env = make_vec_env(
            env_id, 
            n_envs=args.num_envs,
            wrapper_class=CuriosityWrapper,
            wrapper_kwargs={"eta": 0.1}
        )
    else:
        game_env = make_vec_env(env_id, n_envs=args.num_envs)
        
    # 3. Load checkpoint or initialize new model
    prefix = "ppo_element_shooter"
    
    if args.resume:
        if not os.path.exists(args.resume) and not args.resume.endswith(".zip"):
            checkpoint_path = args.resume + ".zip"
        else:
            checkpoint_path = args.resume
            
        print(f"Resuming training from checkpoint: {checkpoint_path}...")
        
        # Load tuned parameters to override during load
        tuned_params = load_best_hyperparams()
        custom_objects = {}
        if tuned_params:
            print("Overriding hyperparameters with new tuned parameters:")
            keys_to_override = ["learning_rate", "gamma", "batch_size", "n_steps", "ent_coef"]
                
            for key in keys_to_override:
                if key in tuned_params:
                    custom_objects[key] = tuned_params[key]
                    print(f"  {key}: {tuned_params[key]}")
            print("Note: Network architecture (net_arch) cannot be changed when resuming a checkpoint.")
            
        model = PPO.load(checkpoint_path, env=game_env, custom_objects=custom_objects)
        # Ensure tensorboard logging path is preserved
        model.tensorboard_log = args.tb_log
    else:
        # Load tuned parameters or use defaults
        tuned_params = load_best_hyperparams()
        
        learning_rate = tuned_params.get("learning_rate", 3e-4)
        gamma = tuned_params.get("gamma", 0.99)
        batch_size = tuned_params.get("batch_size", 256)
        net_arch_width = tuned_params.get("net_arch_width", 512)
        n_steps = tuned_params.get("n_steps", 2048)  # Steps to run per update
        ent_coef = tuned_params.get("ent_coef", 0.02) # Entropy coefficient for discrete exploration
        gae_lambda = tuned_params.get("gae_lambda", 0.95)
        clip_range = tuned_params.get("clip_range", 0.2)
        
        policy_kwargs = dict(
            net_arch=dict(pi=[net_arch_width, net_arch_width], vf=[net_arch_width, net_arch_width])
        )
        
        print("\n--- PPO Training Hyperparameters ---")
        print(f"Learning Rate:    {learning_rate:.2e}")
        print(f"Gamma:            {gamma:.4f}")
        print(f"N Steps:          {n_steps}")
        print(f"Batch Size:       {batch_size}")
        print(f"Entropy Coef:     {ent_coef}")
        print(f"GAE Lambda:       {gae_lambda}")
        print(f"Clip Range:       {clip_range}")
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
            gae_lambda=gae_lambda,
            clip_range=clip_range,
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
    print(f"Starting PPO training for {args.steps} steps...")
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
