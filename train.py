import os
import json
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import env  # Registers ElementShooter-v0
from env.utils import create_vec_env

# Optimize PyTorch CPU performance by limiting thread count (prevents core thrashing)
# Defaults to 1 for vector env efficiency, but customizable via CLI
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

def cosine_annealing_schedule(initial_value: float, min_value: float = 1e-6):
    """
    Cosine Annealing Learning Rate Schedule for Stable Baselines 3 PPO.
    Natively receives progress_remaining which goes from 1.0 (start) to 0.0 (end).
    """
    import numpy as np
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 down to 0.0
        cos_out = np.cos((1.0 - progress_remaining) * np.pi)
        return min_value + 0.5 * (initial_value - min_value) * (1.0 + cos_out)
    return func



def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Element Shooter environment")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--tb-log", type=str, default="./tb_log", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./models", help="Model saving directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint model (.zip) to resume training from")
    parser.add_argument("--curiosity", action="store_true", help="Enable Intrinsic Curiosity Module (ICM) exploration rewards")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments to run (default: 8)")
    parser.add_argument("--vec-env", type=str, default="dummy", choices=["dummy", "subproc"], help="Vector env type (dummy or subproc)")
    parser.add_argument("--torch-threads", type=int, default=1, help="PyTorch CPU thread count limit (default: 1)")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to run PyTorch models on (default: cpu)")
    parser.add_argument("--env-id", type=str, default="DotShot-Level1-v0", help="Gym env ID (DotShot-Level1-v0, Level2-v0, or Level3-v0)")
    args = parser.parse_args()
    
    # Configure PyTorch CPU thread count
    torch.set_num_threads(args.torch_threads)
    
    # 1. Setup folders
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 2. Load tuned parameters
    tuned_params = load_best_hyperparams()

    # 3. Create environment
    env_id = args.env_id
    print(f"Initializing {args.num_envs} parallel '{env_id}' environments ({args.vec_env})...")
    
    wrapper_class = None
    wrapper_kwargs = None
    if args.curiosity:
        from env.curiosity import CuriosityWrapper
        wrapper_class = CuriosityWrapper
        
        # Load tuned curiosity parameters or use default values
        eta = tuned_params.get("eta", 0.1)
        beta = tuned_params.get("beta", 0.2)
        lr_icm = tuned_params.get("lr_icm", 1e-4)
        R_min = tuned_params.get("R_min", -150.0)
        R_max = tuned_params.get("R_max", -40.0)
        
        print("Wrapping environments with Intrinsic Curiosity Module (ICM)...")
        print(f"  eta: {eta}, beta: {beta}, lr_icm: {lr_icm}, R_min: {R_min}, R_max: {R_max}")
        
        wrapper_kwargs = {
            "eta": eta,
            "beta": beta,
            "lr": lr_icm,
            "R_min": R_min,
            "R_max": R_max
        }
        
    game_env = create_vec_env(
        env_id, 
        num_envs=args.num_envs, 
        vec_env_type=args.vec_env, 
        wrapper_class=wrapper_class, 
        wrapper_kwargs=wrapper_kwargs
    )
        
    # 4. Load checkpoint or initialize new model
    prefix = "ppo_element_shooter"
    
    if args.resume:
        if not os.path.exists(args.resume) and not args.resume.endswith(".zip"):
            checkpoint_path = args.resume + ".zip"
        else:
            checkpoint_path = args.resume
            
        print(f"Resuming training from checkpoint: {checkpoint_path}...")
        
        custom_objects = {}
        if tuned_params:
            print("Overriding hyperparameters with new tuned parameters:")
            keys_to_override = ["gamma", "batch_size", "n_steps", "ent_coef"]
                
            for key in keys_to_override:
                if key in tuned_params:
                    custom_objects[key] = tuned_params[key]
                    print(f"  {key}: {tuned_params[key]}")
            
            # Apply learning rate schedule
            initial_lr = tuned_params.get("learning_rate", 3e-4)
            custom_objects["learning_rate"] = cosine_annealing_schedule(initial_lr)
            print(f"  learning_rate: Cosine Annealing (initial={initial_lr})")
            
            print("Note: Network architecture (net_arch) cannot be changed when resuming a checkpoint.")
            
        model = PPO.load(checkpoint_path, env=game_env, custom_objects=custom_objects, device=args.device)
        # Ensure tensorboard logging path is preserved
        model.tensorboard_log = args.tb_log
    else:
        initial_lr = tuned_params.get("learning_rate", 3e-4)
        learning_rate = cosine_annealing_schedule(initial_lr)
        
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
        print(f"Learning Rate:    Cosine Annealing (initial={initial_lr})")
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
            device=args.device,
            verbose=1
        )
    
    # 5. Checkpoints & Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=args.save_dir,
        name_prefix=prefix
    )
    
    callbacks = [checkpoint_callback]
    if args.curiosity:
        from env.curiosity import CuriosityCallback
        import numpy as np
        obs_dim = int(np.prod(game_env.observation_space.shape))
        action_dim = int(np.prod(game_env.action_space.shape))
        curiosity_callback = CuriosityCallback(
            obs_dim=obs_dim,
            action_dim=action_dim,
            eta=eta,
            beta=beta,
            lr=lr_icm,
            device=args.device
        )
        callbacks.append(curiosity_callback)
        print(f"Centralized Intrinsic Curiosity Module (ICM) Callback registered (running on {args.device}).")
    
    # 6. Learn
    print(f"Starting PPO training for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps, 
        callback=callbacks,
        reset_num_timesteps=False if args.resume else True
    )
    
    # 7. Save Final Model
    final_model_path = os.path.join(args.save_dir, f"{prefix}_final")
    model.save(final_model_path)
    print(f"Training finished! Final model saved to {final_model_path}")
    
    game_env.close()

if __name__ == "__main__":
    main()
