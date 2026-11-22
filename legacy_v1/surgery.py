import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
import env  # Register DotShot environments

def perform_surgery(l2_path, l3_path):
    print(f"Loading source Level 2 model: {l2_path}...")
    model_l2 = PPO.load(l2_path, device="cpu")
    
    print("Creating target Level 3 environment...")
    l3_env = gym.make("DotShot-Level3-v0")
    
    print("Initializing fresh Level 3 PPO model structure...")
    # Extract policy network configurations matching source to keep architecture identical
    net_arch_width = model_l2.policy.net_arch["pi"][0] if "pi" in model_l2.policy.net_arch else 512
    policy_kwargs = dict(
        net_arch=dict(pi=[net_arch_width, net_arch_width], vf=[net_arch_width, net_arch_width])
    )
    
    model_l3 = PPO(
        "MlpPolicy",
        l3_env,
        policy_kwargs=policy_kwargs,
        device="cpu"
    )
    
    sd_l2 = model_l2.policy.state_dict()
    sd_l3 = model_l3.policy.state_dict()
    
    print("\n--- Starting Network Weight Surgery ---")
    copied_keys = 0
    surgeried_keys = 0
    
    for key in sd_l3.keys():
        if key in sd_l2:
            if sd_l3[key].shape == sd_l2[key].shape:
                # Direct copying for layers with matching dimensions (hidden layers, biases, value heads)
                sd_l3[key].copy_(sd_l2[key])
                copied_keys += 1
            else:
                # Input dimension mismatch (140 features for Level 3 vs 70 features for Level 2)
                if "mlp_extractor" in key and "weight" in key and sd_l3[key].shape[1] == 140:
                    print(f"  Performing surgery on first linear layer: '{key}'")
                    # Copy Level 2 weights to both Agent 0 (0-69) and Agent 1 (70-139) input slots
                    sd_l3[key][:, :70].copy_(sd_l2[key])
                    sd_l3[key][:, 70:].copy_(sd_l2[key])
                    surgeried_keys += 1
        else:
            # Action head mismatch: Level 3 has 8 action head components (2 agents) vs 4 for Level 2
            if "action_net" in key:
                parts = key.split(".")
                head_idx = int(parts[1])
                weight_or_bias = parts[2]
                
                # Map back to Level 2's action head components
                source_head_idx = head_idx % 4
                source_key = f"action_net.{source_head_idx}.{weight_or_bias}"
                
                print(f"  Mapping Action Head component: {key} <- {source_key}")
                sd_l3[key].copy_(sd_l2[source_key])
                surgeried_keys += 1

    # Load modified state dict into the Level 3 model
    model_l3.policy.load_state_dict(sd_l3)
    
    print(f"\nSurgery Complete! Copied {copied_keys} keys directly. Re-mapped/adjusted {surgeried_keys} keys.")
    print(f"Saving new Level 3 model to: {l3_path}...")
    model_l3.save(l3_path)
    print("Successfully saved model! You can now resume training on Level 3 using this model.")
    
    l3_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO weight transfer surgery from Level 2 (1 agent) to Level 3 (2 agents)")
    parser.add_argument("--source", type=str, default="./models/ppo_element_shooter_final.zip", help="Path to Level 2 trained PPO model (.zip)")
    parser.add_argument("--target", type=str, default="./models/ppo_element_shooter_level3_init.zip", help="Path to save the migrated Level 3 model (.zip)")
    args = parser.parse_args()
    
    if not os.path.exists(args.source) and not args.source.endswith(".zip"):
        args.source += ".zip"
        
    if not os.path.exists(args.source):
        print(f"Error: Source model not found at {args.source}")
        exit(1)
        
    perform_surgery(args.source, args.target)
