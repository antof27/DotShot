import os
import sys
import argparse
import torch

# Add project root to sys.path to resolve imports when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marl.pmat import MATPolicy, PMATPolicy, RoMATPolicy

def perform_marl_surgery(l2_checkpoint_path, l3_checkpoint_path):
    print(f"Loading custom Level 2 MARL checkpoint: {l2_checkpoint_path}...")
    checkpoint_l2 = torch.load(l2_checkpoint_path, map_location="cpu")
    
    algo = checkpoint_l2["algo"]
    obs_dim = checkpoint_l2["obs_dim"]
    action_dim = checkpoint_l2["action_dim"]
    
    print(f"Detected Level 2 Algorithm: {algo.upper()}")
    
    # Define Policy Mapping
    policy_mapping = {
        "mat": MATPolicy,
        "pmat": PMATPolicy,
        "romat": RoMATPolicy
    }
    
    print("\nInitializing target Level 3 MARL Policy...")
    # Initialize a clean Level 3 policy structure (global_state_dim = 140)
    policy_l3 = policy_mapping[algo](
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=obs_dim * 2  # 2 agents, total 140 features
    )
    
    sd_l2 = checkpoint_l2["state_dict"]
    sd_l3 = policy_l3.state_dict()
    
    print("\n--- Starting MARL Checkpoint Weight Surgery ---")
    copied = 0
    surgered = 0
    
    for key in sd_l3.keys():
        if key in sd_l2:
            if sd_l3[key].shape == sd_l2[key].shape:
                # Direct match (encoders, prioritizers, decoders, biases, subsequent critic layers)
                sd_l3[key].copy_(sd_l2[key])
                copied += 1
            else:
                # Target-critic weight mismatch (input dimension 140 vs 70)
                if "critic.net.0.weight" in key:
                    print(f"  Performing surgery on Centralized Critic input layer: '{key}'")
                    # Copy Level 2 weights to both Agent 0 (0-69) and Agent 1 (70-139) slots
                    sd_l3[key][:, :obs_dim].copy_(sd_l2[key])
                    sd_l3[key][:, obs_dim:].copy_(sd_l2[key])
                    surgered += 1
                else:
                    print(f"  Warning: Skipping mismatching layer '{key}': L2 shape {sd_l2[key].shape}, L3 shape {sd_l3[key].shape}")
        else:
            print(f"  Warning: Layer '{key}' not found in source checkpoint.")
            
    print(f"\nSurgery Complete! Copied {copied} layers directly. Adapted {surgered} layer.")
    
    # Save the migrated checkpoint dict
    checkpoint_l3 = {
        "algo": algo,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "global_state_dim": obs_dim * 2,
        "state_dict": sd_l3
    }
    
    print(f"Saving converted Level 3 MARL checkpoint to: {l3_checkpoint_path}...")
    torch.save(checkpoint_l3, l3_checkpoint_path)
    print("Migrated checkpoint successfully saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO weight transfer surgery for custom MARL models from Level 2 to Level 3")
    parser.add_argument("--source", type=str, default="./models_marl/pmat_final.pt", help="Path to Level 2 trained MARL checkpoint (.pt)")
    parser.add_argument("--target", type=str, default="./models_marl/pmat_level3_init.pt", help="Path to save the migrated Level 3 checkpoint (.pt)")
    args = parser.parse_args()
    
    if not os.path.exists(args.source) and not args.source.endswith(".pt"):
        args.source += ".pt"
        
    if not os.path.exists(args.source):
        print(f"Error: Source checkpoint not found at {args.source}")
        exit(1)
        
    perform_marl_surgery(args.source, args.target)
