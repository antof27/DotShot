import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

# Add project root to sys.path to resolve imports when running script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom environments and helpers
import env
from env.utils import create_vec_env
from marl.pmat import MATPolicy, PMATPolicy, RoMATPolicy

# --- Action Encoding / Decoding Helpers ---
# Combines MultiDiscrete [3, 3, 64, 4] into a single discrete action ID (0 to 2303)
# a * (3*64*4) + b * (64*4) + c * 4 + d
def encode_actions(actions_np):
    """
    actions_np: NumPy array of shape [..., 4]
    Returns: Integer array of shape [...]
    """
    a, b, c, d = actions_np[..., 0], actions_np[..., 1], actions_np[..., 2], actions_np[..., 3]
    return a * 768 + b * 256 + c * 4 + d

def decode_actions(flat_actions_torch, device):
    """
    flat_actions_torch: PyTorch tensor of shape [batch_size, num_agents] containing action IDs
    Returns: PyTorch tensor of shape [batch_size, num_agents, 4] matching MultiDiscrete structure
    """
    d = flat_actions_torch % 4
    c = (flat_actions_torch // 4) % 64
    b = (flat_actions_torch // 256) % 3
    a = (flat_actions_torch // 768) % 3
    return torch.stack([a, b, c, d], dim=-1)


# --- Rollout Buffer ---
class MARLBuffer:
    def __init__(self, buffer_size, num_envs, num_agents, obs_dim, state_dim, device):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        self.obs = torch.zeros(buffer_size, num_envs, num_agents, obs_dim, device=device)
        self.actions = torch.zeros(buffer_size, num_envs, num_agents, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(buffer_size, num_envs, device=device)
        self.rewards = torch.zeros(buffer_size, num_envs, device=device)
        self.values = torch.zeros(buffer_size, num_envs, device=device)
        self.dones = torch.zeros(buffer_size, num_envs, device=device)
        self.ptr = 0

    def insert(self, obs, actions, log_prob, reward, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.buffer_size

    def reset(self):
        self.ptr = 0


# --- Main Trainer ---
def train():
    parser = argparse.ArgumentParser(description="Train a custom MARL agent (MAT, PMAT, or RoMAT) on DotShot")
    parser.add_argument("--algo", type=str, default="pmat", choices=["mat", "pmat", "romat"], help="MARL algorithm to train")
    parser.add_argument("--steps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=2048, help="Rollout steps per environment")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO optimization epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO policy clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Critic loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm limit")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to run model on")
    parser.add_argument("--save-dir", type=str, default="./models_marl", help="Directory to save MARL policy checkpoints")
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["dummy", "subproc"], help="Vector environment type")
    parser.add_argument("--env-id", type=str, default="DotShot-Level3-v0", choices=["DotShot-Level1-v0", "DotShot-Level2-v0", "DotShot-Level3-v0"], help="Gym environment ID")
    parser.add_argument("--resume", type=str, default=None, help="Path to custom MARL checkpoint (.pt) to resume training from")
    args = parser.parse_args()

    # Create Save Directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Environment Setup
    env_id = args.env_id
    num_agents = 2 if env_id == "DotShot-Level3-v0" else 1
    obs_dim = 70
    action_dim = 2304  # 3 * 3 * 64 * 4 joint actions representation
    global_state_dim = num_agents * obs_dim

    print(f"Initializing {args.num_envs} parallel '{env_id}' environments ({args.vec_env})...")
    envs = create_vec_env(env_id, num_envs=args.num_envs, vec_env_type=args.vec_env)
    
    # 3. Model instantiation
    policy_mapping = {
        "mat": MATPolicy,
        "pmat": PMATPolicy,
        "romat": RoMATPolicy
    }
    
    policy_class = policy_mapping[args.algo]
    print(f"Instantiating {args.algo.upper()} Policy network...")
    policy = policy_class(
        obs_dim=obs_dim, 
        action_dim=action_dim, 
        global_state_dim=global_state_dim
    ).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    
    # Load checkpoint if resuming
    if args.resume:
        print(f"Loading checkpoint state from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        policy.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            # Only load optimizer state if model is resumed on the same architecture/level
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Optimizer state restored successfully.")
            except Exception as e:
                print(f"  Warning: Could not restore optimizer state ({e}). Re-initializing optimizer.")
                
    buffer = MARLBuffer(args.n_steps, args.num_envs, num_agents, obs_dim, global_state_dim, device)

    # 4. Starting Environment Rollouts
    obs_flat = envs.reset()  # Shape: [num_envs, 140]
    
    global_step = 0
    updates_count = 0
    
    print("\n================ STARTING MARL PPO TRAINING ================")
    
    while global_step < args.steps:
        policy.eval()
        buffer.reset()
        
        # Collect rollouts
        for step in range(args.n_steps):
            # Reshape observation to individual agent dimensions: [num_envs, 2, 70]
            obs_n = torch.tensor(obs_flat, dtype=torch.float32, device=device).view(args.num_envs, num_agents, obs_dim)
            global_state = obs_n.view(args.num_envs, -1)  # Centralized state (concatenated observations)
            
            # Select actions autoregressively
            with torch.no_grad():
                actions_n, _ = policy.select_actions(obs_n, deterministic=False) # [num_envs, num_agents]
                
                # Evaluate log probs and value estimates under current policy
                log_prob, _, value = policy.evaluate_actions(obs_n, actions_n, global_state)
                
            # Decode action tokens back to multi-discrete vectors to step environment
            multi_discrete_actions = decode_actions(actions_n, device).cpu().numpy()  # [num_envs, num_agents, 4]
            env_actions = multi_discrete_actions.reshape(args.num_envs, -1)  # Flatten: [num_envs, 8]
            
            # Step environments
            next_obs_flat, rewards, dones, infos = envs.step(env_actions)
            
            # Insert into buffer
            buffer.insert(
                obs_n,
                actions_n,
                log_prob,
                torch.tensor(rewards, dtype=torch.float32, device=device),
                value.squeeze(-1),
                torch.tensor(dones, dtype=torch.float32, device=device)
            )
            
            obs_flat = next_obs_flat
            global_step += args.num_envs

        # --- Compute GAE Returns and Advantages ---
        policy.eval()
        with torch.no_grad():
            next_obs_n = torch.tensor(obs_flat, dtype=torch.float32, device=device).view(args.num_envs, num_agents, obs_dim)
            next_global_state = next_obs_n.view(args.num_envs, -1)
            next_value = policy.critic(next_global_state).squeeze(-1)
            
            # GAE computation
            advantages = torch.zeros_like(buffer.rewards)
            returns = torch.zeros_like(buffer.rewards)
            last_gae_lam = 0.0
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps - 1:
                    next_non_terminal = 1.0 - torch.tensor(dones, dtype=torch.float32, device=device)
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - buffer.dones[t + 1]
                    next_values = buffer.values[t + 1]
                
                delta = buffer.rewards[t] + args.gamma * next_values * next_non_terminal - buffer.values[t]
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
                
            returns = advantages + buffer.values

        # --- Flatten Rollout Data for PPO Updates ---
        b_obs = buffer.obs.reshape(-1, num_agents, obs_dim)
        b_actions = buffer.actions.reshape(-1, num_agents)
        b_log_probs = buffer.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer.values.reshape(-1)
        b_global_states = b_obs.view(b_obs.shape[0], -1)

        # Standardize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # --- PPO Optimization Epochs ---
        policy.train()
        
        batch_size_total = args.n_steps * args.num_envs
        indices = np.arange(batch_size_total)
        
        loss_epoch = 0.0
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_epoch = 0.0
        
        for epoch in range(args.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size_total, args.batch_size):
                end = start + args.batch_size
                mb_indices = indices[start:end]
                
                mb_obs = b_obs[mb_indices]
                mb_actions = b_actions[mb_indices]
                mb_log_probs = b_log_probs[mb_indices]
                mb_advantages = b_advantages[mb_indices]
                mb_returns = b_returns[mb_indices]
                mb_global_states = b_global_states[mb_indices]
                
                # Evaluate under updated policy weights
                new_log_prob, entropy, new_value = policy.evaluate_actions(mb_obs, mb_actions, mb_global_states)
                
                # Calculate surrogate policy loss
                ratio = torch.exp(new_log_prob - mb_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (MSE between critic values and target GAE returns)
                value_loss = 0.5 * F.mse_loss(new_value.squeeze(-1), mb_returns)
                
                # Total joint PPO loss
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                
                loss_epoch += loss.item()
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.item()

        updates_count += 1
        
        # Calculate training statistics metrics
        div = args.n_epochs * (batch_size_total // args.batch_size)
        avg_loss = loss_epoch / div
        avg_policy_loss = policy_loss_epoch / div
        avg_value_loss = value_loss_epoch / div
        avg_entropy = entropy_epoch / div
        
        # Print progress summary
        mean_reward = buffer.rewards.mean().item()
        print(f"Update {updates_count:03d} | Step: {global_step:07d}/{args.steps:07d} | "
              f"Mean Reward: {mean_reward:6.2f} | Loss: {avg_loss:5.3f} | "
              f"Policy Loss: {avg_policy_loss:5.3f} | Value Loss: {avg_value_loss:5.3f} | "
              f"Entropy: {avg_entropy:5.3f}")
        
        # Save checkpoints
        if updates_count % 10 == 0 or global_step >= args.steps:
            chk_path = os.path.join(args.save_dir, f"{args.algo}_latest.pt")
            torch.save({
                "algo": args.algo,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "global_state_dim": global_state_dim,
                "state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, chk_path)
            print(f"  Checkpoint saved: {chk_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, f"{args.algo}_final.pt")
    torch.save({
        "algo": args.algo,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "global_state_dim": global_state_dim,
        "state_dict": policy.state_dict()
    }, final_path)
    print(f"Training successfully completed! Final model saved to: {final_path}")

    envs.close()

if __name__ == "__main__":
    train()
