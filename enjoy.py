import os
import argparse
import gymnasium as gym
import numpy as np
import pygame
import env  # Registers ElementShooter levels

def play_manual(env_id="DotShot-Level1-v0"):
    """Lets the user play the game manually using WASD and Mouse."""
    print("\n================ MANUAL PLAY MODE ================")
    print("Controls:")
    print("  Movement: W, A, S, D  or  Arrow Keys")
    print("  Aiming:   Move Mouse Cursor")
    print("  Weapons:  Press 1 (Water), 2 (Grass), 3 (Fire)")
    print("            Click Left Mouse Button to shoot selected weapon")
    print("==================================================\n")
    
    # Initialize environment
    game_env = gym.make(env_id, render_mode="human")
    obs, info = game_env.reset()
    
    selected_weapon = 0  # Default to Water
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # 1. Capture user keyboard/mouse input
            pygame.event.pump()
            
            # Check window exit
            for event in pygame.event.get(pygame.QUIT):
                running = False
                break
                
            # Handle weapon selection keys
            keys = pygame.key.get_pressed()
            if keys[pygame.K_1]: selected_weapon = 0
            elif keys[pygame.K_2]: selected_weapon = 1
            elif keys[pygame.K_3]: selected_weapon = 2
            
            # Map movement WASD / Arrows
            move_x = 0.0
            move_y = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:    move_y = -1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  move_y = 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:  move_x = -1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_x = 1.0
            
            # Convert movement values to discrete indices:
            # 0: Left/Up, 1: None, 2: Right/Down
            move_idx_x = int(move_x + 1.0)
            move_idx_y = int(move_y + 1.0)
                
            # Calculate aim vector from agent to mouse cursor
            mouse_x, mouse_y = pygame.mouse.get_pos()
            agent_x = game_env.unwrapped.agent_x[0] if isinstance(game_env.unwrapped.agent_x, list) else game_env.unwrapped.agent_x
            agent_y = game_env.unwrapped.agent_y[0] if isinstance(game_env.unwrapped.agent_y, list) else game_env.unwrapped.agent_y
            
            aim_x = mouse_x - agent_x
            aim_y = mouse_y - agent_y
            aim_len = np.sqrt(aim_x**2 + aim_y**2)
            
            if aim_len > 0:
                # Calculate angle in [0, 2pi]
                # -aim_y because grid y coordinates increase going downwards
                aim_angle = np.arctan2(-aim_y, aim_x)
                if aim_angle < 0:
                    aim_angle += 2.0 * np.pi
                # Map to discrete angle index 0..63
                aim_idx = int(round(aim_angle / (2.0 * np.pi / 64.0))) % 64
            else:
                aim_idx = 0  # Facing right (angle 0)
                
            # Check mouse click for shooting
            click = pygame.mouse.get_pressed()
            shoot_pressed = click[0] or keys[pygame.K_SPACE]
            
            # Map weapon choice to MultiDiscrete weapon action:
            # 0: None, 1..4: weapon index + 1
            weapon_idx = selected_weapon + 1 if shoot_pressed else 0
                
            # Construct Discrete Action (MultiDiscrete size 4)
            action = np.array([
                move_idx_x,
                move_idx_y,
                aim_idx,
                weapon_idx
            ], dtype=np.int64)
            
            # Pad action with idle actions for secondary agents if in multi-agent mode
            num_agents = getattr(game_env.unwrapped, "num_agents", 1)
            if num_agents > 1:
                extra_actions = [1, 1, 0, 0] * (num_agents - 1)
                action = np.concatenate([action, np.array(extra_actions, dtype=np.int64)])
            
            # 2. Step the simulator
            obs, reward, terminated, truncated, info = game_env.step(action)
            
            # Manually force the weapon selection highlight on HUD if we haven't shot recently
            if game_env.unwrapped.shoot_cooldown <= 0:
                game_env.unwrapped.last_fired_weapon = selected_weapon
                
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Survived: {info['steps_survived']} steps ({info['steps_survived']/60:.1f}s)")
                obs, info = game_env.reset()
                selected_weapon = 0
                
    except KeyboardInterrupt:
        print("\nExiting manual mode.")
    finally:
        game_env.close()

def play_agent(model_path, env_id="DotShot-Level1-v0"):
    """Run a trained PPO agent in the environment with visualization."""
    import torch
    
    # Check if this is a custom PyTorch MARL model (.pt) or an SB3 model (.zip)
    is_marl = model_path.endswith(".pt") or model_path.endswith(".pth") or (
        not model_path.endswith(".zip") and os.path.exists(model_path + ".pt")
    )
    
    if is_marl:
        if not model_path.endswith(".pt") and not model_path.endswith(".pth"):
            model_path += ".pt"
            
        print(f"Loading custom MARL policy from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu")
        algo = checkpoint["algo"]
        obs_dim = checkpoint["obs_dim"]
        action_dim = checkpoint["action_dim"]
        global_state_dim = checkpoint["global_state_dim"]
        
        from marl.pmat import MATPolicy, PMATPolicy, RoMATPolicy
        policy_mapping = {
            "mat": MATPolicy,
            "pmat": PMATPolicy,
            "romat": RoMATPolicy
        }
        
        policy = policy_class = policy_mapping[algo](
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim
        )
        policy.load_state_dict(checkpoint["state_dict"])
        policy.eval()
    else:
        from stable_baselines3 import PPO
        if not os.path.exists(model_path) and not model_path.endswith(".zip"):
            model_path = model_path + ".zip"
            
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train a model first with: python3 train.py")
            return
            
        print(f"Loading trained SB3 PPO agent from {model_path}...")
        model = PPO.load(model_path)
    
    # Initialize environment in human mode
    game_env = gym.make(env_id, render_mode="human")
    obs, info = game_env.reset()
    
    running = True
    episode_count = 0
    print("Running evaluation. Press Ctrl+C in terminal to stop.")
    
    try:
        while running:
            # Predict actions using the loaded model
            if is_marl:
                # Reshape observation to individual agent dimensions: [1, num_agents, obs_dim]
                obs_n = torch.tensor(obs, dtype=torch.float32).view(1, 2, obs_dim)
                with torch.no_grad():
                    actions_n, _ = policy.select_actions(obs_n, deterministic=True)
                    
                # Decode action tokens back to multi-discrete vectors and flatten
                from marl.train_marl import decode_actions
                action = decode_actions(actions_n, "cpu").numpy().flatten()
            else:
                action, _states = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = game_env.step(action)
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count} finished. Score: {info['score']}, Survived: {info['steps_survived']} steps ({info['steps_survived']/60:.1f}s)")
                obs, info = game_env.reset()
                
    except KeyboardInterrupt:
        print("\nExiting evaluation.")
    finally:
        game_env.close()

def main():
    parser = argparse.ArgumentParser(description="Play or evaluate Element Shooter")
    parser.add_argument("--model", type=str, default="./models/ppo_element_shooter_final", help="Path to trained model")
    parser.add_argument("--manual", action="store_true", help="Play the game manually instead of using an agent")
    parser.add_argument("--env-id", type=str, default="DotShot-Level1-v0", help="Gym env ID (DotShot-Level1-v0, Level2-v0, or Level3-v0)")
    args = parser.parse_args()
    
    if args.manual:
        play_manual(env_id=args.env_id)
    else:
        play_agent(args.model, env_id=args.env_id)

if __name__ == "__main__":
    main()
