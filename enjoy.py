import os
import argparse
import gymnasium as gym
import numpy as np
import pygame
import env  # Registers ElementShooter-v0

def play_manual():
    """Lets the user play the game manually using WASD and Mouse."""
    print("\n================ MANUAL PLAY MODE ================")
    print("Controls:")
    print("  Movement: W, A, S, D  or  Arrow Keys")
    print("  Aiming:   Move Mouse Cursor")
    print("  Weapons:  Press 1 (Water), 2 (Grass), 3 (Fire), 4 (Wind)")
    print("            Click Left Mouse Button to shoot selected weapon")
    print("==================================================\n")
    
    # Initialize environment
    game_env = gym.make("ElementShooter-v0", render_mode="human")
    obs, info = game_env.reset()
    
    selected_weapon = 0  # Default to Water
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # 1. Capture user keyboard/mouse input
            # We must pump pygame events
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
            elif keys[pygame.K_4]: selected_weapon = 3
            
            # Map movement WASD / Arrows
            move_x = 0.0
            move_y = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:    move_y = -1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  move_y = 1.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:  move_x = -1.0
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_x = 1.0
            
            # Normalize movement vector if diagonal
            move_len = np.sqrt(move_x**2 + move_y**2)
            if move_len > 0:
                move_x /= move_len
                move_y /= move_len
                
            # Calculate aim vector from agent to mouse cursor
            mouse_x, mouse_y = pygame.mouse.get_pos()
            agent_x = game_env.unwrapped.agent_x
            agent_y = game_env.unwrapped.agent_y
            
            aim_x = mouse_x - agent_x
            aim_y = mouse_y - agent_y
            aim_len = np.sqrt(aim_x**2 + aim_y**2)
            if aim_len > 0:
                aim_x /= aim_len
                aim_y /= aim_len
            else:
                aim_x, aim_y = 1.0, 0.0
                
            # Check mouse click for shooting
            click = pygame.mouse.get_pressed()
            shoot_trigger = 1.0 if (click[0] or keys[pygame.K_SPACE]) else -1.0
            
            # Map selected_weapon (0, 1, 2, 3) to weapon_select in [-1, 1]
            weapon_select = -0.75 + selected_weapon * 0.5
                
            # Construct Action Array for continuous environment (size 6)
            # action: [move_x, move_y, aim_x, aim_y, shoot_trigger, weapon_select]
            action = np.array([
                move_x, move_y,
                aim_x, aim_y,
                shoot_trigger,
                weapon_select
            ], dtype=np.float32)
            
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

def play_agent(model_path, algo="sac"):
    """Run a trained agent in the environment with visualization."""
    
    if algo == "sac":
        from stable_baselines3 import SAC as AlgoClass
    elif algo == "ppo":
        from stable_baselines3 import PPO as AlgoClass
    else:
        print(f"Error: Unknown algorithm '{algo}'. Use 'sac' or 'ppo'.")
        return
    
    if not os.path.exists(model_path) and not model_path.endswith(".zip"):
        model_path = model_path + ".zip"
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train a model first with: python3 train.py")
        return
        
    print(f"Loading trained {algo.upper()} agent from {model_path}...")
    model = AlgoClass.load(model_path)
    
    # Initialize environment in human mode
    env_id = "ElementShooter-Discrete-v0" if algo == "ppo" else "ElementShooter-v0"
    game_env = gym.make(env_id, render_mode="human")
    obs, info = game_env.reset()
    
    running = True
    episode_count = 0
    print("Running evaluation. Press Ctrl+C in terminal to stop.")
    
    try:
        while running:
            # Predict actions using model
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
    parser.add_argument("--model", type=str, default="./models/sac_element_shooter_final", help="Path to trained model")
    parser.add_argument("--algo", type=str, default=None, choices=["sac", "ppo"], help="Algorithm used for the model (sac or ppo). Auto-detected if not specified.")
    parser.add_argument("--manual", action="store_true", help="Play the game manually instead of using an agent")
    args = parser.parse_args()
    
    if args.manual:
        play_manual()
    else:
        # Auto-detect algorithm based on model name
        algo = args.algo
        if algo is None:
            if "ppo" in args.model.lower():
                algo = "ppo"
            elif "sac" in args.model.lower():
                algo = "sac"
            else:
                algo = "sac"  # Fallback default
        
        play_agent(args.model, algo=algo)

if __name__ == "__main__":
    main()
