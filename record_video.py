import os
import argparse
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import env

def main():
    parser = argparse.ArgumentParser(description="Record a video of a trained PPO agent playing Element Shooter")
    parser.add_argument("--model", type=str, default="./models/ppo_element_shooter_final", help="Path to trained PPO model (.zip)")
    parser.add_argument("--env-id", type=str, default="ElementShooter-v0", help="Gym env ID (ElementShooter-v0, v1, or v2)")
    parser.add_argument("--output", type=str, default="agent_gameplay.mp4", help="Output MP4 video path")
    parser.add_argument("--steps", type=int, default=3600, help="Maximum steps to record (default: 3600)")
    parser.add_argument("--fps", type=int, default=60, help="Video frame rate (default: 60)")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions (default: True)")
    parser.add_argument("--no-deterministic", action="store_false", dest="deterministic", help="Use stochastic actions")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"], help="Device to run PPO model inference on (default: cpu)")
    args = parser.parse_args()

    model_path = args.model
    if not model_path.endswith(".zip") and not os.path.exists(model_path):
        model_path += ".zip"

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading PPO model from {model_path} on device {args.device}...")
    model = PPO.load(model_path, device=args.device)

    print(f"Initializing {args.env_id} with rgb_array render mode...")
    game_env = gym.make(args.env_id, render_mode="rgb_array")
    
    obs, info = game_env.reset()
    
    width, height = game_env.unwrapped.width, game_env.unwrapped.height
    print(f"Video resolution: {width}x{height} @ {args.fps} FPS")
    
    # We use mp4v codec for standard .mp4 recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
    
    print(f"Recording agent gameplay. Saving to {args.output}...")
    
    steps = 0
    episode_reward = 0.0
    
    try:
        while True:
            # Render frame
            frame = game_env.render()
            if frame is None:
                print("Warning: Render frame is None")
                break
                
            # Pygame surfarray transposes width and height
            # Transpose frame from (width, height, 3) to (height, width, 3)
            frame = np.transpose(frame, (1, 0, 2))
            
            # Convert RGB (Pygame default) to BGR (OpenCV default)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video_writer.write(frame_bgr)
            
            # Predict action from observation
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Step the simulator
            obs, reward, terminated, truncated, info = game_env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated or steps >= args.steps:
                print(f"Episode finished!")
                print(f"  - Total Steps: {steps}")
                print(f"  - Total Reward: {episode_reward:.2f}")
                print(f"  - Final Score: {info.get('score', 0)}")
                break
                
    finally:
        video_writer.release()
        game_env.close()
        print(f"Video recording complete. Saved file: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
