import gymnasium as gym
import numpy as np
import env  # Registers ElementShooter-v0 and ElementShooter-Discrete-v0

def test_env(env_id):
    print(f"\n================ TESTING {env_id} ================")
    try:
        env_instance = gym.make(env_id)
    except Exception as e:
        print(f"Error making environment: {e}")
        return False
        
    print(f"Observation space: {env_instance.observation_space}")
    print(f"Action space: {env_instance.action_space}")
    
    # Reset
    obs, info = env_instance.reset()
    print("Environment reset successful.")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Step through 100 times with random actions
    print("Stepping with random actions for 100 steps...")
    step_count = 0
    total_reward = 0
    
    while step_count < 100:
        action = env_instance.action_space.sample()
        obs, reward, terminated, truncated, info = env_instance.step(action)
        total_reward += reward
        step_count += 1
        
        if terminated or truncated:
            obs, info = env_instance.reset()
            
    print(f"Completed {step_count} steps successfully!")
    print(f"Total reward accumulated: {total_reward:.2f}")
    print(f"Final observation stats: Mean={obs.mean():.4f}, Std={obs.std():.4f}")
    env_instance.close()
    print(f"Sanity check for {env_id} completed successfully!")
    return True

def main():
    ok1 = test_env("ElementShooter-v0")
    ok2 = test_env("ElementShooter-Discrete-v0")
    if ok1 and ok2:
        print("\nAll sanity checks completed successfully!")

if __name__ == "__main__":
    main()
