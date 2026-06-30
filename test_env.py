import gymnasium as gym
import numpy as np
import env  # Registers DotShot levels

def test_env(env_id, expected_obs_size=65):
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
    assert obs.shape[0] == expected_obs_size, f"Expected obs size {expected_obs_size}, got {obs.shape[0]}"
    print(f"Curriculum level: {info.get('curriculum_level', 'N/A')}")
    print(f"Initial info: {info}")
    
    # Step through 200 times with random actions
    print("Stepping with random actions for 200 steps...")
    step_count = 0
    total_reward = 0
    episodes = 0
    
    while step_count < 200:
        action = env_instance.action_space.sample()
        obs, reward, terminated, truncated, info = env_instance.step(action)
        total_reward += reward
        step_count += 1
        
        assert obs.shape[0] == expected_obs_size, f"Obs size mismatch at step {step_count}"
        assert not np.any(np.isnan(obs)), f"NaN in obs at step {step_count}"
        
        if terminated or truncated:
            episodes += 1
            obs, info = env_instance.reset()
            
    print(f"Completed {step_count} steps, {episodes} episode resets.")
    print(f"Total reward: {total_reward:.2f}, Avg reward/step: {total_reward/step_count:.4f}")
    print(f"Obs stats: min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}")
    env_instance.close()
    print(f"✓ {env_id} passed!")
    return True

def main():
    results = []
    for env_id in ["DotShot-Level1-v0", "DotShot-Level2-v0"]:
        ok = test_env(env_id, expected_obs_size=70)
        results.append((env_id, ok))
        
    ok_v2 = test_env("DotShot-Level3-v0", expected_obs_size=140)
    results.append(("DotShot-Level3-v0", ok_v2))
    
    print("\n================ SUMMARY ================")
    all_ok = True
    for env_id, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {env_id}")
        if not ok:
            all_ok = False
    
    if all_ok:
        print("\nAll curriculum levels passed!")
    else:
        print("\nSome tests FAILED!")
        exit(1)

if __name__ == "__main__":
    main()
