from gymnasium.envs.registration import register
from env.game_env import ElementShooterEnv

# Register the continuous custom game environment for SAC
register(
    id="ElementShooter-v0",
    entry_point="env.game_env:ElementShooterEnv",
    max_episode_steps=3600,
    kwargs={"multidiscrete": False}
)

# Register the discrete custom game environment for PPO
register(
    id="ElementShooter-Discrete-v0",
    entry_point="env.game_env:ElementShooterEnv",
    max_episode_steps=3600,
    kwargs={"multidiscrete": True}
)

