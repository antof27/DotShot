from gymnasium.envs.registration import register
from env.game_env import ElementShooterEnv

# Curriculum Level 1: Easy — Fire enemies only, max 2
register(
    id="ElementShooter-v0",
    entry_point="env.game_env:ElementShooterEnv",
    max_episode_steps=3600,
    kwargs={"curriculum_level": 1},
)

# Curriculum Level 2: Medium — Water + Fire enemies, max 3
register(
    id="ElementShooter-v1",
    entry_point="env.game_env:ElementShooterEnv",
    max_episode_steps=3600,
    kwargs={"curriculum_level": 2},
)

# Curriculum Level 3: Full game — all 4 types, max 5
register(
    id="ElementShooter-v2",
    entry_point="env.game_env:ElementShooterEnv",
    max_episode_steps=3600,
    kwargs={"curriculum_level": 3},
)
