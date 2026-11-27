"""Rendering must not change the simulation (the renderer once drew from the engine's RNG)."""

import os

import numpy as np

from dotshot.envs.multi_agent import DotShotParallelEnv


def _play(render: bool) -> tuple:
    if render:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = DotShotParallelEnv(stage=4, render_mode="rgb_array" if render else None,
                             max_steps=300, env_overrides={"action_repeat": 1})
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(300):
        actions = {a: rng.integers(0, [3, 3, 32, 4]) for a in env.possible_agents}
        _, _, terms, truncs, infos = env.step(actions)
        if render:
            env.render()
        if any(terms.values()) or any(truncs.values()):
            break
    e = env.engine
    state = (e.step_count, e.telemetry["kills"], e.telemetry["shots_fired"], round(e.core.hp, 6),
             tuple(round(x.x, 6) for x in e.enemies))
    env.close()
    return state


def test_rendering_does_not_change_the_episode():
    assert _play(render=True) == _play(render=False)
