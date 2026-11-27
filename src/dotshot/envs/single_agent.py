"""Gymnasium single-agent environment wrapper for DotShot Stages 1 and 2."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dotshot.core.engine import DotShotEngine, OBS_DIM


class DotShotSingleEnv(gym.Env):
    """Single-agent Gymnasium environment for DotShot Stage 1 (Fundamentals) and Stage 2 (Elemental Mastery)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        stage: int = 1,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        env_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError(f"DotShotSingleEnv supports stage 1 or 2, got {stage}.")
        self.stage = stage
        self.render_mode = render_mode
        self.engine = DotShotEngine(stage=stage, max_steps=max_steps, env_overrides=env_overrides)

        self.action_space = spaces.MultiDiscrete([3, 3, self.engine.cfg.aim_bins, 4])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._renderer = None
        self.overlay_info: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        derived_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.engine.reset(seed=derived_seed)
        obs = self.engine.get_local_obs(0)
        return obs, {"step": 0, "stage": self.stage}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        rewards, terminated, truncated, info = self.engine.step({0: action})
        obs = self.engine.get_local_obs(0)
        reward = float(rewards[0])
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from dotshot.render.pygame_renderer import DotShotPygameRenderer
            self._renderer = DotShotPygameRenderer(
                width=self.engine.cfg.width,
                height=self.engine.cfg.height,
                render_mode=self.render_mode,
            )
        return self._renderer.render(self.engine, overlay_info=self.overlay_info)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
