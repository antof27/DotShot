"""PettingZoo ParallelEnv multi-agent Dec-POMDP wrapper for DotShot Stages 3 and 4."""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from dotshot.core.engine import DotShotEngine, OBS_DIM


class DotShotParallelEnv(ParallelEnv):
    """Multi-agent PettingZoo ParallelEnv for DotShot Stage 3 (Homogeneous Co-op) and Stage 4 (Heterogeneous Squad)."""

    metadata = {"render_modes": ["human", "rgb_array"], "name": "dotshot_parallel_v2", "render_fps": 60}

    def __init__(
        self,
        stage: int = 3,
        render_mode: Optional[str] = None,
        partial_obs: Optional[bool] = None,
        reward_alpha: Optional[float] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        if stage not in (3, 4):
            raise ValueError(f"DotShotParallelEnv supports stage 3 or 4, got {stage}.")
        self.stage = stage
        self.render_mode = render_mode
        self.engine = DotShotEngine(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
        )

        self.possible_agents: List[str] = [f"agent_{i}" for i in range(self.engine.cfg.num_agents)]
        self.agents: List[str] = list(self.possible_agents)
        self._agent_idx_map: Dict[str, int] = {name: i for i, name in enumerate(self.possible_agents)}
        self._np_random = np.random.default_rng(0)

        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self._act_space = spaces.MultiDiscrete([3, 3, self.engine.cfg.aim_bins, 4])
        self._state_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.engine.state_dim,), dtype=np.float32
        )

        self._renderer = None
        self.overlay_info: Optional[Dict[str, Any]] = None

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space

    def state_space(self) -> spaces.Space:
        return self._state_space

    def state(self) -> np.ndarray:
        """Return privileged global state s^t for centralized critics and mixers."""
        return self.engine.get_global_state()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        derived_seed = int(self._np_random.integers(0, 2**31 - 1))
        self.engine.reset(seed=derived_seed)
        self.agents = list(self.possible_agents)

        obs = {name: self.engine.get_local_obs(idx) for idx, name in enumerate(self.agents)}
        infos = {name: {"stage": self.stage, "role": self.engine.agents[idx].role_name} for idx, name in enumerate(self.agents)}
        return obs, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if not self.agents:
            return {}, {}, {}, {}, {}

        idx_actions: Dict[int, np.ndarray] = {}
        for name in self.agents:
            idx = self._agent_idx_map[name]
            idx_actions[idx] = actions.get(name, np.array([1, 1, 0, 0], dtype=np.int64))

        rewards_idx, terminated, truncated, shared_info = self.engine.step(idx_actions)

        obs: Dict[str, np.ndarray] = {}
        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        for name in self.agents:
            idx = self._agent_idx_map[name]
            obs[name] = self.engine.get_local_obs(idx)
            rewards[name] = float(rewards_idx.get(idx, 0.0))
            terminations[name] = bool(terminated)
            truncations[name] = bool(truncated)
            infos[name] = {
                **shared_info,
                "agent_alive": self.engine.agents[idx].alive,
                "role": self.engine.agents[idx].role_name,
            }

        if terminated or truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

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
