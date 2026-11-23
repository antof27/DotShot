"""HARL-compatible adapter for DotShot Stages 3 and 4."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from dotshot.envs.multi_agent import DotShotParallelEnv


class DotShotHARLEnv:
    """Expose DotShotParallelEnv through the vector-env contract used by HARL.

    HARL expects one flat batch dimension per agent, a shared global state for
    every agent, and per-agent ``MultiDiscrete`` spaces. DotShot already provides
    all of that; this class only reshapes the PettingZoo dictionaries into lists
    and adds the truncation flag HARL needs for correct bootstrapping.
    """

    def __init__(
        self,
        stage: int = 3,
        partial_obs: bool = True,
        reward_alpha: float = 0.5,
        max_steps: Optional[int] = None,
        seed: int = 0,
    ):
        if stage not in (3, 4):
            raise ValueError(f"DotShotHARLEnv supports stage 3 or 4, got {stage}.")

        self.env = DotShotParallelEnv(
            stage=stage,
            partial_obs=partial_obs,
            reward_alpha=reward_alpha,
            max_steps=max_steps,
        )
        self.n_agents: int = len(self.env.possible_agents)
        self.observation_space: List[spaces.Box] = [
            self.env.observation_space(agent) for agent in self.env.possible_agents
        ]
        state_dim = int(self.env.state_space().shape[0])
        share_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.share_observation_space: List[spaces.Box] = [
            share_space for _ in range(self.n_agents)
        ]
        self.action_space: List[spaces.MultiDiscrete] = [
            self.env.action_space(agent) for agent in self.env.possible_agents
        ]
        self.seed(seed)

    def seed(self, seed: int) -> None:
        """Reseed the environment without returning observations."""
        self.env.reset(seed=int(seed))

    def _obs_list(self, obs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        return [
            np.asarray(obs[agent], dtype=np.float32)
            for agent in self.env.possible_agents
        ]

    def _share_obs_list(self) -> List[np.ndarray]:
        state = np.asarray(self.env.state(), dtype=np.float32)
        return [state.copy() for _ in self.env.possible_agents]

    def reset(self) -> Tuple[List[np.ndarray], List[np.ndarray], None]:
        obs, _ = self.env.reset()
        return self._obs_list(obs), self._share_obs_list(), None

    def step(
        self, actions: np.ndarray
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[List[float]],
        np.ndarray,
        List[Dict[str, Any]],
        None,
    ]:
        agents = self.env.possible_agents
        action_dict = {
            agent: np.rint(np.asarray(actions[idx])).astype(np.int64)
            for idx, agent in enumerate(agents)
        }
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

        bad_transition = bool(any(truncations.values()) and not any(terminations.values()))
        dones = np.array(
            [
                bool(terminations.get(agent, True) or truncations.get(agent, False))
                for agent in agents
            ],
            dtype=bool,
        )
        infos_list = [
            {
                "bad_transition": bad_transition,
                "episode_metrics": dict(infos.get(agent, {})),
            }
            for agent in agents
        ]
        rewards_list = [[float(rewards.get(agent, 0.0))] for agent in agents]

        return (
            self._obs_list(obs),
            self._share_obs_list(),
            rewards_list,
            dones,
            infos_list,
            None,
        )

    def close(self) -> None:
        self.env.close()
