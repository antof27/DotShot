"""Vectorized PettingZoo ParallelEnv runner for Windows and Linux."""

import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dotshot.envs.multi_agent import DotShotParallelEnv


def _marl_worker(remote, parent_remote, stage: int, partial_obs: Optional[bool], reward_alpha: Optional[float], max_steps: Optional[int], seed: int):
    parent_remote.close()
    env = DotShotParallelEnv(stage=stage, partial_obs=partial_obs, reward_alpha=reward_alpha, max_steps=max_steps)
    agent_names = list(env.possible_agents)
    obs, _ = env.reset(seed=seed)

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = {name: data[i] for i, name in enumerate(agent_names)}
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            done_all = len(env.agents) == 0
            first_info = next(iter(infos.values())) if infos else {}
            if done_all:
                terminal_info = dict(first_info)
                next_obs, _ = env.reset()
                first_info = {**terminal_info, "episode_done": True}
            else:
                first_info = {**first_info, "episode_done": False}

            obs_arr = np.stack([next_obs[a] for a in agent_names], axis=0)
            rew_arr = np.array([rewards.get(a, 0.0) for a in agent_names], dtype=np.float32)
            done_arr = np.array([1.0 if done_all else 0.0 for _ in agent_names], dtype=np.float32)
            active_arr = np.array([1.0 if env.engine.agents[i].alive else 0.0 for i in range(len(agent_names))], dtype=np.float32)
            state_arr = env.state()
            remote.send((obs_arr, state_arr, rew_arr, done_arr, active_arr, first_info))
        elif cmd == "reset":
            obs, _ = env.reset(seed=data)
            obs_arr = np.stack([obs[a] for a in agent_names], axis=0)
            state_arr = env.state()
            active_arr = np.ones(len(agent_names), dtype=np.float32)
            remote.send((obs_arr, state_arr, active_arr))
        elif cmd == "close":
            env.close()
            remote.close()
            break


class VecMARLEnv:
    """Vectorized wrapper around DotShotParallelEnv with synchronous Dummy or Subproc stepping."""

    def __init__(
        self,
        stage: int,
        num_envs: int,
        seed: int = 0,
        vec_type: str = "subproc",
        partial_obs: Optional[bool] = None,
        reward_alpha: Optional[float] = None,
        max_steps: Optional[int] = None,
    ):
        self.stage = stage
        self.num_envs = num_envs
        self.vec_type = vec_type if num_envs > 1 else "dummy"

        probe = DotShotParallelEnv(stage=stage, partial_obs=partial_obs, reward_alpha=reward_alpha, max_steps=max_steps)
        self.agent_names = list(probe.possible_agents)
        self.num_agents = len(self.agent_names)
        self.obs_dim = probe.observation_space(self.agent_names[0]).shape[0]
        self.state_dim = probe.state_space().shape[0]
        self.max_steps = probe.engine.cfg.max_steps
        probe.close()

        if self.vec_type == "subproc":
            ctx = mp.get_context("spawn")
            self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
            self.ps = []
            for rank, (work_r, rem) in enumerate(zip(self.work_remotes, self.remotes)):
                p = ctx.Process(
                    target=_marl_worker,
                    args=(work_r, rem, stage, partial_obs, reward_alpha, max_steps, seed + rank * 1000),
                    daemon=True,
                )
                p.start()
                work_r.close()
                self.ps.append(p)
        else:
            self.envs = [
                DotShotParallelEnv(stage=stage, partial_obs=partial_obs, reward_alpha=reward_alpha, max_steps=max_steps)
                for _ in range(num_envs)
            ]
            for rank, e in enumerate(self.envs):
                e.reset(seed=seed + rank * 1000)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.vec_type == "subproc":
            for i, rem in enumerate(self.remotes):
                rem.send(("reset", None if seed is None else seed + i * 1000))
            results = [rem.recv() for rem in self.remotes]
            obs = np.stack([r[0] for r in results], axis=0)
            states = np.stack([r[1] for r in results], axis=0)
            actives = np.stack([r[2] for r in results], axis=0)
            return obs, states, actives

        obs_list, state_list, act_list = [], [], []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=None if seed is None else seed + i * 1000)
            obs_list.append(np.stack([o[a] for a in self.agent_names], axis=0))
            state_list.append(e.state())
            act_list.append(np.ones(self.num_agents, dtype=np.float32))
        return np.stack(obs_list, axis=0), np.stack(state_list, axis=0), np.stack(act_list, axis=0)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self.vec_type == "subproc":
            for i, rem in enumerate(self.remotes):
                rem.send(("step", actions[i]))
            results = [rem.recv() for rem in self.remotes]
            obs = np.stack([r[0] for r in results], axis=0)
            states = np.stack([r[1] for r in results], axis=0)
            rewards = np.stack([r[2] for r in results], axis=0)
            dones = np.stack([r[3] for r in results], axis=0)
            actives = np.stack([r[4] for r in results], axis=0)
            infos = [r[5] for r in results]
            return obs, states, rewards, dones, actives, infos

        obs_l, st_l, rew_l, done_l, act_l, info_l = [], [], [], [], [], []
        for i, e in enumerate(self.envs):
            act_dict = {name: actions[i, j] for j, name in enumerate(self.agent_names)}
            no, r, _, _, inf = e.step(act_dict)
            done_all = len(e.agents) == 0
            first_info = dict(next(iter(inf.values()))) if inf else {}
            if done_all:
                first_info["episode_done"] = True
                no, _ = e.reset()
            else:
                first_info["episode_done"] = False
            obs_l.append(np.stack([no[a] for a in self.agent_names], axis=0))
            st_l.append(e.state())
            rew_l.append([r.get(a, 0.0) for a in self.agent_names])
            done_l.append([1.0 if done_all else 0.0] * self.num_agents)
            act_l.append([1.0 if e.engine.agents[j].alive else 0.0 for j in range(self.num_agents)])
            info_l.append(first_info)

        return (
            np.stack(obs_l, axis=0),
            np.stack(st_l, axis=0),
            np.array(rew_l, dtype=np.float32),
            np.array(done_l, dtype=np.float32),
            np.array(act_l, dtype=np.float32),
            info_l,
        )

    def close(self) -> None:
        if self.vec_type == "subproc":
            for rem in self.remotes:
                rem.send(("close", None))
            for p in self.ps:
                p.join(timeout=2)
        else:
            for e in self.envs:
                e.close()
