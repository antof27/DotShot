"""On-policy MARL rollout buffer and off-policy episode replay buffer."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch


class MARLRolloutBuffer:
    """Fixed-horizon vectorized buffer with GAE and active-agent masking."""

    def __init__(
        self,
        n_steps: int,
        num_envs: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        device: torch.device,
    ):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device

        self.obs = torch.zeros((n_steps, num_envs, num_agents, obs_dim), dtype=torch.float32, device=device)
        self.states = torch.zeros((n_steps, num_envs, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((n_steps, num_envs, num_agents, 4), dtype=torch.long, device=device)
        self.orders = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.active_masks = torch.ones((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.returns = torch.zeros((n_steps, num_envs, num_agents), dtype=torch.float32, device=device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        active_masks: torch.Tensor,
        orders: Optional[torch.Tensor] = None,
    ) -> None:
        self.obs[self.ptr].copy_(obs)
        self.states[self.ptr].copy_(state)
        self.actions[self.ptr].copy_(actions)
        self.log_probs[self.ptr].copy_(log_probs)
        self.rewards[self.ptr].copy_(rewards)
        self.values[self.ptr].copy_(values)
        self.dones[self.ptr].copy_(dones)
        self.active_masks[self.ptr].copy_(active_masks)
        if orders is not None:
            self.orders[self.ptr].copy_(orders)
        self.ptr += 1

    def compute_gae(self, next_values: torch.Tensor, next_dones: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        gae = torch.zeros((self.num_envs, self.num_agents), dtype=torch.float32, device=self.device)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nv = next_values
                non_term = 1.0 - next_dones
            else:
                nv = self.values[t + 1]
                non_term = 1.0 - self.dones[t + 1]

            delta = self.rewards[t] + gamma * nv * non_term - self.values[t]
            gae = delta + gamma * gae_lambda * non_term * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def reset(self) -> None:
        self.ptr = 0


class EpisodeReplayBuffer:
    """Off-policy sequence chunk replay buffer for VDN and QMIX."""

    def __init__(
        self,
        capacity: int,
        chunk_len: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
    ):
        self.capacity = capacity
        self.chunk_len = chunk_len
        self.num_agents = num_agents
        self.obs = np.zeros((capacity, chunk_len, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, chunk_len, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((capacity, chunk_len, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, chunk_len, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, chunk_len, num_agents, 4), dtype=np.int64)
        self.rewards = np.zeros((capacity, chunk_len), dtype=np.float32)
        self.dones = np.zeros((capacity, chunk_len), dtype=np.float32)
        self.masks = np.zeros((capacity, chunk_len), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add_chunk(self, chunk: Dict[str, np.ndarray]) -> None:
        idx = self.ptr
        self.obs[idx] = chunk["obs"]
        self.next_obs[idx] = chunk["next_obs"]
        self.states[idx] = chunk["states"]
        self.next_states[idx] = chunk["next_states"]
        self.actions[idx] = chunk["actions"]
        self.rewards[idx] = chunk["rewards"]
        self.dones[idx] = chunk["dones"]
        self.masks[idx] = chunk["masks"]
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "states": torch.as_tensor(self.states[idxs], device=device),
            "next_states": torch.as_tensor(self.next_states[idxs], device=device),
            "actions": torch.as_tensor(self.actions[idxs], device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=device),
            "dones": torch.as_tensor(self.dones[idxs], device=device),
            "masks": torch.as_tensor(self.masks[idxs], device=device),
        }
