"""Unified checkpoint loader for MARL inference in enjoy.py and evaluate.py."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch

from benchmarks.marl.networks import (
    BranchingQNetwork,
    FactoredMATNetwork,
    FactoredMultiDiscreteActor,
)


class LoadedMARLPolicy:
    """Uniform deterministic inference wrapper across all supported MARL algorithms."""

    def __init__(self, algo: str, models: Dict[str, torch.nn.Module], device: torch.device):
        self.algo = algo
        self.models = models
        self.device = device
        for m in self.models.values():
            m.eval()

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        if self.algo in ("ippo", "mappo"):
            acts, _ = self.models["actor"].act(obs_t, deterministic=True)
            return acts.cpu().numpy(), None
        if self.algo in ("happo", "hatrpo"):
            acts_list = []
            for i in range(obs_t.shape[0]):
                act_i, _ = self.models[f"actor_{i}"].act(obs_t[i : i + 1], deterministic=True)
                acts_list.append(act_i.squeeze(0))
            return torch.stack(acts_list, dim=0).cpu().numpy(), None
        if self.algo in ("vdn", "qmix"):
            acts = self.models["q_net"].greedy_actions(obs_t)
            return acts.cpu().numpy(), None
        if self.algo in ("mat", "pmat", "romat"):
            acts, _, _, _, overlay = self.models["mat"].act(obs_t.unsqueeze(0), deterministic=True)
            return acts.squeeze(0).cpu().numpy(), overlay
        raise ValueError(f"Unsupported MARL algo in LoadedMARLPolicy: {self.algo}")


def load_marl_policy(ckpt_path: str, env: Any, device: str = "cpu") -> Tuple[LoadedMARLPolicy, str]:
    dev = torch.device(device)
    payload = torch.load(ckpt_path, map_location=dev, weights_only=False)
    algo = payload["algo"].lower()
    obs_dim = int(payload["obs_dim"])
    num_agents = int(payload["num_agents"])
    hidden_dim = int(payload.get("hidden_dim", 128))
    sd = payload["models"]

    models: Dict[str, torch.nn.Module] = {}
    if algo in ("ippo", "mappo"):
        actor = FactoredMultiDiscreteActor(obs_dim, hidden_dim).to(dev)
        actor.load_state_dict(sd["actor"])
        models["actor"] = actor
    elif algo in ("happo", "hatrpo"):
        for i in range(num_agents):
            actor = FactoredMultiDiscreteActor(obs_dim, hidden_dim).to(dev)
            actor.load_state_dict(sd[f"actor_{i}"])
            models[f"actor_{i}"] = actor
    elif algo in ("vdn", "qmix"):
        q_net = BranchingQNetwork(obs_dim, hidden_dim).to(dev)
        q_net.load_state_dict(sd["q_net"])
        models["q_net"] = q_net
    elif algo in ("mat", "pmat", "romat"):
        mat = FactoredMATNetwork(obs_dim, num_agents, hidden_dim=hidden_dim, mode=algo).to(dev)
        mat.load_state_dict(sd["mat"])
        models["mat"] = mat
    else:
        raise ValueError(f"Unknown MARL checkpoint algo: {algo}")

    return LoadedMARLPolicy(algo, models, dev), algo
