"""Neural architectures for factored MultiDiscrete MARL (IPPO, VDN/QMIX, MAPPO, HAPPO, MAT/PMAT/RoMAT)."""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


ACTION_BRANCHES: Tuple[int, ...] = (3, 3, 32, 4)


def ortho_init(layer: nn.Module, gain: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


class ValueNorm(nn.Module):
    """Running mean/std return normalization (MAPPO PopArt-style stabilization)."""

    def __init__(self, beta: float = 0.995, epsilon: float = 1e-5):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("running_mean_sq", torch.ones(1, dtype=torch.float32))
        self.register_buffer("debiasing_term", torch.zeros(1, dtype=torch.float32))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean()
        batch_sq = (x ** 2).mean()
        self.running_mean.mul_(self.beta).add_(batch_mean * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(batch_sq * (1.0 - self.beta))
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        var = (debiased_sq - debiased_mean ** 2).clamp(min=1e-4)
        return debiased_mean, torch.sqrt(var)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean_std()
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.mean_std()
        return x * std + mean


class FactoredMultiDiscreteActor(nn.Module):
    """Factored categorical policy over MultiDiscrete([3, 3, 32, 4])."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, branches: Tuple[int, ...] = ACTION_BRANCHES):
        super().__init__()
        self.branches = branches
        self.trunk = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([ortho_init(nn.Linear(hidden_dim, b), gain=0.01) for b in branches])

    def get_logits(self, obs: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(obs)
        return [head(h) for head in self.heads]

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_list = self.get_logits(obs)
        actions = []
        log_probs = torch.zeros(obs.shape[:-1], device=obs.device, dtype=torch.float32)
        for logits in logits_list:
            dist = Categorical(logits=logits)
            act = logits.argmax(dim=-1) if deterministic else dist.sample()
            actions.append(act)
            log_probs = log_probs + dist.log_prob(act)
        return torch.stack(actions, dim=-1), log_probs

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        logits_list = self.get_logits(obs)
        log_probs = torch.zeros(obs.shape[:-1], device=obs.device, dtype=torch.float32)
        entropy = torch.zeros(obs.shape[:-1], device=obs.device, dtype=torch.float32)
        for k, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            act_k = actions[..., k]
            log_probs = log_probs + dist.log_prob(act_k)
            entropy = entropy + dist.entropy()
        return log_probs, entropy, logits_list


class CentralizedCritic(nn.Module):
    """State-value critic V(x) used by IPPO (x=obs), MAPPO/HAPPO/HATRPO (x=global_state)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            ortho_init(nn.Linear(input_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            ortho_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BranchingQNetwork(nn.Module):
    """Branching Dueling Q-Network (Tavakoli et al., 2018) for factored MultiDiscrete actions."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128, branches: Tuple[int, ...] = ACTION_BRANCHES):
        super().__init__()
        self.branches = branches
        self.trunk = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.value_head = ortho_init(nn.Linear(hidden_dim, 1), gain=1.0)
        self.adv_heads = nn.ModuleList([ortho_init(nn.Linear(hidden_dim, b), gain=1.0) for b in branches])

    def forward_branches(self, obs: torch.Tensor) -> List[torch.Tensor]:
        h = self.trunk(obs)
        v = self.value_head(h)  # [..., 1]
        q_branches: List[torch.Tensor] = []
        for head in self.adv_heads:
            adv = head(h)
            q_k = v + (adv - adv.mean(dim=-1, keepdim=True))
            q_branches.append(q_k)
        return q_branches

    def q_total_for_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        q_branches = self.forward_branches(obs)
        chosen = []
        for k, q_k in enumerate(q_branches):
            act_k = actions[..., k].unsqueeze(-1)
            chosen.append(torch.gather(q_k, dim=-1, index=act_k).squeeze(-1))
        return torch.stack(chosen, dim=-1).mean(dim=-1)

    def greedy_actions(self, obs: torch.Tensor) -> torch.Tensor:
        q_branches = self.forward_branches(obs)
        return torch.stack([q_k.argmax(dim=-1) for q_k in q_branches], dim=-1)


class VDNMixer(nn.Module):
    """Additive Value Decomposition Network (Sunehag et al., 2018)."""

    def forward(self, agent_qs: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        return agent_qs.sum(dim=-1)


class QMIXMixer(nn.Module):
    """Monotonic Hypernetwork Mixer (Rashid et al., 2018): dQ_tot / dQ_i >= 0."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32, hyper_hidden: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, num_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        orig_shape = agent_qs.shape[:-1]
        bs = int(np.prod(orig_shape))
        qs = agent_qs.reshape(bs, 1, self.num_agents)
        st = state.reshape(bs, -1)

        w1 = torch.abs(self.hyper_w1(st)).view(bs, self.num_agents, self.embed_dim)
        b1 = self.hyper_b1(st).view(bs, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(qs, w1) + b1)

        w2 = torch.abs(self.hyper_w2(st)).view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(st).view(bs, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(*orig_shape)


class FactoredMATNetwork(nn.Module):
    """
    Factored Multi-Agent Transformer supporting MAT, PMAT (Plackett-Luce ordering),
    and RoMAT (learnable agent role embeddings).
    """

    def __init__(
        self,
        obs_dim: int,
        num_agents: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        branches: Tuple[int, ...] = ACTION_BRANCHES,
        mode: str = "mat",
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.branches = branches
        self.mode = mode  # "mat", "pmat", or "romat"

        self.obs_embed = nn.Sequential(
            ortho_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.role_embed = nn.Embedding(num_agents, hidden_dim) if mode == "romat" else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.value_head = ortho_init(nn.Linear(hidden_dim, 1), gain=1.0)

        self.priority_head = (
            nn.Sequential(
                ortho_init(nn.Linear(hidden_dim, hidden_dim // 2)),
                nn.ReLU(),
                ortho_init(nn.Linear(hidden_dim // 2, 1), gain=0.01),
            )
            if mode == "pmat"
            else None
        )

        self.start_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.branch_embeds = nn.ModuleList([nn.Embedding(b, hidden_dim) for b in branches])

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.action_heads = nn.ModuleList([ortho_init(nn.Linear(hidden_dim, b), gain=0.01) for b in branches])

    def encode(self, obs_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n, _ = obs_n.shape
        x = self.obs_embed(obs_n)
        if self.role_embed is not None:
            role_ids = torch.arange(n, device=obs_n.device).unsqueeze(0).expand(bs, -1)
            x = x + self.role_embed(role_ids)
        enc = self.encoder(x)
        values = self.value_head(enc).squeeze(-1)  # [B, N]
        return enc, values

    def _sample_order(self, enc: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n, _ = enc.shape
        device = enc.device
        if self.priority_head is None:
            order = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)
            return order, torch.zeros(bs, device=device)

        logits = self.priority_head(enc).squeeze(-1)
        if deterministic:
            order = torch.argsort(logits, dim=-1, descending=True)
            return order, torch.zeros(bs, device=device)

        masked = logits.clone()
        order_cols = []
        order_lp = torch.zeros(bs, device=device)
        for _ in range(n):
            probs = torch.softmax(masked, dim=-1)
            dist = Categorical(probs=probs)
            choice = dist.sample()
            order_cols.append(choice)
            order_lp = order_lp + dist.log_prob(choice)
            mask = F.one_hot(choice, num_classes=n).bool()
            masked = masked.masked_fill(mask, -1e9)
        return torch.stack(order_cols, dim=1), order_lp

    def _embed_factored_actions(self, actions: torch.Tensor) -> torch.Tensor:
        emb = torch.zeros((*actions.shape[:-1], self.hidden_dim), device=actions.device, dtype=torch.float32)
        for k, emb_layer in enumerate(self.branch_embeds):
            emb = emb + emb_layer(actions[..., k])
        return emb / len(self.branch_embeds)

    @staticmethod
    def _gather_by_order(tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        expand_shape = list(order.shape) + [tensor.shape[-1]]
        idx = order.unsqueeze(-1).expand(*expand_shape)
        return torch.gather(tensor, dim=1, index=idx)

    @staticmethod
    def _scatter_by_order(ordered_tensor: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(ordered_tensor)
        if ordered_tensor.ndim == 2:
            out.scatter_(1, order, ordered_tensor)
        else:
            idx = order.unsqueeze(-1).expand_as(ordered_tensor)
            out.scatter_(1, idx, ordered_tensor)
        return out

    @torch.no_grad()
    def act(
        self, obs_n: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        bs, n, _ = obs_n.shape
        device = obs_n.device
        enc, values = self.encode(obs_n)
        order, _ = self._sample_order(enc, deterministic=deterministic)
        ordered_enc = self._gather_by_order(enc, order)

        dec_tokens = self.start_token.expand(bs, 1, -1)
        ordered_actions = torch.zeros(bs, n, len(self.branches), dtype=torch.long, device=device)
        ordered_lps = torch.zeros(bs, n, dtype=torch.float32, device=device)

        for m in range(n):
            causal_mask = torch.triu(torch.full((m + 1, m + 1), float("-inf"), device=device), diagonal=1)
            dec_out = self.decoder(dec_tokens, ordered_enc, tgt_mask=causal_mask)
            h_m = dec_out[:, -1, :]

            step_acts = []
            step_lp = torch.zeros(bs, device=device)
            for head in self.action_heads:
                logits = head(h_m)
                dist = Categorical(logits=logits)
                act_k = logits.argmax(dim=-1) if deterministic else dist.sample()
                step_acts.append(act_k)
                step_lp = step_lp + dist.log_prob(act_k)

            step_act_tensor = torch.stack(step_acts, dim=-1)
            ordered_actions[:, m, :] = step_act_tensor
            ordered_lps[:, m] = step_lp
            if m + 1 < n:
                next_tok = self._embed_factored_actions(step_act_tensor).unsqueeze(1)
                dec_tokens = torch.cat([dec_tokens, next_tok], dim=1)

        actions_n = self._scatter_by_order(ordered_actions, order)
        log_probs_n = self._scatter_by_order(ordered_lps, order)
        overlay = {"order": order[0].detach().cpu().tolist()}
        return actions_n, log_probs_n, values, order, overlay

    def evaluate_actions(
        self,
        obs_n: torch.Tensor,
        actions_n: torch.Tensor,
        order: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, n, _ = obs_n.shape
        device = obs_n.device
        enc, values = self.encode(obs_n)
        if order is None:
            order = torch.arange(n, device=device).unsqueeze(0).expand(bs, -1)

        ordered_enc = self._gather_by_order(enc, order)
        ordered_actions = self._gather_by_order(actions_n, order)

        if n > 1:
            shifted_emb = self._embed_factored_actions(ordered_actions[:, :-1, :])
            dec_input = torch.cat([self.start_token.expand(bs, 1, -1), shifted_emb], dim=1)
        else:
            dec_input = self.start_token.expand(bs, 1, -1)

        causal_mask = torch.triu(torch.full((n, n), float("-inf"), device=device), diagonal=1)
        dec_out = self.decoder(dec_input, ordered_enc, tgt_mask=causal_mask)

        ordered_lps = torch.zeros(bs, n, device=device)
        ordered_ent = torch.zeros(bs, n, device=device)
        for k, head in enumerate(self.action_heads):
            logits = head(dec_out)
            dist = Categorical(logits=logits)
            ordered_lps = ordered_lps + dist.log_prob(ordered_actions[..., k])
            ordered_ent = ordered_ent + dist.entropy()

        log_probs_n = self._scatter_by_order(ordered_lps, order)
        entropy_n = self._scatter_by_order(ordered_ent, order)
        return log_probs_n, entropy_n, values
