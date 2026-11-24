"""Factored categorical heads for upstream MAT on DotShot's MultiDiscrete actions.

Upstream MAT supports ``Discrete`` and ``Box`` actions only. Instead of the old
continuous Gaussian adapter (finding G), the decoder here emits ``sum(nvec)``
logits per agent, split into one categorical per branch ``(3, 3, 32, 4)``. The
joint log-prob and entropy of an agent's action are sums over branches, and
agents are still decoded autoregressively exactly as in upstream MAT: agent i
conditions on the one-hot actions of agents < i. Encoder, decoder blocks,
trainer and buffer are the unmodified upstream classes.
"""

from typing import List, Sequence

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat.algorithms.utils.util import check

# Upstream MAT feeds the encoder a zero "state" of this hard-coded width (state unused).
_UPSTREAM_STATE_WIDTH = 37


class MultiDiscreteMAT(MultiAgentTransformer):
    """MultiAgentTransformer whose decoder head is split into per-branch categoricals."""

    def __init__(self, state_dim, obs_dim, nvec: Sequence[int], n_agent, n_block, n_embd, n_head,
                 encode_state=False, device=torch.device("cpu"), dec_actor=False, share_actor=False):
        self.nvec: List[int] = [int(n) for n in nvec]
        super().__init__(state_dim, obs_dim, sum(self.nvec), n_agent, n_block, n_embd, n_head,
                         encode_state=encode_state, device=device, action_type="Discrete",
                         dec_actor=dec_actor, share_actor=share_actor)

    def _encode(self, obs: torch.Tensor):
        state = torch.zeros((*obs.shape[:-1], _UPSTREAM_STATE_WIDTH), **self.tpdv)
        return self.encoder(state, obs)

    def _one_hot(self, action: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [F.one_hot(action[..., k], n) for k, n in enumerate(self.nvec)], dim=-1
        ).to(**self.tpdv)

    def _dists(self, logits: torch.Tensor) -> List[Categorical]:
        return [Categorical(logits=part) for part in torch.split(logits, self.nvec, dim=-1)]

    def forward(self, state, obs, action, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(self.device).long()
        batch, n_agent = obs.shape[0], obs.shape[1]
        v_loc, obs_rep = self._encode(obs)
        shifted = torch.zeros((batch, n_agent, self.action_dim + 1), **self.tpdv)
        shifted[:, 0, 0] = 1
        shifted[:, 1:, 1:] = self._one_hot(action)[:, :-1, :]
        dists = self._dists(self.decoder(shifted, obs_rep, obs))
        action_log = sum(d.log_prob(action[..., k]) for k, d in enumerate(dists)).unsqueeze(-1)
        entropy = sum(d.entropy() for d in dists).unsqueeze(-1)
        return action_log, v_loc, entropy

    @torch.no_grad()
    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        batch, n_agent = obs.shape[0], obs.shape[1]
        v_loc, obs_rep = self._encode(obs)
        shifted = torch.zeros((batch, n_agent, self.action_dim + 1), **self.tpdv)
        shifted[:, 0, 0] = 1
        actions = torch.zeros((batch, n_agent, len(self.nvec)), dtype=torch.long, device=self.device)
        log_probs = torch.zeros((batch, n_agent, 1), **self.tpdv)
        for i in range(n_agent):
            dists = self._dists(self.decoder(shifted, obs_rep, obs)[:, i, :])
            branch = [d.probs.argmax(dim=-1) if deterministic else d.sample() for d in dists]
            actions[:, i, :] = torch.stack(branch, dim=-1)
            log_probs[:, i, 0] = sum(d.log_prob(a) for d, a in zip(dists, branch))
            if i + 1 < n_agent:
                shifted[:, i + 1, 1:] = self._one_hot(actions[:, i, :])
        return actions, log_probs, v_loc


class DotShotMATPolicy(TransformerPolicy):
    """Upstream TransformerPolicy around ``MultiDiscreteMAT``.

    Actions are stored as ``(…, len(nvec))`` integers; log-probs and entropies as
    one joint column, so MATTrainer's ratio ``exp(new - old)`` is the true joint ratio.
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        self.action_type = "MultiDiscrete"
        self.obs_dim = int(obs_space.shape[0])
        self.share_obs_dim = int(cent_obs_space.shape[0])
        self.nvec = [int(n) for n in act_space.nvec]
        self.act_dim = sum(self.nvec)
        self.act_num = len(self.nvec)
        self.num_agents = num_agents
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.transformer = MultiDiscreteMAT(
            self.share_obs_dim, self.obs_dim, self.nvec, num_agents,
            n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
            encode_state=args.encode_state, device=device,
            dec_actor=args.dec_actor, share_actor=args.share_actor,
        )
        self.optimizer = torch.optim.Adam(
            self.transformer.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay
        )

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks,
                    available_actions=None, deterministic=False):
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions, log_probs, values = self.transformer.get_actions(cent_obs, obs, None, deterministic)
        return (
            values.view(-1, 1),
            actions.view(-1, self.act_num),
            log_probs.view(-1, 1),
            check(rnn_states_actor).to(**self.tpdv),
            check(rnn_states_critic).to(**self.tpdv),
        )

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                         available_actions=None, active_masks=None):
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = np.asarray(actions).reshape(-1, self.num_agents, self.act_num)
        log_probs, values, entropy = self.transformer(cent_obs, obs, actions, None)
        log_probs = log_probs.view(-1, 1)
        entropy = entropy.view(-1, 1)
        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum().clamp(min=1.0)
        else:
            entropy = entropy.mean()
        return values.view(-1, 1), log_probs, entropy
