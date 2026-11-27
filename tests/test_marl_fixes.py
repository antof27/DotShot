"""Regression tests for the Milestone 3 performance fixes (findings A-D, F, G, I)."""

import numpy as np
import pytest
import torch
from harl.envs.env_wrappers import ShareDummyVecEnv
from harl.models.base.act import ACTLayer
from gymnasium import spaces

from benchmarks.eval.marl_eval import evaluate_marl
from benchmarks.official import harl_runner as hr
from benchmarks.official.harl_dotshot_env import DotShotHARLEnv
from benchmarks.runners.run_marl_baselines import ChargePolicy

T, E = 8, 2  # rollout length and threads of the tiny test runner


def _tiny_runner(algo: str, stage: int = 3, state_type: str = "EP", **cfg):
    args = hr.build_harl_args(
        {"episode_length": T, "n_rollout_threads": E, "actor_num_mini_batch": 1,
         "critic_num_mini_batch": 1, **cfg},
        algo,
    )
    envs = ShareDummyVecEnv(
        [lambda r=r: DotShotHARLEnv(stage=stage, seed=r, state_type=state_type) for r in range(E)]
    )
    runner = hr._DotShotHARLRunner(algo, args, state_type, envs, envs.n_agents, torch.device("cpu"))
    runner.warmup()
    return runner, envs


def _rollout(runner, envs, steps=T):
    runner.prep_rollout()
    for step in range(steps):
        values, actions, logp, rnn, rnn_c = runner.collect(step)
        obs, share_obs, rewards, dones, infos, avail = envs.step(actions)
        runner.insert((obs, share_obs, rewards, dones, infos, avail, values, actions, logp, rnn, rnn_c))


@pytest.mark.parametrize("algo", ["mappo", "happo", "haa2c", "hatrpo"])
def test_harl_update_uses_true_joint_ratio(algo, monkeypatch):
    """Finding A: through the real buffer insert + update path, ratio == exp(new - old)."""
    torch.manual_seed(0)
    runner, envs = _tiny_runner(algo, share_param=False)
    _rollout(runner, envs)
    actor, buf = runner.actor[0], runner.actor_buffer[0]
    if algo != "mappo":
        buf.update_factor(np.ones((T, E, 1), dtype=np.float32))
    # The 4 stored columns are copies of the joint log-prob (that is what made "prod" = r^4).
    assert np.allclose(buf.action_log_probs, buf.action_log_probs[..., :1])

    with torch.no_grad():
        for p in actor.actor.parameters():
            p.add_(0.05 * torch.randn_like(p))

    captured = {}
    real_eval = actor.evaluate_actions

    def spy_eval(*a, **k):
        out = real_eval(*a, **k)
        captured.setdefault("new_lp", out[0].detach().clone())
        return out

    real_agg = torch.mean

    def spy_mean(x, *a, **k):
        out = real_agg(x, *a, **k)
        if "ratio_in" not in captured and k.get("keepdim") and x.dim() == 2 and x.shape[-1] == 4:
            captured["ratio_in"], captured["ratio"] = x.detach().clone(), out.detach().clone()
        return out

    monkeypatch.setattr(actor, "evaluate_actions", spy_eval)
    monkeypatch.setattr(torch, "mean", spy_mean)
    sample = next(buf.feed_forward_generator_actor(np.ones((T, E, 1), dtype=np.float32), 1))
    actor.update(sample)
    monkeypatch.undo()

    old_joint = torch.as_tensor(sample[5][:, :1])
    expected = torch.exp(captured["new_lp"] - old_joint)
    assert actor.action_aggregation == "mean"
    assert torch.allclose(captured["ratio"], expected, atol=1e-5)
    assert not torch.allclose(expected, torch.ones_like(expected))
    # The upstream default would have produced r^4.
    assert torch.allclose(torch.prod(captured["ratio_in"], dim=-1, keepdim=True), expected**4, rtol=1e-4)
    envs.close()


def test_masked_multidiscrete_entropy_excludes_dead_rows():
    """Finding B: scalar entropy, averaged over active rows only, summed over branches."""
    torch.manual_seed(0)
    layer = ACTLayer(spaces.MultiDiscrete([3, 3, 32, 4]), 16, "orthogonal_", 0.01)
    x = torch.randn(6, 16)
    actions = torch.stack([torch.randint(0, n, (6,)) for n in (3, 3, 32, 4)], dim=-1)
    masks = torch.tensor([[1.0], [1.0], [0.0], [1.0], [0.0], [1.0]])
    _, ent, _ = layer.evaluate_actions(x, actions, active_masks=masks)
    assert ent.dim() == 0
    active = masks.squeeze(-1).bool()
    manual = sum(head(x).entropy()[active].mean() for head in layer.action_outs)
    assert torch.allclose(ent, manual, atol=1e-6)
    # With every row active the value matches upstream (entropy_coef keeps its scale).
    ones = torch.ones(6, 1)
    _, ent_all, _ = layer.evaluate_actions(x, actions, active_masks=ones)
    _, ent_up, _ = hr._ORIGINAL_ACT_EVALUATE(layer, x, actions, active_masks=ones)
    assert torch.allclose(ent_all, ent_up, atol=1e-6)


def test_dead_agent_is_masked_until_reset():
    """Finding C: a dead agent is done (active_mask 0) while the episode continues."""
    runner, envs = _tiny_runner("mappo")
    envs.envs[0].env.engine.agents[0].hp = 0.0
    envs.envs[0].env.engine.agents[0].alive = False
    _rollout(runner, envs, steps=1)
    assert runner.actor_buffer[0].active_masks[1][0, 0] == 0.0  # dead agent, thread 0
    assert runner.actor_buffer[1].active_masks[1][0, 0] == 1.0  # teammate stays active
    assert runner.actor_buffer[0].active_masks[1][1, 0] == 1.0  # same agent, other thread
    obs, share_obs, _ = envs.envs[0].reset()
    assert envs.envs[0].env.engine.agents[0].alive
    envs.close()


def test_fp_critic_input_is_agent_specific_and_team_reward_shared():
    """Finding D: FP critic input differs per agent; team mode gives one reward to all agents."""
    env = DotShotHARLEnv(stage=3, seed=0, state_type="FP")
    obs, share, _ = env.reset()
    assert share[0].shape[0] == env.env.state_space().shape[0] + env.n_agents + obs[0].shape[0]
    assert not np.allclose(share[0], share[1])
    env.env.engine.agents[1].hp = 0.0
    env.env.engine.agents[1].alive = False
    for _ in range(5):
        _, _, rewards, dones, _, _ = env.step(np.array([[1, 1, 0, 1], [1, 1, 0, 1]]))
        assert rewards[0] == rewards[1]
        assert dones[1] and not dones[0]
    ep = DotShotHARLEnv(stage=3, seed=0, state_type="EP")
    _, share_ep, _ = ep.reset()
    assert np.allclose(share_ep[0], share_ep[1])


def test_stage4_mappo_shares_parameters():
    """Finding F: MAPPO shares one actor across all three Stage 4 roles."""
    runner, envs = _tiny_runner("mappo", stage=4)
    assert runner.actor[0] is runner.actor[1] is runner.actor[2]
    envs.close()


def test_mat_discrete_heads_joint_logprob_and_bootstrap():
    """Finding G: factored categorical heads; truncated episodes bootstrap gamma * V(s_T)."""
    from benchmarks.official.mat_runner import _build_args, bootstrap_truncation
    from benchmarks.official.mat_dotshot_env import DotShotMATEnv
    from benchmarks.official.mat_discrete import DotShotMATPolicy

    torch.manual_seed(0)
    env = DotShotMATEnv(stage=3, seed=0)
    args = _build_args({"n_block": 1, "n_embd": 32, "n_head": 1})
    policy = DotShotMATPolicy(args, env.observation_space[0], env.share_observation_space[0],
                              env.action_space[0], env.n_agents)
    obs, share = env.reset()
    zeros = np.zeros((env.n_agents, 1, 1), dtype=np.float32)
    ones = np.ones((env.n_agents, 1), dtype=np.float32)
    _, actions, logp, _, _ = policy.get_actions(share, obs, zeros, zeros, ones)
    actions = actions.numpy()
    assert actions.shape == (env.n_agents, 4)
    assert all(0 <= actions[:, k].min() and actions[:, k].max() < n for k, n in enumerate((3, 3, 32, 4)))
    _, new_logp, entropy = policy.evaluate_actions(share, obs, zeros, zeros, actions, ones,
                                                   active_masks=torch.ones(env.n_agents, 1))
    assert torch.allclose(new_logp, logp, atol=1e-5)  # joint ratio is exactly 1 on-policy
    assert entropy.dim() == 0 and entropy.item() > 0.0
    next_obs, next_share, rewards, dones, infos = env.step(actions)

    infos[0].update(bad_transition=True, terminal_obs=next_obs, terminal_share_obs=next_share)
    base = rewards[None].copy()
    out = bootstrap_truncation(base.copy(), [infos], policy, None, 0.9)
    v = policy.get_values(next_share, next_obs, zeros, ones).detach().numpy()
    assert np.allclose(out[0], base[0] + 0.9 * v.reshape(base[0].shape), atol=1e-5)
    infos[0]["bad_transition"] = False
    assert np.allclose(bootstrap_truncation(base.copy(), [infos], policy, None, 0.9), base)


def test_marl_eval_returns_exact_episode_count():
    """Finding I: evaluation always completes exactly n episodes on fixed seeds."""
    metrics, episodes = evaluate_marl({"stage": 3}, ChargePolicy(), 3, split="val")
    assert len(episodes) == 3 and metrics["eval/n_episodes"] == 3.0
    again, _ = evaluate_marl({"stage": 3}, ChargePolicy(), 3, split="val")
    assert again["eval/return_mean"] == metrics["eval/return_mean"]
    comps = sum(episodes[0]["reward_components"].values())
    assert comps == pytest.approx(episodes[0]["return"], abs=1e-6)


def test_behaviour_cloning_warm_start_lowers_nll(tmp_path, monkeypatch):
    """Warm start (Phase 3): cloning the calibrated script lowers its NLL for HARL and MAT."""
    from benchmarks.official import behaviour_cloning as bc
    from benchmarks.official.mat_runner import _build_args
    from benchmarks.official.mat_dotshot_env import DotShotMATEnv
    from benchmarks.official.mat_discrete import DotShotMATPolicy

    monkeypatch.setattr(bc, "CACHE_DIR", tmp_path)
    data = bc.load_dataset(3, n_episodes=1)
    assert data["obs"].shape[1:] == (2, data["obs"].shape[2]) and data["actions"].shape[2] == 4
    torch.manual_seed(0)
    runner, envs = _tiny_runner("happo", share_param=False)
    first = bc.bc_harl_actor(runner, data, epochs=1, batch_size=512)["bc/nll"]
    last = bc.bc_harl_actor(runner, data, epochs=4, batch_size=512)["bc/nll"]
    assert last < first
    assert all(torch.equal(p, q) for p, q in zip(runner.actor[0].actor.parameters(), runner.actor[1].actor.parameters()))
    envs.close()

    env = DotShotMATEnv(stage=3, seed=0)
    policy = DotShotMATPolicy(_build_args({"n_embd": 32}), env.observation_space[0],
                              env.share_observation_space[0], env.action_space[0], env.n_agents)
    first = bc.bc_mat(policy, data, epochs=1, batch_size=256)["bc/nll"]
    last = bc.bc_mat(policy, data, epochs=4, batch_size=256)["bc/nll"]
    assert last < first


def test_full_wipe_with_respawn_does_not_end_harl_episode():
    """v2.2 respawn: all agents dead at once is not an episode end for HARL's vec env."""
    envs = ShareDummyVecEnv([lambda r=r: DotShotHARLEnv(stage=3, seed=r) for r in range(2)])
    envs.reset()
    engine = envs.envs[0].env.engine
    for agent in engine.agents:
        agent.hp, agent.alive, agent.respawn_timer = 0.0, False, 50
    step_before = engine.step_count
    _, _, _, dones, _, _ = envs.step(np.ones((2, engine.cfg.num_agents, 4), dtype=np.int64))
    assert not dones[0].any()
    assert engine.step_count == step_before + engine.cfg.action_repeat  # not reset
    envs.close()
