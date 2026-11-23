"""Play DotShot Game v2 manually or evaluate trained SB3 / MARL checkpoints with optional MP4 export."""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

import gymnasium as gym
import numpy as np

import dotshot  # Registers DotShot-Stage1-v1 and DotShot-Stage2-v1
from dotshot.envs.multi_agent import DotShotParallelEnv
from dotshot.envs.single_agent import DotShotSingleEnv


def _capture_manual_action(env_engine, selected_weapon: int) -> np.ndarray:
    import pygame

    pygame.event.pump()
    keys = pygame.key.get_pressed()
    mx, my = 0, 0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        mx -= 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        mx += 1
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        my -= 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        my += 1

    mouse_x, mouse_y = pygame.mouse.get_pos()
    agent = env_engine.agents[0]
    dx = mouse_x - agent.x
    dy = mouse_y - agent.y
    if dx != 0 or dy != 0:
        ang = np.arctan2(-dy, dx)
        if ang < 0:
            ang += 2.0 * np.pi
        aim_idx = int(round(ang / (2.0 * np.pi / env_engine.cfg.aim_bins))) % env_engine.cfg.aim_bins
    else:
        aim_idx = 0

    click = pygame.mouse.get_pressed()
    shoot = bool(click[0] or keys[pygame.K_SPACE])
    weapon_action = (selected_weapon + 1) if shoot else 0
    return np.array([mx + 1, my + 1, aim_idx, weapon_action], dtype=np.int64)


def run_manual(stage: int, max_steps: int) -> None:
    import pygame

    print("\n=== DotShot Game v2 - Manual Play ===")
    print("Move: WASD / Arrows | Aim: Mouse | Element: 1=Water, 2=Grass, 3=Fire")
    print("Shoot: Left Click / Space | Ability (Hold Fire): Release trigger | Overlay: TAB\n")

    if stage in (1, 2):
        env = DotShotSingleEnv(stage=stage, render_mode="human", max_steps=max_steps)
        obs, _ = env.reset()
        selected_weapon = 0
        try:
            while True:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_1]:
                    selected_weapon = 0
                elif keys[pygame.K_2]:
                    selected_weapon = 1
                elif keys[pygame.K_3]:
                    selected_weapon = 2

                action = _capture_manual_action(env.engine, selected_weapon)
                obs, reward, terminated, truncated, info = env.step(action)
                if env._renderer and env._renderer.quit_requested:
                    break
                if terminated or truncated:
                    print(f"Episode ended | Score: {info['score']:.0f} | Kills: {info['kills']:.0f} | Win: {info['win']}")
                    obs, _ = env.reset()
        finally:
            env.close()
    else:
        env = DotShotParallelEnv(stage=stage, render_mode="human", max_steps=max_steps)
        obs, _ = env.reset()
        selected_weapon = 0
        try:
            while True:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_1]:
                    selected_weapon = 0
                elif keys[pygame.K_2]:
                    selected_weapon = 1
                elif keys[pygame.K_3]:
                    selected_weapon = 2

                actions = {name: np.array([1, 1, 0, 0], dtype=np.int64) for name in env.agents}
                if "agent_0" in actions:
                    actions["agent_0"] = _capture_manual_action(env.engine, selected_weapon)
                obs, rewards, terms, truncs, infos = env.step(actions)
                if env._renderer and env._renderer.quit_requested:
                    break
                if not env.agents:
                    first_info = next(iter(infos.values()))
                    print(f"Episode ended | Score: {first_info['score']:.0f} | Core HP: {first_info['core_hp_ratio']*100:.1f}%")
                    obs, _ = env.reset()
        finally:
            env.close()


def run_policy(
    stage: int,
    model_path: str,
    max_steps: int,
    record_path: str | None,
    fps: int,
    seed: int,
    render_mode: str | None = None,
) -> None:
    import torch

    render_mode = render_mode or ("rgb_array" if record_path else "human")
    if record_path:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    frames = []
    if stage in (1, 2):
        from stable_baselines3 import A2C, PPO

        env = DotShotSingleEnv(stage=stage, render_mode=render_mode, max_steps=max_steps)
        obs, _ = env.reset(seed=seed)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        loader = A2C if "a2c" in os.path.basename(model_path).lower() else PPO
        model = loader.load(model_path, device="cpu")

        try:
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if record_path:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                elif env._renderer and env._renderer.quit_requested:
                    break
                if terminated or truncated:
                    print(f"Finished | Score: {info['score']:.0f} | Kills: {info['kills']:.0f} | Win: {info['win']}")
                    break
        finally:
            env.close()
    else:
        from benchmarks.marl.policy_loader import load_marl_policy

        env = DotShotParallelEnv(stage=stage, render_mode=render_mode, max_steps=max_steps)
        obs, _ = env.reset(seed=seed)
        policy, algo_name = load_marl_policy(model_path, env, device="cpu")

        try:
            for _ in range(max_steps):
                if not env.agents:
                    break
                obs_batch = np.stack([obs[a] for a in env.possible_agents], axis=0)
                act_np, overlay = policy.act_deterministic(obs_batch)
                env.overlay_info = overlay
                actions = {a: act_np[i] for i, a in enumerate(env.agents)}
                obs, _, terms, truncs, infos = env.step(actions)
                if record_path:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                elif env._renderer and env._renderer.quit_requested:
                    break
                if not env.agents:
                    first_info = next(iter(infos.values()))
                    print(f"Finished ({algo_name}) | Score: {first_info['score']:.0f} | Core HP: {first_info['core_hp_ratio']*100:.1f}%")
                    break
        finally:
            env.close()

    if record_path and frames:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        imageio.mimsave(record_path, frames, fps=fps)
        print(f"Saved gameplay video ({len(frames)} frames) to {record_path}")


def _save_frames(record_path: str | None, frames: list, fps: int) -> None:
    if record_path and frames:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        imageio.mimsave(record_path, frames, fps=fps)
        print(f"Saved gameplay video ({len(frames)} frames) to {record_path}")


def _print_parallel_episode(label: str, infos: dict) -> None:
    if not infos:
        return
    first_info = next(iter(infos.values()))
    print(
        f"Finished ({label}) | Score: {first_info['score']:.0f} | "
        f"Core HP: {first_info['core_hp_ratio']*100:.1f}%"
    )


def _load_official_harl(ckpt_dir: str, env: DotShotParallelEnv, device):
    import json

    import torch
    from harl.models.policy_models.stochastic_policy import StochasticPolicy

    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as file:
        meta = json.load(file)

    hidden_sizes = [int(size) for size in meta.get("hidden_sizes", [128, 128])]
    recurrent = bool(meta.get("recurrent", False))
    share_param = bool(meta.get("share_param", False))
    model_args = {
        "hidden_sizes": hidden_sizes,
        "activation_func": "relu",
        "use_feature_normalization": True,
        "initialization_method": "orthogonal_",
        "gain": 0.01,
        "use_naive_recurrent_policy": False,
        "use_recurrent_policy": recurrent,
        "recurrent_n": 1,
        "data_chunk_length": 10,
        "lr": 5e-4,
        "critic_lr": 5e-4,
        "opti_eps": 1e-5,
        "weight_decay": 0.0,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
        "use_policy_active_masks": True,
    }

    actor_ids = [0] if share_param else list(range(len(env.possible_agents)))
    actors = {}
    for agent_id in actor_ids:
        path = os.path.join(ckpt_dir, f"actor_agent{agent_id}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HARL actor checkpoint not found: {path}")
        actor = StochasticPolicy(
            model_args,
            env.observation_space("agent_0"),
            env.action_space("agent_0"),
            device=device,
        )
        actor.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        actor.eval()
        actors[agent_id] = actor
    return actors, share_param, hidden_sizes[-1]


def run_official_harl(
    stage: int,
    ckpt_dir: str,
    max_steps: int,
    record_path: str | None,
    fps: int,
    seed: int,
    render_mode: str | None = None,
) -> None:
    import torch

    render_mode = render_mode or ("rgb_array" if record_path else "human")
    if record_path:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = DotShotParallelEnv(stage=stage, render_mode=render_mode, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    actors, share_param, rnn_hidden = _load_official_harl(ckpt_dir, env, torch.device("cpu"))
    rnn_states = np.zeros((len(env.possible_agents), 1, rnn_hidden), dtype=np.float32)
    masks = np.ones((len(env.possible_agents), 1), dtype=np.float32)
    frames: list = []

    try:
        for _ in range(max_steps):
            if not env.agents:
                break
            actions = {}
            for agent_id, agent in enumerate(env.possible_agents):
                actor = actors[0] if share_param else actors[agent_id]
                action, _, new_rnn_state = actor(
                    np.expand_dims(obs[agent], axis=0),
                    rnn_states[agent_id : agent_id + 1],
                    masks[agent_id : agent_id + 1],
                    None,
                    True,
                )
                actions[agent] = action[0].cpu().numpy().astype(np.int64)
                rnn_states[agent_id] = new_rnn_state[0].cpu().numpy()

            obs, _, _, _, infos = env.step(actions)
            if record_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif env._renderer and env._renderer.quit_requested:
                break
            if not env.agents:
                _print_parallel_episode("official_harl", infos)
                break
    finally:
        env.close()

    _save_frames(record_path, frames, fps)


def run_official_mat(
    stage: int,
    ckpt_dir: str,
    max_steps: int,
    record_path: str | None,
    fps: int,
    seed: int,
    render_mode: str | None = None,
) -> None:
    import json

    import torch
    from gymnasium import spaces

    mat_root = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "Multi-Agent-Transformer"
    if str(mat_root) not in sys.path:
        sys.path.insert(0, str(mat_root))

    from benchmarks.official.mat_dotshot_env import continuous_to_multidiscrete
    from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy

    render_mode = render_mode or ("rgb_array" if record_path else "human")
    if record_path:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    env = DotShotParallelEnv(stage=stage, render_mode=render_mode, max_steps=max_steps)
    obs, _ = env.reset(seed=seed)

    with open(os.path.join(ckpt_dir, "meta.json"), "r", encoding="utf-8") as file:
        meta = json.load(file)

    n_agents = len(env.possible_agents)
    hidden_size = int(meta.get("hidden_size", meta.get("n_embd", 64)))
    args = argparse.Namespace(
        algorithm_name="mat",
        lr=5e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        use_policy_active_masks=True,
        n_block=int(meta.get("n_block", 1)),
        n_embd=int(meta.get("n_embd", 64)),
        n_head=int(meta.get("n_head", 1)),
        encode_state=bool(meta.get("encode_state", False)),
        dec_actor=bool(meta.get("dec_actor", False)),
        share_actor=bool(meta.get("share_actor", False)),
        env_name="DotShot",
    )
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
    policy = TransformerPolicy(
        args,
        env.observation_space("agent_0"),
        env.state_space(),
        act_space,
        n_agents,
        device=torch.device("cpu"),
    )
    transformer_path = os.path.join(ckpt_dir, "transformer.pt")
    if not os.path.isfile(transformer_path):
        raise FileNotFoundError(f"MAT checkpoint not found: {transformer_path}")
    policy.transformer.load_state_dict(
        torch.load(transformer_path, map_location=torch.device("cpu"), weights_only=True)
    )
    policy.eval()

    rnn_states = np.zeros((n_agents, 1, hidden_size), dtype=np.float32)
    masks = np.ones((n_agents, 1), dtype=np.float32)
    frames: list = []

    try:
        for _ in range(max_steps):
            if not env.agents:
                break
            obs_batch = np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
            state = np.asarray(env.state(), dtype=np.float32)
            state_batch = np.repeat(state[None, :], n_agents, axis=0)
            actions, _ = policy.act(
                state_batch,
                obs_batch,
                rnn_states,
                masks,
                deterministic=True,
            )
            actions_np = actions.detach().cpu().numpy()
            action_dict = {
                agent: continuous_to_multidiscrete(actions_np[index])
                for index, agent in enumerate(env.possible_agents)
            }

            obs, _, _, _, infos = env.step(action_dict)
            if record_path:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            elif env._renderer and env._renderer.quit_requested:
                break
            if not env.agents:
                _print_parallel_episode("official_mat", infos)
                break
    finally:
        env.close()

    _save_frames(record_path, frames, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="DotShot Game v2 Player & Evaluator")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="DotShot curriculum stage (1-4)")
    parser.add_argument("--manual", action="store_true", help="Play manually with keyboard and mouse")
    parser.add_argument("--model", type=str, default=None, help="Path to trained SB3 (.zip) or MARL (.pt) checkpoint")
    parser.add_argument("--official-harl", type=str, default=None, metavar="CKPT_DIR", help="HARL checkpoint directory from the official runner")
    parser.add_argument("--official-mat", type=str, default=None, metavar="CKPT_DIR", help="MAT checkpoint directory from the official runner")
    parser.add_argument("--steps", "--max-steps", dest="steps", type=int, default=1500, help="Maximum episode steps")
    parser.add_argument("--record", type=str, default=None, help="Optional output MP4/GIF path for headless recording")
    parser.add_argument("--render_mode", choices=["human", "rgb_array"], default=None, help="Override render mode (defaults to rgb_array with --record, else human)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS when --record is set")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()

    if args.manual:
        run_manual(args.stage, args.steps)
    elif args.official_harl and args.official_mat:
        print("Choose only one of --official-harl or --official-mat.", file=sys.stderr)
        sys.exit(1)
    elif args.official_harl:
        if args.stage not in (3, 4):
            print("--official-harl requires --stage 3 or 4.", file=sys.stderr)
            sys.exit(1)
        run_official_harl(
            args.stage,
            args.official_harl,
            args.steps,
            args.record,
            args.fps,
            args.seed,
            args.render_mode,
        )
    elif args.official_mat:
        if args.stage not in (3, 4):
            print("--official-mat requires --stage 3 or 4.", file=sys.stderr)
            sys.exit(1)
        run_official_mat(
            args.stage,
            args.official_mat,
            args.steps,
            args.record,
            args.fps,
            args.seed,
            args.render_mode,
        )
    else:
        if not args.model:
            print("Provide --manual, --model, --official-harl, or --official-mat.", file=sys.stderr)
            sys.exit(1)
        run_policy(
            args.stage,
            args.model,
            args.steps,
            args.record,
            args.fps,
            args.seed,
            args.render_mode,
        )


if __name__ == "__main__":
    main()
