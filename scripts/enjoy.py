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


def run_policy(stage: int, model_path: str, max_steps: int, record_path: str | None, fps: int, seed: int) -> None:
    import torch

    render_mode = "rgb_array" if record_path else "human"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="DotShot Game v2 Player & Evaluator")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4], help="DotShot curriculum stage (1-4)")
    parser.add_argument("--manual", action="store_true", help="Play manually with keyboard and mouse")
    parser.add_argument("--model", type=str, default=None, help="Path to trained SB3 (.zip) or MARL (.pt) checkpoint")
    parser.add_argument("--steps", type=int, default=1500, help="Maximum episode steps")
    parser.add_argument("--record", type=str, default=None, help="Optional output MP4/GIF path for headless recording")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS when --record is set")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed")
    args = parser.parse_args()

    if args.manual:
        run_manual(args.stage, args.steps)
    else:
        if not args.model:
            print("Provide --manual or --model <checkpoint_path>.", file=sys.stderr)
            sys.exit(1)
        run_policy(args.stage, args.model, args.steps, args.record, args.fps, args.seed)


if __name__ == "__main__":
    main()
