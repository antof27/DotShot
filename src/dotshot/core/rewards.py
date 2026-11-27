"""Decomposed individual and team reward engine for DotShot Game v2."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class AgentStepEvents:
    """Raw per-step combat events accumulated for one agent."""
    hits_correct: int = 0
    hits_wrong: int = 0
    precision_hits: int = 0
    kills: int = 0
    boss_kills: int = 0
    shield_breaks: int = 0
    combo_kills: int = 0
    missed_bullets: int = 0
    well_aimed_misses: int = 0
    ally_hits: int = 0
    ally_kills: int = 0
    damage_taken: float = 0.0
    shield_absorbed: float = 0.0
    heal_utility: float = 0.0
    mark_utility: int = 0
    aim_alignment: float = 0.0
    fire_element_credit: float = 0.0  # raw, already scaled by StageConfig.fire_element_credit
    trigger_discipline: float = 0.0
    proximity_danger: float = 0.0
    wall_bump: float = 0.0
    died: bool = False
    alive: bool = True


@dataclass
class TeamStepEvents:
    """Shared team-level tactical events per step."""
    enemies_killed: int = 0
    bosses_killed: int = 0
    shields_broken: int = 0
    combo_kills: int = 0
    core_damage_taken: float = 0.0
    core_healed: float = 0.0
    core_destroyed: bool = False
    any_agent_died: bool = False
    wave_cleared: bool = False
    core_alive: bool = True
    win_bonus: float = 0.0  # raw, StageConfig.win_bonus x Core HP ratio on victory


def compute_decomposed_rewards(
    agent_events: Dict[int, AgentStepEvents],
    team_events: TeamStepEvents,
    alpha: float,
    num_agents: int,
    episode_misses: Dict[int, int],
    team_shared: bool = False,
) -> tuple[Dict[int, float], Dict[str, float]]:
    """
    Compute per-agent rewards.

    Mixed mode (default):  r_i^t = (1 - alpha) * r_{i, indiv}^t + alpha * r_team^t
    Team mode (``team_shared``, multi-agent only): every agent, alive or dead, gets
        r^t = mean_i r_{i, indiv}^t + r_team^t
    so a single centralized value function sees one team reward (HAPPO/HATRPO assume this).

    Returns (per_agent_rewards, breakdown). ``breakdown["components"]`` holds each
    named term's contribution to the per-agent-mean reward, so the components of
    one step sum to ``mean_i r_i``.
    """
    # 1. Shared team signal (raw units; divided by 10 below)
    team_terms = {
        "team_kills": 2.5 * team_events.enemies_killed,
        "team_boss_kills": 8.0 * team_events.bosses_killed,
        "team_shield_breaks": 3.5 * team_events.shields_broken,
        "team_combo_kills": 4.0 * team_events.combo_kills,
        "team_wave_cleared": 6.0 * (1.0 if team_events.wave_cleared else 0.0),
        "team_core_healed": 0.04 * team_events.core_healed,
        "team_core_damage": -0.08 * team_events.core_damage_taken,
        "team_core_destroyed": -(12.0 if team_events.core_destroyed else 0.0),
        "team_agent_died": -(3.0 if team_events.any_agent_died else 0.0),
        "team_core_alive": 0.02 if team_events.core_alive else 0.0,
        "team_win_bonus": team_events.win_bonus,
    }
    r_team = sum(team_terms.values()) / 10.0

    rewards: Dict[int, float] = {}
    components: Dict[str, float] = {}
    indiv_sum = 0.0
    n = max(1, len(agent_events))
    team_shared = team_shared and num_agents > 1
    if team_shared:
        team_weight, indiv_weight = 1.0, 1.0
    else:
        team_weight = 0.0 if num_agents == 1 else alpha
        indiv_weight = 1.0 if num_agents == 1 else 1.0 - alpha

    for idx, ev in agent_events.items():
        if not ev.alive and not ev.died:
            rewards[idx] = 0.0
            continue

        indiv_terms = {
            "hits_correct": 2.2 * ev.hits_correct,
            "precision_hits": 0.8 * ev.precision_hits,
            "kills": 5.0 * ev.kills,
            "boss_kills": 10.0 * ev.boss_kills,
            "shield_breaks": 4.5 * ev.shield_breaks,
            "combo_kills": 5.0 * ev.combo_kills,
            "aim_shaping": 0.20 * ev.aim_alignment,
            "fire_element_credit": ev.fire_element_credit,
            "shield_absorbed": 0.05 * ev.shield_absorbed,
            "heal_utility": 0.05 * ev.heal_utility,
            "mark_utility": 0.8 * ev.mark_utility,
            "survival": 0.015 if ev.alive else 0.0,
            "hits_wrong": -0.20 * ev.hits_wrong,
            "missed_bullets": -0.04 * ev.missed_bullets,
            "ally_hits": -2.0 * ev.ally_hits,
            "ally_kills": -5.0 * ev.ally_kills,
            "damage_taken": -0.20 * ev.damage_taken,
            "proximity_danger": -0.15 * ev.proximity_danger,
            "wall_bump": -0.08 * ev.wall_bump,
            "death": -(5.0 if ev.died else 0.0),
        }
        r_indiv = sum(indiv_terms.values()) / 10.0
        indiv_sum += r_indiv
        rewards[idx] = float(indiv_weight * r_indiv + team_weight * r_team)

        for key, val in indiv_terms.items():
            components[key] = components.get(key, 0.0) + indiv_weight * val / 10.0 / n
        if not team_shared:
            for key, val in team_terms.items():
                components[key] = components.get(key, 0.0) + team_weight * val / 10.0 / n

    if team_shared:
        shared = indiv_sum / n + r_team
        rewards = {idx: float(shared) for idx in agent_events}
        for key, val in team_terms.items():
            components[key] = components.get(key, 0.0) + val / 10.0

    breakdown = {
        "r_team": float(r_team),
        "r_indiv_mean": float(indiv_sum / n),
        "components": components,
    }
    return rewards, breakdown
