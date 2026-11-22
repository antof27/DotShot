"""Headless simulation engine for DotShot Game v2 (shared by Gymnasium and PettingZoo wrappers)."""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dotshot.core.arena import (
    compute_lidar_8,
    has_line_of_sight,
    point_in_obstacles,
    resolve_circle_obstacles,
)
from dotshot.core.config import (
    ELEMENT_COUNTERS,
    ELEMENT_FIRE,
    ELEMENT_GRASS,
    ELEMENT_WATER,
    ELEMENT_WEAKNESS,
    ROLE_SUPPORT,
    ROLE_VANGUARD,
    StageConfig,
    get_stage_config,
)
from dotshot.core.entities import (
    AgentState,
    CentralCore,
    Enemy,
    EnemyProjectile,
    NeutralAlly,
    Projectile,
)
from dotshot.core.rewards import (
    AgentStepEvents,
    TeamStepEvents,
    compute_decomposed_rewards,
)


OBS_DIM = 129
TOP_K_HOSTILES = 5
TOP_M_TEAMMATES = 3


class DotShotEngine:
    """Deterministic headless physics and Dec-POMDP state engine for DotShot v2."""

    def __init__(
        self,
        stage: int = 1,
        partial_obs: Optional[bool] = None,
        reward_alpha: Optional[float] = None,
        max_steps: Optional[int] = None,
    ):
        base_cfg = get_stage_config(stage)
        self.cfg: StageConfig = replace(
            base_cfg,
            partial_obs=base_cfg.partial_obs if partial_obs is None else partial_obs,
            reward_alpha=base_cfg.reward_alpha if reward_alpha is None else reward_alpha,
            max_steps=base_cfg.max_steps if max_steps is None else max_steps,
        )

        self.np_random: np.random.Generator = np.random.default_rng(0)
        self.obs_dim = OBS_DIM
        self.state_dim = self.cfg.num_agents * 14 + 5 + self.cfg.max_enemies * 15 + 4

        self.agents: List[AgentState] = []
        self.core: Optional[CentralCore] = None
        self.enemies: List[Enemy] = []
        self.bullets: List[Projectile] = []
        self.enemy_bullets: List[EnemyProjectile] = []
        self.neutral_allies: List[NeutralAlly] = []
        self.visual_events: List[Dict[str, Any]] = []

        self._uid_counter = 0
        self.step_count = 0
        self.spawn_timer = 0
        self.ally_spawn_timer = 180
        self.wave_index = 1
        self.difficulty = 1.0
        self.kills_since_boss = 0
        self.last_damage_sources: Dict[int, set] = {}
        self.episode_misses: Dict[int, int] = {}
        self.telemetry: Dict[str, float] = {}

    def _next_uid(self) -> int:
        self._uid_counter += 1
        return self._uid_counter

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self._uid_counter = 0
        self.step_count = 0
        self.spawn_timer = 10
        self.ally_spawn_timer = int(self.np_random.integers(120, 240))
        self.wave_index = 1
        self.difficulty = 1.0
        self.kills_since_boss = 0
        self.last_damage_sources = {}
        self.visual_events = []

        cx, cy = self.cfg.width * 0.5, self.cfg.height * 0.5
        self.core = CentralCore(x=cx, y=cy, max_hp=self.cfg.core_max_hp, hp=self.cfg.core_max_hp) if self.cfg.has_core else None

        # Spawn agents safely around center/core
        self.agents = []
        self.episode_misses = {}
        n = self.cfg.num_agents
        for idx in range(n):
            role = self.cfg.agent_roles[idx]
            if n == 1:
                ax, ay = cx, cy
            else:
                angle = (2.0 * np.pi * idx) / n
                ax = cx + np.cos(angle) * 115.0
                ay = cy + np.sin(angle) * 115.0
                ax, ay, _ = resolve_circle_obstacles(
                    ax, ay, 22.0, self.cfg.obstacles, self.cfg.width, self.cfg.height
                )
            self.agents.append(AgentState(idx=idx, x=ax, y=ay, role_id=role))
            self.episode_misses[idx] = 0

        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.neutral_allies = []

        self.telemetry = {
            "score": 0.0,
            "kills": 0.0,
            "boss_kills": 0.0,
            "shield_breaks": 0.0,
            "combo_kills": 0.0,
            "shots_fired": 0.0,
            "hits_correct": 0.0,
            "hits_wrong": 0.0,
            "ally_hits": 0.0,
            "damage_dealt": 0.0,
            "damage_taken": 0.0,
            "healing_done": 0.0,
            "marks_applied": 0.0,
            "win": 0.0,
        }

        # Seed initial enemies on reset so observations are populated immediately
        initial_enemies = min(2, self.cfg.max_enemies)
        for _ in range(initial_enemies):
            self._try_spawn_enemy(force=True)

    def _spawn_position_on_edge(self) -> Tuple[float, float]:
        for _ in range(12):
            edge = int(self.np_random.integers(0, 4))
            if edge == 0:
                x, y = float(self.np_random.uniform(30.0, self.cfg.width - 30.0)), 24.0
            elif edge == 1:
                x, y = self.cfg.width - 24.0, float(self.np_random.uniform(30.0, self.cfg.height - 30.0))
            elif edge == 2:
                x, y = float(self.np_random.uniform(30.0, self.cfg.width - 30.0)), self.cfg.height - 24.0
            else:
                x, y = 24.0, float(self.np_random.uniform(30.0, self.cfg.height - 30.0))

            if point_in_obstacles(x, y, 22.0, self.cfg.obstacles):
                continue

            min_dist = min(
                np.hypot(x - a.x, y - a.y) for a in self.agents if a.alive
            ) if any(a.alive for a in self.agents) else 999.0
            if self.core is not None:
                min_dist = min(min_dist, float(np.hypot(x - self.core.x, y - self.core.y)))
            if min_dist >= 220.0:
                return x, y
        return 24.0, 24.0

    def _try_spawn_enemy(self, force: bool = False) -> None:
        if len(self.enemies) >= self.cfg.max_enemies:
            return
        if not force and self.spawn_timer > 0:
            return

        x, y = self._spawn_position_on_edge()
        etype = int(self.cfg.allowed_enemy_types[int(self.np_random.integers(0, len(self.cfg.allowed_enemy_types)))])

        has_active_boss = any(e.is_boss and e.alive for e in self.enemies)
        spawn_boss = (
            self.cfg.boss_enabled
            and not has_active_boss
            and self.kills_since_boss >= self.cfg.boss_kill_interval
        )
        if spawn_boss:
            self.kills_since_boss = 0

        shoot_cd = int(self.np_random.integers(70, 120))
        enemy = Enemy(
            uid=self._next_uid(),
            x=x,
            y=y,
            enemy_type=etype,
            speed_mult=self.cfg.enemy_speed_mult,
            difficulty=self.difficulty,
            is_boss=spawn_boss,
            shoot_cooldown_init=shoot_cd,
        )
        self.enemies.append(enemy)
        self.last_damage_sources[enemy.uid] = set()
        self.spawn_timer = max(14, int(self.cfg.spawn_cooldown / self.difficulty))
        if spawn_boss:
            self.visual_events.append({"type": "callout", "x": x, "y": y - 24, "text": f"{enemy.element_name.upper()} BOSS!", "color": (250, 204, 21)})

    def _try_spawn_neutral_ally(self) -> None:
        if not self.cfg.spawn_allies:
            return
        self.ally_spawn_timer -= 1
        if self.ally_spawn_timer <= 0 and len(self.neutral_allies) < 2:
            side = int(self.np_random.integers(0, 4))
            if side == 0:
                sx, sy = 15.0, float(self.np_random.uniform(140.0, self.cfg.height - 140.0))
                tx, ty = self.cfg.width - 15.0, float(self.np_random.uniform(140.0, self.cfg.height - 140.0))
            elif side == 1:
                sx, sy = self.cfg.width - 15.0, float(self.np_random.uniform(140.0, self.cfg.height - 140.0))
                tx, ty = 15.0, float(self.np_random.uniform(140.0, self.cfg.height - 140.0))
            elif side == 2:
                sx, sy = float(self.np_random.uniform(160.0, self.cfg.width - 160.0)), 15.0
                tx, ty = float(self.np_random.uniform(160.0, self.cfg.width - 160.0)), self.cfg.height - 15.0
            else:
                sx, sy = float(self.np_random.uniform(160.0, self.cfg.width - 160.0)), self.cfg.height - 15.0
                tx, ty = float(self.np_random.uniform(160.0, self.cfg.width - 160.0)), 15.0

            self.neutral_allies.append(NeutralAlly(sx, sy, tx, ty, self.np_random))
            self.ally_spawn_timer = int(self.np_random.integers(180, 320))

    def _is_enemy_visible(self, agent: AgentState, enemy: Enemy) -> bool:
        dist = float(np.hypot(enemy.x - agent.x, enemy.y - agent.y))
        if not self.cfg.partial_obs:
            return True
        if dist > self.cfg.vision_range:
            return False
        return has_line_of_sight(agent.x, agent.y, enemy.x, enemy.y, self.cfg.obstacles)

    def _find_ray_target_uid(self, agent: AgentState, dx: float, dy: float) -> Optional[int]:
        best_dist = 9999.0
        best_uid: Optional[int] = None
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            ex, ey = enemy.x - agent.x, enemy.y - agent.y
            proj = ex * dx + ey * dy
            if proj <= 0.0:
                continue
            perp = abs(ex * (-dy) + ey * dx)
            if perp <= enemy.radius + 14.0 and proj < best_dist:
                if has_line_of_sight(agent.x, agent.y, enemy.x, enemy.y, self.cfg.obstacles):
                    best_dist = proj
                    best_uid = enemy.uid
        return best_uid

    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, float], bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.visual_events = []

        if self.core is not None and self.core.under_attack_timer > 0:
            self.core.under_attack_timer -= 1

        agent_events: Dict[int, AgentStepEvents] = {
            a.idx: AgentStepEvents(alive=a.alive) for a in self.agents
        }
        team_events = TeamStepEvents(core_alive=(self.core.alive if self.core is not None else True))

        # 1. Process agent actions (movement, aim, shooting, class abilities)
        for agent in self.agents:
            ev = agent_events[agent.idx]
            if not agent.alive:
                continue

            if agent.fire_cooldown > 0:
                agent.fire_cooldown -= 1
            if agent.ability_cooldown > 0:
                agent.ability_cooldown -= 1
            if agent.shield_active_timer > 0:
                agent.shield_active_timer -= 1
            if agent.repair_pulse_timer > 0:
                agent.repair_pulse_timer -= 1

            act = np.asarray(actions[agent.idx], dtype=np.int64)
            move_x = float(act[0] - 1)
            move_y = float(act[1] - 1)
            mag = float(np.hypot(move_x, move_y))
            if mag > 0.0:
                vx = (move_x / mag) * agent.speed
                vy = (move_y / mag) * agent.speed
            else:
                vx, vy = 0.0, 0.0

            agent.vx, agent.vy = vx, vy
            nx, ny, hit_wall = resolve_circle_obstacles(
                agent.x + vx,
                agent.y + vy,
                agent.radius,
                self.cfg.obstacles,
                self.cfg.width,
                self.cfg.height,
            )
            agent.x, agent.y = nx, ny
            if hit_wall:
                ev.wall_bump += 0.25

            aim_idx = int(act[2]) % self.cfg.aim_bins
            angle = aim_idx * (2.0 * np.pi / self.cfg.aim_bins)
            aim_dx = float(np.cos(angle))
            aim_dy = float(-np.sin(angle))
            agent.aim_dx, agent.aim_dy = aim_dx, aim_dy

            # Aim alignment check against nearest visible enemy
            visible_enemies = [e for e in self.enemies if e.alive and self._is_enemy_visible(agent, e)]
            best_dot = -1.0
            nearest_visible: Optional[Enemy] = None
            if visible_enemies:
                nearest_visible = min(visible_enemies, key=lambda e: np.hypot(e.x - agent.x, e.y - agent.y))
                tx, ty = nearest_visible.x - agent.x, nearest_visible.y - agent.y
                td = float(np.hypot(tx, ty))
                if td > 1e-6:
                    best_dot = (tx * aim_dx + ty * aim_dy) / td
                if best_dot > 0.90:
                    ev.aim_alignment += 0.15

            weapon_action = int(act[3])
            if weapon_action == 0:
                # Hold fire -> trigger discipline bonus if poorly aimed, or Stage 4 class ability
                if visible_enemies and best_dot < 0.70:
                    ev.trigger_discipline += 0.08
                if self.cfg.stage == 4 and agent.ability_cooldown == 0:
                    self._trigger_class_ability(agent, ev, team_events)
            else:
                weapon_type = weapon_action - 1  # 0: Water, 1: Grass, 2: Fire
                if agent.fire_cooldown == 0:
                    target_uid = self._find_ray_target_uid(agent, aim_dx, aim_dy)
                    in_flight_same_target = (
                        target_uid is not None
                        and any(b.active and b.owner_idx == agent.idx and b.target_uid == target_uid for b in self.bullets)
                    )
                    if not in_flight_same_target:
                        well_aimed = best_dot >= 0.88
                        bx = agent.x + aim_dx * (agent.radius + 6.0)
                        by = agent.y + aim_dy * (agent.radius + 6.0)
                        self.bullets.append(
                            Projectile(
                                x=bx,
                                y=by,
                                dx=aim_dx,
                                dy=aim_dy,
                                element_type=weapon_type,
                                owner_idx=agent.idx,
                                damage=agent.bullet_damage,
                                speed=agent.bullet_speed,
                                target_uid=target_uid,
                                well_aimed=well_aimed,
                            )
                        )
                        agent.fire_cooldown = agent.fire_cooldown_max
                        agent.last_weapon = weapon_type
                        self.telemetry["shots_fired"] += 1.0

        # 2. Update bullets & check obstacle / enemy / neutral ally collisions
        for b in self.bullets:
            if not b.active:
                continue
            prev_x, prev_y = b.x, b.y
            b.step()

            if b.x < 0 or b.x > self.cfg.width or b.y < 0 or b.y > self.cfg.height:
                b.active = False
                ev = agent_events[b.owner_idx]
                if b.well_aimed:
                    ev.well_aimed_misses += 1
                else:
                    ev.missed_bullets += 1
                    self.episode_misses[b.owner_idx] = self.episode_misses.get(b.owner_idx, 0) + 1
                continue

            if not has_line_of_sight(prev_x, prev_y, b.x, b.y, self.cfg.obstacles):
                b.active = False
                self.visual_events.append({"type": "spark", "x": b.x, "y": b.y, "element": b.element_type, "count": 5})
                continue

            # Check neutral allies (Stage 2)
            hit_neutral = False
            for ally in self.neutral_allies:
                if not ally.alive:
                    continue
                if np.hypot(b.x - ally.x, b.y - ally.y) <= b.radius + ally.radius:
                    b.active = False
                    ally.take_damage(b.damage)
                    ev = agent_events[b.owner_idx]
                    ev.ally_hits += 1
                    self.telemetry["ally_hits"] += 1.0
                    if not ally.alive:
                        ev.ally_kills += 1
                    hit_neutral = True
                    self.visual_events.append({"type": "spark", "x": b.x, "y": b.y, "element": 0, "count": 8})
                    break
            if hit_neutral:
                continue

            # Check enemies
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                if np.hypot(b.x - enemy.x, b.y - enemy.y) <= b.radius + enemy.radius:
                    b.active = False
                    ev = agent_events[b.owner_idx]
                    owner = self.agents[b.owner_idx]
                    is_effective = ELEMENT_COUNTERS[b.element_type] == enemy.enemy_type

                    if is_effective:
                        ev.hits_correct += 1
                        self.telemetry["hits_correct"] += 1.0
                        if b.well_aimed:
                            ev.precision_hits += 1
                        self.last_damage_sources.setdefault(enemy.uid, set()).add(b.owner_idx)

                        if enemy.is_boss and enemy.shield_hp > 0.0:
                            shattered = enemy.take_shield_damage(b.damage * 1.15)
                            self.telemetry["damage_dealt"] += b.damage * 1.15
                            if shattered:
                                ev.shield_breaks += 1
                                team_events.shields_broken += 1
                                self.telemetry["shield_breaks"] += 1.0
                                self.telemetry["score"] += 80.0
                                self.visual_events.append(
                                    {"type": "callout", "x": enemy.x, "y": enemy.y - 36, "text": "SHIELD BREAK!", "color": (56, 189, 248)}
                                )
                        else:
                            dmg = b.damage * (owner.shattered_bonus_mult if (enemy.is_boss and enemy.shield_broken_timer > 0) else 1.0)
                            dealt = enemy.take_hull_damage(dmg)
                            self.telemetry["damage_dealt"] += dealt

                        self.visual_events.append({"type": "spark", "x": b.x, "y": b.y, "element": b.element_type, "count": 9})

                        if not enemy.alive:
                            ev.kills += 1
                            team_events.enemies_killed += 1
                            self.telemetry["kills"] += 1.0
                            self.kills_since_boss += 1
                            self.telemetry["score"] += 250.0 if enemy.is_boss else 100.0
                            if enemy.is_boss:
                                ev.boss_kills += 1
                                team_events.bosses_killed += 1
                                self.telemetry["boss_kills"] += 1.0
                                self.visual_events.append({"type": "shake", "intensity": 6})
                            # Synergy combo if >=2 teammates damaged, or marked, or shattered boss
                            sources = self.last_damage_sources.get(enemy.uid, set())
                            if len(sources) >= 2 or enemy.marked_timer > 0 or (enemy.is_boss and enemy.shield_broken_timer > 0):
                                ev.combo_kills += 1
                                team_events.combo_kills += 1
                                self.telemetry["combo_kills"] += 1.0
                                self.visual_events.append(
                                    {"type": "callout", "x": enemy.x, "y": enemy.y - 20, "text": "COMBO KILL!", "color": (34, 197, 94)}
                                )
                    else:
                        ev.hits_wrong += 1
                        self.telemetry["hits_wrong"] += 1.0
                        self.visual_events.append({"type": "callout", "x": enemy.x, "y": enemy.y - 16, "text": "IMMUNE", "color": (148, 163, 184)})
                    break

        # 3. Update neutral allies & enemies
        for ally in self.neutral_allies:
            if ally.alive:
                ally.step(self.cfg.width, self.cfg.height)

        for idx, enemy in enumerate(self.enemies):
            if not enemy.alive:
                continue
            if enemy.marked_timer > 0:
                enemy.marked_timer -= 1
            if enemy.shield_broken_timer > 0:
                enemy.shield_broken_timer -= 1

            tx, ty = self._select_enemy_target(enemy, idx)
            dx, dy = tx - enemy.x, ty - enemy.y
            dist = float(np.hypot(dx, dy))
            if dist > 1e-6:
                enemy.vx = (dx / dist) * enemy.speed
                enemy.vy = (dy / dist) * enemy.speed
            else:
                enemy.vx, enemy.vy = 0.0, 0.0

            ex, ey, _ = resolve_circle_obstacles(
                enemy.x + enemy.vx,
                enemy.y + enemy.vy,
                enemy.radius,
                self.cfg.obstacles,
                self.cfg.width,
                self.cfg.height,
            )
            enemy.x, enemy.y = ex, ey

            # Enemy ranged fire (Stage 2+)
            if self.cfg.enemy_can_shoot:
                enemy.shoot_cooldown -= 1
                if enemy.shoot_cooldown <= 0 and dist < 560.0 and has_line_of_sight(enemy.x, enemy.y, tx, ty, self.cfg.obstacles):
                    if dist > 1e-6:
                        self.enemy_bullets.append(
                            EnemyProjectile(
                                x=enemy.x,
                                y=enemy.y,
                                dx=dx / dist,
                                dy=dy / dist,
                                element_type=enemy.enemy_type,
                                damage=14.0 if enemy.is_boss else 10.0,
                            )
                        )
                    enemy.shoot_cooldown = int(self.np_random.integers(75, 130))

            # Contact damage to agents & Core
            for agent in self.agents:
                if not agent.alive:
                    continue
                d_ag = float(np.hypot(agent.x - enemy.x, agent.y - enemy.y))
                if d_ag < 140.0:
                    agent_events[agent.idx].proximity_danger += (140.0 - d_ag) / 140.0 * 0.25
                if d_ag <= agent.radius + enemy.radius:
                    raw_dmg = enemy.contact_damage_rate * 0.04
                    taken = agent.take_damage(raw_dmg)
                    agent_events[agent.idx].damage_taken += taken
                    if agent.shield_active_timer > 0:
                        agent_events[agent.idx].shield_absorbed += max(0.0, raw_dmg - taken)
                    self.telemetry["damage_taken"] += taken
                    if not agent.alive:
                        agent_events[agent.idx].died = True
                        team_events.any_agent_died = True

            if self.core is not None and self.core.alive:
                d_core = float(np.hypot(self.core.x - enemy.x, self.core.y - enemy.y))
                if d_core <= self.core.radius + enemy.radius:
                    taken = self.core.take_damage(enemy.contact_damage_rate * 0.045)
                    team_events.core_damage_taken += taken

        # 4. Update enemy projectiles
        for eb in self.enemy_bullets:
            if not eb.active:
                continue
            prev_x, prev_y = eb.x, eb.y
            eb.step()
            if eb.x < 0 or eb.x > self.cfg.width or eb.y < 0 or eb.y > self.cfg.height:
                eb.active = False
                continue
            if not has_line_of_sight(prev_x, prev_y, eb.x, eb.y, self.cfg.obstacles):
                eb.active = False
                continue

            if self.core is not None and self.core.alive:
                if np.hypot(eb.x - self.core.x, eb.y - self.core.y) <= eb.radius + self.core.radius:
                    eb.active = False
                    taken = self.core.take_damage(eb.damage)
                    team_events.core_damage_taken += taken
                    continue

            for agent in self.agents:
                if not agent.alive:
                    continue
                if np.hypot(eb.x - agent.x, eb.y - agent.y) <= eb.radius + agent.radius:
                    eb.active = False
                    taken = agent.take_damage(eb.damage)
                    agent_events[agent.idx].damage_taken += taken
                    if agent.shield_active_timer > 0:
                        agent_events[agent.idx].shield_absorbed += max(0.0, eb.damage - taken)
                    self.telemetry["damage_taken"] += taken
                    if not agent.alive:
                        agent_events[agent.idx].died = True
                        team_events.any_agent_died = True
                    break

        # 5. Cleanup & spawning
        self.bullets = [b for b in self.bullets if b.active]
        self.enemy_bullets = [eb for eb in self.enemy_bullets if eb.active]
        self.enemies = [e for e in self.enemies if e.alive]
        self.neutral_allies = [a for a in self.neutral_allies if a.alive]

        self.spawn_timer -= 1
        self._try_spawn_enemy(force=False)
        self._try_spawn_neutral_ally()

        new_wave = 1 + int(self.telemetry["kills"] // 10)
        if new_wave > self.wave_index:
            self.wave_index = new_wave
            self.difficulty = min(2.4, 1.0 + (self.wave_index - 1) * 0.14)
            team_events.wave_cleared = True
            self.visual_events.append({"type": "callout", "x": self.cfg.width * 0.5, "y": 90, "text": f"WAVE {self.wave_index}", "color": (248, 250, 252)})

        # 6. Termination & Truncation
        all_agents_dead = not any(a.alive for a in self.agents)
        core_destroyed = self.core is not None and not self.core.alive
        team_events.core_destroyed = core_destroyed
        team_events.core_alive = self.core.alive if self.core is not None else True

        terminated = bool(all_agents_dead or core_destroyed)
        truncated = bool((not terminated) and (self.step_count >= self.cfg.max_steps))
        if truncated:
            self.telemetry["win"] = 1.0

        rewards, breakdown = compute_decomposed_rewards(
            agent_events=agent_events,
            team_events=team_events,
            alpha=self.cfg.reward_alpha,
            num_agents=self.cfg.num_agents,
            episode_misses=self.episode_misses,
        )

        shots = max(1.0, self.telemetry["shots_fired"])
        total_hits = self.telemetry["hits_correct"] + self.telemetry["hits_wrong"]
        info: Dict[str, Any] = {
            "step": self.step_count,
            "wave": self.wave_index,
            "score": self.telemetry["score"],
            "kills": self.telemetry["kills"],
            "boss_kills": self.telemetry["boss_kills"],
            "shield_breaks": self.telemetry["shield_breaks"],
            "synergy_combos": self.telemetry["combo_kills"],
            "hit_accuracy": total_hits / shots,
            "elemental_accuracy": self.telemetry["hits_correct"] / max(1.0, total_hits),
            "hits_correct": self.telemetry["hits_correct"],
            "hits_wrong": self.telemetry["hits_wrong"],
            "shots_fired": self.telemetry["shots_fired"],
            "core_hp_ratio": (self.core.hp / self.core.max_hp) if self.core is not None else 1.0,
            "win": self.telemetry["win"],
            "reward_breakdown": breakdown,
        }
        return rewards, terminated, truncated, info

    def _trigger_class_ability(self, agent: AgentState, ev: AgentStepEvents, team_events: TeamStepEvents) -> None:
        if agent.role_id == ROLE_VANGUARD:
            nearby_threat = any(
                e.alive and np.hypot(e.x - agent.x, e.y - agent.y) < 380.0 for e in self.enemies
            ) or any(
                eb.active and np.hypot(eb.x - agent.x, eb.y - agent.y) < 320.0 for eb in self.enemy_bullets
            )
            if nearby_threat:
                agent.shield_active_timer = 90
                agent.ability_cooldown = agent.ability_cooldown_max
                self.visual_events.append({"type": "callout", "x": agent.x, "y": agent.y - 24, "text": "AEGIS ON", "color": (56, 189, 248)})
        elif agent.role_id == ROLE_SUPPORT:
            healed_total = 0.0
            marked_total = 0
            needs_heal = (self.core is not None and self.core.hp < self.core.max_hp * 0.96) or any(
                a.alive and a.hp < a.max_hp * 0.96 for a in self.agents
            )
            nearby_enemies = [e for e in self.enemies if e.alive and np.hypot(e.x - agent.x, e.y - agent.y) <= 440.0]
            if needs_heal or nearby_enemies:
                if self.core is not None and self.core.alive and np.hypot(self.core.x - agent.x, self.core.y - agent.y) <= 420.0:
                    h = self.core.heal(26.0)
                    healed_total += h
                    team_events.core_healed += h
                for ally in self.agents:
                    if ally.alive and np.hypot(ally.x - agent.x, ally.y - agent.y) <= 400.0:
                        healed_total += ally.heal(22.0)
                for enemy in sorted(nearby_enemies, key=lambda e: np.hypot(e.x - agent.x, e.y - agent.y))[:2]:
                    if enemy.marked_timer == 0:
                        marked_total += 1
                    enemy.marked_timer = 150
                agent.ability_cooldown = agent.ability_cooldown_max
                agent.repair_pulse_timer = 22
                ev.heal_utility += healed_total
                ev.mark_utility += marked_total
                self.telemetry["healing_done"] += healed_total
                self.telemetry["marks_applied"] += marked_total
                self.visual_events.append({"type": "callout", "x": agent.x, "y": agent.y - 24, "text": "PULSE", "color": (34, 197, 94)})

    def _select_enemy_target(self, enemy: Enemy, enemy_idx: int) -> Tuple[float, float]:
        # Vanguard with active shield pulls enemies within 420px
        for agent in self.agents:
            if agent.alive and agent.role_id == ROLE_VANGUARD and agent.shield_active_timer > 0:
                if np.hypot(agent.x - enemy.x, agent.y - enemy.y) <= 420.0:
                    return agent.x, agent.y

        # In Core-defense modes, ~45% of standard enemies pressure the Core directly
        if self.core is not None and self.core.alive and (enemy.uid % 2 == 0):
            return self.core.x, self.core.y

        alive_agents = [a for a in self.agents if a.alive]
        if alive_agents:
            nearest = min(alive_agents, key=lambda a: np.hypot(a.x - enemy.x, a.y - enemy.y))
            return nearest.x, nearest.y
        if self.core is not None:
            return self.core.x, self.core.y
        return self.cfg.width * 0.5, self.cfg.height * 0.5

    def get_local_obs(self, agent_idx: int) -> np.ndarray:
        """Build 129-dim normalized observation vector in [-1.0, 1.0] for agent_idx."""
        agent = self.agents[agent_idx]
        w, h = float(self.cfg.width), float(self.cfg.height)
        diag = float(np.hypot(w, h))
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if not agent.alive:
            return obs

        ptr = 0
        # 1. Agent Self-State (14)
        obs[ptr] = (agent.x / w) * 2.0 - 1.0
        obs[ptr + 1] = (agent.y / h) * 2.0 - 1.0
        obs[ptr + 2] = agent.vx / max(1.0, agent.speed)
        obs[ptr + 3] = agent.vy / max(1.0, agent.speed)
        obs[ptr + 4] = (agent.hp / agent.max_hp) * 2.0 - 1.0
        obs[ptr + 5] = (agent.fire_cooldown / max(1, agent.fire_cooldown_max)) * 2.0 - 1.0
        obs[ptr + 6] = (agent.ability_cooldown / max(1, agent.ability_cooldown_max)) * 2.0 - 1.0
        obs[ptr + 7] = 1.0
        if 0 <= agent.last_weapon <= 2:
            obs[ptr + 8 + agent.last_weapon] = 1.0
        obs[ptr + 11 + agent.role_id] = 1.0
        ptr += 14

        # 2. Central Core State (5)
        if self.core is not None:
            dx = (self.core.x - agent.x) / w
            dy = (self.core.y - agent.y) / h
            dist = float(np.hypot(self.core.x - agent.x, self.core.y - agent.y)) / diag
            obs[ptr] = dx
            obs[ptr + 1] = dy
            obs[ptr + 2] = dist * 2.0 - 1.0
            obs[ptr + 3] = (self.core.hp / self.core.max_hp) * 2.0 - 1.0
            obs[ptr + 4] = 1.0 if self.core.under_attack_timer > 0 else -1.0
        ptr += 5

        # 3. 8-Ray Obstacle / Wall LIDAR (8)
        obs[ptr:ptr + 8] = compute_lidar_8(agent.x, agent.y, self.cfg.obstacles, w, h, max_range=480.0)
        ptr += 8

        # 4. Top-K Visible Hostiles (5 * 15 = 75)
        visible = [e for e in self.enemies if e.alive and self._is_enemy_visible(agent, e)]
        visible.sort(key=lambda e: (e.x - agent.x) ** 2 + (e.y - agent.y) ** 2)
        for k in range(TOP_K_HOSTILES):
            if k < len(visible):
                e = visible[k]
                rx = (e.x - agent.x) / w
                ry = (e.y - agent.y) / h
                ang = float(np.arctan2(-(e.y - agent.y), e.x - agent.x))
                dist = float(np.hypot(e.x - agent.x, e.y - agent.y)) / diag
                obs[ptr] = rx
                obs[ptr + 1] = ry
                obs[ptr + 2] = float(np.sin(ang))
                obs[ptr + 3] = float(np.cos(ang))
                obs[ptr + 4] = dist * 2.0 - 1.0
                obs[ptr + 5] = float(np.clip(e.vx / 4.5, -1.0, 1.0))
                obs[ptr + 6] = float(np.clip(e.vy / 4.5, -1.0, 1.0))
                obs[ptr + 7] = (e.hp / e.max_hp) * 2.0 - 1.0
                obs[ptr + 8] = ((e.shield_hp / e.max_shield) * 2.0 - 1.0) if e.max_shield > 0.0 else -1.0
                obs[ptr + 9 + e.enemy_type] = 1.0
                obs[ptr + 12] = 1.0 if e.is_boss else -1.0
                obs[ptr + 13] = 1.0 if e.marked_timer > 0 else -1.0
                obs[ptr + 14] = 1.0
            ptr += 15

        # 5. Teammates & Neutral Allies (3 * 9 = 27)
        others: List[Tuple[float, float, float, float, int, bool]] = []
        for other in self.agents:
            if other.idx == agent_idx:
                continue
            d = float(np.hypot(other.x - agent.x, other.y - agent.y))
            others.append((d, other.x, other.y, other.hp / other.max_hp if other.max_hp > 0 else 0.0, other.role_id, other.alive))
        for ally in self.neutral_allies:
            if not ally.alive:
                continue
            d = float(np.hypot(ally.x - agent.x, ally.y - agent.y))
            others.append((d, ally.x, ally.y, ally.hp / ally.max_hp, 3, True))
        others.sort(key=lambda item: item[0])

        for m in range(TOP_M_TEAMMATES):
            if m < len(others):
                d, ox, oy, hp_r, role_slot, is_alive = others[m]
                obs[ptr] = (ox - agent.x) / w
                obs[ptr + 1] = (oy - agent.y) / h
                obs[ptr + 2] = (d / diag) * 2.0 - 1.0
                obs[ptr + 3] = hp_r * 2.0 - 1.0
                obs[ptr + 4 + role_slot] = 1.0
                obs[ptr + 8] = 1.0 if is_alive else -1.0
            ptr += 9

        return np.clip(obs, -1.0, 1.0)

    def get_global_state(self) -> np.ndarray:
        """Privileged noise-free CTDE global state s^t in [-1.0, 1.0]."""
        w, h = float(self.cfg.width), float(self.cfg.height)
        diag = float(np.hypot(w, h))
        state = np.zeros(self.state_dim, dtype=np.float32)
        ptr = 0

        for agent in self.agents:
            if agent.alive:
                state[ptr] = (agent.x / w) * 2.0 - 1.0
                state[ptr + 1] = (agent.y / h) * 2.0 - 1.0
                state[ptr + 2] = agent.vx / max(1.0, agent.speed)
                state[ptr + 3] = agent.vy / max(1.0, agent.speed)
                state[ptr + 4] = (agent.hp / agent.max_hp) * 2.0 - 1.0
                state[ptr + 5] = (agent.fire_cooldown / max(1, agent.fire_cooldown_max)) * 2.0 - 1.0
                state[ptr + 6] = (agent.ability_cooldown / max(1, agent.ability_cooldown_max)) * 2.0 - 1.0
                state[ptr + 7] = 1.0
                if 0 <= agent.last_weapon <= 2:
                    state[ptr + 8 + agent.last_weapon] = 1.0
                state[ptr + 11 + agent.role_id] = 1.0
            else:
                state[ptr + 7] = -1.0
                state[ptr + 11 + agent.role_id] = 1.0
            ptr += 14

        if self.core is not None:
            state[ptr] = (self.core.x / w) * 2.0 - 1.0
            state[ptr + 1] = (self.core.y / h) * 2.0 - 1.0
            state[ptr + 2] = (self.core.hp / self.core.max_hp) * 2.0 - 1.0
            state[ptr + 3] = 1.0 if self.core.under_attack_timer > 0 else -1.0
            state[ptr + 4] = 1.0 if self.core.alive else -1.0
        ptr += 5

        cx, cy = w * 0.5, h * 0.5
        active_enemies = sorted(
            [e for e in self.enemies if e.alive],
            key=lambda e: (e.x - cx) ** 2 + (e.y - cy) ** 2,
        )
        for k in range(self.cfg.max_enemies):
            if k < len(active_enemies):
                e = active_enemies[k]
                state[ptr] = (e.x / w) * 2.0 - 1.0
                state[ptr + 1] = (e.y / h) * 2.0 - 1.0
                state[ptr + 2] = float(np.clip(e.vx / 4.5, -1.0, 1.0))
                state[ptr + 3] = float(np.clip(e.vy / 4.5, -1.0, 1.0))
                state[ptr + 4] = float(np.hypot(e.x - cx, e.y - cy) / diag) * 2.0 - 1.0
                state[ptr + 5] = (e.hp / e.max_hp) * 2.0 - 1.0
                state[ptr + 6] = ((e.shield_hp / e.max_shield) * 2.0 - 1.0) if e.max_shield > 0.0 else -1.0
                state[ptr + 7 + e.enemy_type] = 1.0
                state[ptr + 10] = 1.0 if e.is_boss else -1.0
                state[ptr + 11] = 1.0 if e.marked_timer > 0 else -1.0
                state[ptr + 12] = float(ELEMENT_WEAKNESS[e.enemy_type]) / 2.0
                state[ptr + 13] = 1.0 if e.shield_broken_timer > 0 else -1.0
                state[ptr + 14] = 1.0
            ptr += 15

        state[ptr] = float(np.clip(self.wave_index / 10.0, 0.0, 1.0)) * 2.0 - 1.0
        state[ptr + 1] = (len(active_enemies) / max(1, self.cfg.max_enemies)) * 2.0 - 1.0
        state[ptr + 2] = 1.0 if any(e.is_boss for e in active_enemies) else -1.0
        state[ptr + 3] = (self.step_count / max(1, self.cfg.max_steps)) * 2.0 - 1.0

        return np.clip(state, -1.0, 1.0)
