import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.generator import EnemySpawner, Enemy
try:
    from env.reward_function import calculate_reward
except ImportError:
    calculate_reward = None

class Bullet:
    def __init__(self, x, y, dx, dy, bullet_type):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.bullet_type = bullet_type  # 0: Water, 1: Grass, 2: Fire
        self.speed = 10.0
        self.radius = 5.0
        self.active = True
        self.well_aimed = False
        
        if bullet_type == 0:
            self.color = (0, 100, 255)
            self.name = "Water Bullet"
        elif bullet_type == 1:
            self.color = (0, 255, 100)
            self.name = "Grass Bullet"
        else:
            self.color = (255, 100, 0)
            self.name = "Fire Bullet"

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class EnemyBullet:
    def __init__(self, x, y, dx, dy):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.speed = 5.0  # Slower than player bullets to allow dodging
        self.radius = 5.0
        self.active = True
        self.color = (255, 140, 0) # Orange for enemy bullets

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class Ally:
    def __init__(self, x, y, tx, ty):
        self.x = x
        self.y = y
        self.tx = tx
        self.ty = ty
        self.spawn_x = x
        self.spawn_y = y
        self.enemy_type = 4  # Special type index
        self.is_ally = True  # Identification flag
        self.radius = 15.0
        self.max_health = 50.0
        self.health = self.max_health
        self.speed = 1.6  # Smooth traverse speed
        self.alive = True
        self.color = (255, 255, 255) # Glowing white
        self.name = "Ally"
        self.damage_rate = 0.0 # No contact damage
        
        # Base angle from spawn to target
        self.angle = np.arctan2(ty - y, tx - x)
        
        # Determine movement axis to detect exit
        if abs(x - tx) > abs(y - ty):
            self.direction = "horizontal"
        else:
            self.direction = "vertical"

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def update(self, agent_x, agent_y):
        # Smoothly adjust angle towards target
        target_angle = np.arctan2(self.ty - self.y, self.tx - self.x)
        
        # Angle difference normalized to [-pi, pi]
        angle_diff = target_angle - self.angle
        angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
        
        # Steer towards target
        self.angle += angle_diff * 0.05
        
        # Add random walk noise to angle
        self.angle += np.random.uniform(-0.15, 0.15)
        
        # Calculate velocity
        vx = np.cos(self.angle) * self.speed
        vy = np.sin(self.angle) * self.speed
        
        self.x += vx
        self.y += vy
        
        # Exit condition: if they cross the target line/edge
        margin = self.radius + 10.0
        if self.direction == "horizontal":
            if self.spawn_x < self.tx:
                if self.x > self.tx + margin:
                    self.alive = False
            else:
                if self.x < self.tx - margin:
                    self.alive = False
        else:
            if self.spawn_y < self.ty:
                if self.y > self.ty + margin:
                    self.alive = False
            else:
                if self.y < self.ty - margin:
                    self.alive = False
                    
        # Safety boundary check
        if self.x < -100.0 or self.x > 900.0 or self.y < -100.0 or self.y > 900.0:
            self.alive = False


class ElementShooterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # --- Curriculum difficulty presets ---
    CURRICULUM = {
        1: {  # Level 1: Easy — learn to shoot one enemy type
            "max_enemies": 3,           # (was 2)
            "enemy_types": [2],         # Fire only
            "spawn_cooldown": 60,       # Spawns 1.5x faster (was 90)
            "enemy_speed_mult": 0.9,    # Faster enemies (was 0.7)
        },
        2: {  # Level 2: Medium — two enemy types, moderate pressure
            "max_enemies": 4,           # (was 3)
            "enemy_types": [0, 2],      # Water + Fire
            "spawn_cooldown": 50,       # (was 75)
            "enemy_speed_mult": 1.1,    # (was 0.85)
        },
        3: {  # Level 3: Full game & Multi-Agent — 2 agents, all 3 enemy types, projectiles, allies
            "max_enemies": 6,
            "enemy_types": [0, 1, 2],
            "spawn_cooldown": 30,
            "enemy_speed_mult": 1.25,
        },
    }

    def __init__(self, render_mode=None, width=800, height=800, curriculum_level=3):
        super(ElementShooterEnv, self).__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # --- Curriculum setup ---
        self.curriculum_level = curriculum_level
        preset = self.CURRICULUM[curriculum_level]
        self.allowed_enemy_types = preset["enemy_types"]
        self.enemy_speed_mult = preset["enemy_speed_mult"]
        
        # Game constants
        self.agent_speed = 4.0
        self.agent_radius = 15.0
        self.max_health = 100.0
        self.shoot_cooldown_max = 16   # ticks between shots (slightly faster than 20, still forces precision)
        self.max_enemies_on_screen = preset["max_enemies"]
        self.max_enemies_tracked = 5  # For RL observation (always 5 slots)
        
        # Action space:
        # action[0]: movement dx (0: Left, 1: None, 2: Right)
        # action[1]: movement dy (0: Up, 1: None, 2: Down)
        # action[2]: aim direction (0..63 -> 64 discrete angles)
        # action[3]: weapon selection (0: None, 1: Water, 2: Grass, 3: Fire)
        self.num_agents = 2 if curriculum_level == 3 else 1
        self.aim_angles_count = 64
        self.action_space = spaces.MultiDiscrete([3, 3, self.aim_angles_count, 4] * self.num_agents)
        
        # Observation space size:
        # Agent status: [x, y, vel_x, vel_y, health, cooldown, last_fired_weapon_one_hot (3)] -> 9 features
        # 5 Closest enemies: 5 * [rel_x, rel_y, sin_angle, cos_angle, distance, health, type_one_hot (3), correct_weapon_hint, active_flag] -> 5 * 11 = 55 features
        # Total = 64 features per agent
        obs_size = (9 + (self.max_enemies_tracked * 11)) * self.num_agents
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Element effectiveness counters: weapon_type -> effective_against_enemy_type
        # Weapon: 0: Water, 1: Grass, 2: Fire
        # Enemy: 0: Water, 1: Grass, 2: Fire
        self.counters = {
            0: 2,  # Water beats Fire
            1: 0,  # Grass beats Water
            2: 1   # Fire beats Grass
        }
        
        # Reverse counters (which enemy is countered by which weapon)
        self.enemy_weakness = {v: k for k, v in self.counters.items()}
        
        # Setup Spawner
        self.spawner = EnemySpawner(
            width=self.width, height=self.height,
            spawn_cooldown=preset["spawn_cooldown"],
            allowed_types=self.allowed_enemy_types
        )
        
        # Game State Variables
        self.agent_x = [0.0] * self.num_agents
        self.agent_y = [0.0] * self.num_agents
        self.agent_vel_x = [0.0] * self.num_agents
        self.agent_vel_y = [0.0] * self.num_agents
        self.agent_health = [0.0] * self.num_agents
        self.shoot_cooldown = [0] * self.num_agents
        self.last_fired_weapon = [-1] * self.num_agents  # -1 means none, 0..3 for types
        
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.ally_spawn_cooldown = np.random.randint(150, 300)
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        
        # Aim direction for renderer
        self.last_aim_x = [1.0] * self.num_agents   # Default facing right
        self.last_aim_y = [0.0] * self.num_agents
        
        # Pygame renderer placeholder
        self.renderer = None
        
        # Rolling history of episode rewards for dynamic reward adaptation
        self.episode_rewards_history = []
        self.mean_episode_reward = 0.0
        self.current_episode_reward = 0.0

    def reset(self, seed=None, options=None):
        # Accumulate reward for history tracking
        if hasattr(self, "current_episode_reward") and self.steps_survived > 0:
            self.episode_rewards_history.append(self.current_episode_reward)
            if len(self.episode_rewards_history) > 50:
                self.episode_rewards_history.pop(0)
            self.mean_episode_reward = float(np.mean(self.episode_rewards_history))
            
        self.current_episode_reward = 0.0
        
        super().reset(seed=seed)
        
        # Initialize agents
        if self.num_agents == 1:
            self.agent_x = [self.width / 2.0]
            self.agent_y = [self.height / 2.0]
        else:
            # Spread them out from center slightly
            self.agent_x = [self.width / 2.0 - 40.0, self.width / 2.0 + 40.0]
            self.agent_y = [self.height / 2.0, self.height / 2.0]
            
        self.agent_vel_x = [0.0] * self.num_agents
        self.agent_vel_y = [0.0] * self.num_agents
        self.agent_health = [self.max_health] * self.num_agents
        self.shoot_cooldown = [0] * self.num_agents
        self.last_fired_weapon = [-1] * self.num_agents
        
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.ally_spawn_cooldown = np.random.randint(150, 300)
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        self.episode_missed_bullets = 0
        self.spawner.reset()
        
        # Reset aim state
        self.last_aim_x = [1.0] * self.num_agents
        self.last_aim_y = [0.0] * self.num_agents
        
        # Initial enemies spawn
        for _ in range(2):
            hostile_enemies = [e for e in self.enemies if not getattr(e, 'is_ally', False)]
            enemy = self.spawner.step(hostile_enemies, self.agent_x, self.agent_y, self.max_enemies_on_screen, self.difficulty)
            if enemy is not None:
                enemy.speed *= self.enemy_speed_mult
                self.enemies.append(enemy)
                
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, info

    def _get_agent_obs(self, agent_idx):
        ax = self.agent_x[agent_idx]
        ay = self.agent_y[agent_idx]
        avx = self.agent_vel_x[agent_idx]
        avy = self.agent_vel_y[agent_idx]
        ahp = self.agent_health[agent_idx]
        acd = self.shoot_cooldown[agent_idx]
        alw = self.last_fired_weapon[agent_idx]
        
        # 1. Agent status (10 items)
        agent_obs = [
            ax / self.width,
            ay / self.height,
            avx / self.agent_speed,
            avy / self.agent_speed,
            ahp / self.max_health,
            acd / self.shoot_cooldown_max,
        ]
        # Weapon one-hot (3 items)
        weapon_one_hot = [0.0] * 3
        if alw != -1:
            weapon_one_hot[alw] = 1.0
        agent_obs.extend(weapon_one_hot)
        
        # 2. Track closest hostile enemies vs closest allies separately
        hostiles_info = []
        allies_info = []
        for enemy in self.enemies:
            if enemy.alive:
                dx = enemy.x - ax
                dy = enemy.y - ay
                dist = np.sqrt(dx**2 + dy**2)
                if getattr(enemy, 'is_ally', False):
                    allies_info.append((dist, dx, dy, enemy))
                else:
                    hostiles_info.append((dist, dx, dy, enemy))
                    
        # Sort by distance
        hostiles_info.sort(key=lambda x: x[0])
        allies_info.sort(key=lambda x: x[0])
        
        # Build enemy observation vector (3 closest hostiles, 2 closest allies)
        enemy_obs = []
        
        # 3 Closest Hostiles (3 * 11 = 33 features)
        for i in range(3):
            if i < len(hostiles_info):
                dist, dx, dy, enemy = hostiles_info[i]
                rel_x = dx / self.width
                rel_y = dy / self.height
                norm_dist = dist / (np.sqrt(self.width**2 + self.height**2))
                norm_health = enemy.health / enemy.max_health
                
                enemy_angle = np.arctan2(-dy, dx)
                sin_angle = np.sin(enemy_angle)
                cos_angle = np.cos(enemy_angle)
                
                type_one_hot = [0.0] * 3
                type_one_hot[enemy.enemy_type] = 1.0
                correct_weapon = self.enemy_weakness.get(enemy.enemy_type, 0)
                weapon_hint = correct_weapon / 2.0
                
                enemy_obs.extend([
                    rel_x,
                    rel_y,
                    sin_angle,
                    cos_angle,
                    norm_dist,
                    norm_health,
                    type_one_hot[0],
                    type_one_hot[1],
                    type_one_hot[2],
                    weapon_hint,
                    1.0  # active flag
                ])
            else:
                enemy_obs.extend([0.0] * 11)
                
        # 2 Closest Allies (2 * 11 = 22 features)
        for i in range(2):
            if i < len(allies_info):
                dist, dx, dy, enemy = allies_info[i]
                rel_x = dx / self.width
                rel_y = dy / self.height
                norm_dist = dist / (np.sqrt(self.width**2 + self.height**2))
                norm_health = enemy.health / enemy.max_health
                
                enemy_angle = np.arctan2(-dy, dx)
                sin_angle = np.sin(enemy_angle)
                cos_angle = np.cos(enemy_angle)
                
                enemy_obs.extend([
                    rel_x,
                    rel_y,
                    sin_angle,
                    cos_angle,
                    norm_dist,
                    norm_health,
                    0.0, 0.0, 0.0, -1.0,  # All zeros type_one_hot and hint = -1.0
                    1.0  # active flag
                ])
            else:
                enemy_obs.extend([0.0] * 11)
                
        return agent_obs + enemy_obs

    def _get_obs(self):
        obs_list = []
        for idx in range(self.num_agents):
            obs_list.extend(self._get_agent_obs(idx))
        total_obs = np.array(obs_list, dtype=np.float32)
        return np.clip(total_obs, -1.0, 1.0)

    def _get_info(self):
        # Calculate dynamic rewards for logging/info
        mean_rew = self.mean_episode_reward
        R_min = 200.0
        R_max = 600.0
        k = np.clip((mean_rew - R_min) / (R_max - R_min), 0.0, 1.0)
        r_aim_decent = 1.8 * (1.0 - k) + 0.6 * k
        r_aim_great = 2.6 * (1.0 - k) + 3.8 * k
        
        return {
            "score": self.score,
            "health": self.agent_health[0] if self.num_agents == 1 else self.agent_health,
            "steps_survived": self.steps_survived,
            "difficulty": self.difficulty,
            "enemies_count": len([e for e in self.enemies if e.alive]),
            "curriculum_level": self.curriculum_level,
            "mean_episode_reward": self.mean_episode_reward,
            "r_aim_decent": r_aim_decent,
            "r_aim_great": r_aim_great
        }

    def _find_closest_targetable_enemy(self, agent_idx=0):
        """Find the closest enemy that the agent has the correct weapon type for."""
        best_dist = float('inf')
        best_enemy = None
        ax = self.agent_x[agent_idx]
        ay = self.agent_y[agent_idx]
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - ax
            dy = enemy.y - ay
            dist = np.sqrt(dx**2 + dy**2)
            if dist < best_dist:
                best_dist = dist
                best_enemy = enemy
        return best_enemy, best_dist

    def step(self, action):
        self.steps_survived += 1
        
        # Increase difficulty slightly over time
        self.difficulty = 1.0 + (self.steps_survived / 1200.0)
        
        alignment_reward = 0.0
        enemy_hit_correct = 0
        enemy_hit_incorrect = 0
        enemy_killed_count = 0
        ally_hit_count = 0
        ally_killed_count = 0
        bullets_fired = 0
        ally_aim_penalty = 0.0
        ally_shoot_penalty = 0.0
        trigger_discipline_reward = 0.0
        precision_hits = 0
        
        # 1. Parse and update each agent
        for idx in range(self.num_agents):
            agent_action = action[idx*4 : (idx+1)*4]
            move_x = float(agent_action[0]) - 1.0
            move_y = float(agent_action[1]) - 1.0
            
            # Normalize movement vector if diagonal
            move_len = np.sqrt(move_x**2 + move_y**2)
            if move_len > 0:
                move_x /= move_len
                move_y /= move_len
                
            # aim direction (0..63 -> 64 discrete angles)
            aim_angle = float(agent_action[2]) * (2.0 * np.pi / 64.0)
            aim_x = np.cos(aim_angle)
            aim_y = -np.sin(aim_angle) # y axis goes down
            
            # weapon (0 -> None, 1..4 -> weapons 0..3)
            fired_weapon_type = int(agent_action[3]) - 1
            # Prevent shooting if there are no active hostile enemies on the map
            active_hostiles_count = len([e for e in self.enemies if e.alive and not getattr(e, 'is_ally', False)])
            shoot_triggered = (fired_weapon_type >= 0) and (active_hostiles_count > 0)
            
            # Option C: Calculate alignment warning with close friendly allies
            active_allies = [e for e in self.enemies if e.alive and getattr(e, 'is_ally', False)]
            for ally in active_allies:
                ax = self.agent_x[idx]
                ay = self.agent_y[idx]
                dx = ally.x - ax
                dy = ally.y - ay
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    ex = dx / dist
                    ey = dy / dist
                    alignment = aim_x * ex + aim_y * ey
                    if alignment > 0.7:
                        closeness = max(0.0, 1.0 - (dist / 300.0))
                        # Continuous aiming warning penalty
                        ally_aim_penalty += 0.2 * closeness * (alignment - 0.7) / 0.3
                        # Stronger warning penalty if pulling the trigger in ally direction
                        if shoot_triggered and self.shoot_cooldown[idx] <= 0:
                            ally_shoot_penalty = max(ally_shoot_penalty, 3.0)
            
            # Keep track of last aiming vector for renderer
            self.last_aim_x[idx] = aim_x
            self.last_aim_y[idx] = aim_y
            
            # Calculate alignment with closest hostile enemy (for continuous aiming reward)
            enemy_alignment = 0.0
            active_enemies = [e for e in self.enemies if e.alive and not getattr(e, 'is_ally', False)]
            if active_enemies:
                closest_target = min(active_enemies, key=lambda e: np.sqrt((e.x - self.agent_x[idx])**2 + (e.y - self.agent_y[idx])**2))
                ex = closest_target.x - self.agent_x[idx]
                ey = closest_target.y - self.agent_y[idx]
                edist = np.sqrt(ex**2 + ey**2)
                if edist > 0:
                    ex /= edist
                    ey /= edist
                    enemy_alignment = aim_x * ex + aim_y * ey
                    
            # Continuous alignment reward (every step, strong signal to learn aiming)
            if enemy_alignment > 0.90:
                r_aim_decent = 0.25
                r_aim_great = 0.60
                r_aim_perfect = 0.55
                alignment_reward += r_aim_decent
                if enemy_alignment > 0.96:
                    alignment_reward += r_aim_great
                if enemy_alignment > 0.99:
                    alignment_reward += r_aim_perfect
            
            # Update agent velocity & position
            self.agent_vel_x[idx] = move_x * self.agent_speed
            self.agent_vel_y[idx] = move_y * self.agent_speed
            
            self.agent_x[idx] += self.agent_vel_x[idx]
            self.agent_y[idx] += self.agent_vel_y[idx]
            
            # Clamp to bounds
            self.agent_x[idx] = np.clip(self.agent_x[idx], self.agent_radius, self.width - self.agent_radius)
            self.agent_y[idx] = np.clip(self.agent_y[idx], self.agent_radius, self.height - self.agent_radius)
            
            # Handle Weapon Shooting cooldown
            self.shoot_cooldown[idx] = max(0, self.shoot_cooldown[idx] - 1)
            
            if shoot_triggered and self.shoot_cooldown[idx] <= 0:
                # Normalize aim direction
                aim_len = np.sqrt(aim_x**2 + aim_y**2)
                if aim_len > 0.1:
                    bullet_dx = aim_x / aim_len
                    bullet_dy = aim_y / aim_len
                else:
                    move_len = np.sqrt(move_x**2 + move_y**2)
                    if move_len > 0.1:
                        bullet_dx = move_x / move_len
                        bullet_dy = move_y / move_len
                    else:
                        bullet_dx = 1.0
                        bullet_dy = 0.0
                
                new_bullet = Bullet(self.agent_x[idx], self.agent_y[idx], bullet_dx, bullet_dy, fired_weapon_type)
                new_bullet.well_aimed = (enemy_alignment > 0.96)
                self.bullets.append(new_bullet)
                bullets_fired += 1
                self.shoot_cooldown[idx] = self.shoot_cooldown_max
                self.last_fired_weapon[idx] = fired_weapon_type
            else:
                # Trigger discipline: reward NOT shooting when poorly aligned
                # Only counts when enemies exist and cooldown is ready (agent COULD shoot but chose not to)
                if active_enemies and self.shoot_cooldown[idx] <= 0 and enemy_alignment < 0.90:
                    trigger_discipline_reward += 0.12
                                
        # 4. Update Bullets
        active_bullets = []
        missed_bullets_count = 0
        well_aimed_misses_count = 0
        for bullet in self.bullets:
            bullet.update()
            if bullet.active:
                if 0 <= bullet.x <= self.width and 0 <= bullet.y <= self.height:
                    active_bullets.append(bullet)
                else:
                    if bullet.well_aimed:
                        well_aimed_misses_count += 1
                    else:
                        missed_bullets_count += 1
        self.bullets = active_bullets
        self.episode_missed_bullets += missed_bullets_count + well_aimed_misses_count
        
        # 5. Spawning & Updating Enemies
        hostile_enemies = [e for e in self.enemies if not getattr(e, 'is_ally', False)]
        # Cap dynamic enemy scaling to prevent overwhelming spawn numbers at high difficulty
        max_allowed_cap = 5 if self.curriculum_level == 1 else (6 if self.curriculum_level == 2 else 8)
        scaled_max_enemies = min(max_allowed_cap, int(self.max_enemies_on_screen * (1.0 + (self.difficulty - 1.0) * 0.5)))
        new_enemy = self.spawner.step(
            hostile_enemies, self.agent_x, self.agent_y, 
            scaled_max_enemies, self.difficulty
        )
        if new_enemy is not None:
            new_enemy.speed *= self.enemy_speed_mult
            self.enemies.append(new_enemy)
            
        # Spawn Allies (Level 2+ only — let Level 1 focus on basics)
        if self.curriculum_level >= 2:
          self.ally_spawn_cooldown -= 1
        if self.ally_spawn_cooldown <= 0:
            edge = np.random.randint(0, 4)
            if edge == 0:  # Top
                ax = np.random.uniform(0, self.width)
                ay = 0
                tx = np.random.uniform(0, self.width)
                ty = self.height
            elif edge == 1:  # Right
                ax = self.width
                ay = np.random.uniform(0, self.height)
                tx = 0
                ty = np.random.uniform(0, self.height)
            elif edge == 2:  # Bottom
                ax = np.random.uniform(0, self.width)
                ay = self.height
                tx = np.random.uniform(0, self.width)
                ty = 0
            else:  # Left
                ax = 0
                ay = np.random.uniform(0, self.height)
                tx = self.width
                ty = np.random.uniform(0, self.height)
                
            # Distance from closest agent
            min_dist = min(np.sqrt((ax - agent_x_val)**2 + (ay - agent_y_val)**2) for agent_x_val, agent_y_val in zip(self.agent_x, self.agent_y))
            if min_dist > 200.0:
                self.enemies.append(Ally(ax, ay, tx, ty))
                self.ally_spawn_cooldown = np.random.randint(200, 400)
          # end if curriculum_level >= 2
            
        # Update enemies positions
        for enemy in self.enemies:
            if enemy.alive:
                # Find closest active agent
                closest_agent_idx = 0
                closest_dist = float('inf')
                for idx in range(self.num_agents):
                    dist = np.sqrt((enemy.x - self.agent_x[idx])**2 + (enemy.y - self.agent_y[idx])**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_agent_idx = idx
                enemy.update(self.agent_x[closest_agent_idx], self.agent_y[closest_agent_idx])
                
        # Enemy Shooting (Curriculum Level 3 and 4)
        if self.curriculum_level >= 3:
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                if getattr(enemy, 'is_ally', False):
                    continue
                if not hasattr(enemy, 'shoot_cooldown'):
                    enemy.shoot_cooldown = np.random.randint(40, 120)
                
                enemy.shoot_cooldown -= 1
                if enemy.shoot_cooldown <= 0:
                    # Fire directly towards closest agent
                    closest_agent_idx = 0
                    closest_dist = float('inf')
                    for idx in range(self.num_agents):
                        dist = np.sqrt((enemy.x - self.agent_x[idx])**2 + (enemy.y - self.agent_y[idx])**2)
                        if dist < closest_dist:
                            closest_dist = dist
                            closest_agent_idx = idx
                            
                    dx = self.agent_x[closest_agent_idx] - enemy.x
                    dy = self.agent_y[closest_agent_idx] - enemy.y
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        bullet_dx = dx / dist
                        bullet_dy = dy / dist
                        new_eb = EnemyBullet(enemy.x, enemy.y, bullet_dx, bullet_dy)
                        self.enemy_bullets.append(new_eb)
                    cooldown_max = max(60, int(np.random.randint(120, 240) / self.difficulty))
                    enemy.shoot_cooldown = cooldown_max
                    
        # Update Enemy Bullets
        active_enemy_bullets = []
        for eb in self.enemy_bullets:
            eb.update()
            if eb.active:
                if 0 <= eb.x <= self.width and 0 <= eb.y <= self.height:
                    active_enemy_bullets.append(eb)
        self.enemy_bullets = active_enemy_bullets
                
        # 6. Collisions: Bullet vs Enemy
        for bullet in self.bullets:
            if not bullet.active:
                continue
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                dist = np.sqrt((bullet.x - enemy.x)**2 + (bullet.y - enemy.y)**2)
                if dist < (bullet.radius + enemy.radius):
                    bullet.active = False
                    
                    if getattr(enemy, 'is_ally', False):
                        # Penalize shooting at Allies (any weapon type)
                        enemy.take_damage(75.0) # Hitting kills or heavily damages (matched to 75.0)
                        ally_hit_count += 1
                        if not enemy.alive:
                            ally_killed_count += 1
                    else:
                        correct_weapon = self.enemy_weakness.get(enemy.enemy_type, -1)
                        if bullet.bullet_type == correct_weapon:
                            # --- R_hit: Effective hit! ---
                            damage = 75.0  # Increased bullet damage from 50.0 to 75.0 to balance DPS
                            enemy.take_damage(damage)
                            enemy_hit_correct += 1
                            # Track precision hits (well-aimed bullets that connect)
                            if bullet.well_aimed:
                                precision_hits += 1
                            
                            if not enemy.alive:
                                # --- R_kill: Enemy killed! ---
                                self.score += 10
                                enemy_killed_count += 1
                        else:
                            enemy_hit_incorrect += 1
                    break  # Bullet hit this enemy, stop checking others
                    
        # Filter dead enemies
        self.enemies = [e for e in self.enemies if e.alive]
        
        # 7. Collisions: Enemy vs Agent
        damage_taken = 0.0
        for idx in range(self.num_agents):
            ax = self.agent_x[idx]
            ay = self.agent_y[idx]
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                dist = np.sqrt((enemy.x - ax)**2 + (enemy.y - ay)**2)
                if dist < (self.agent_radius + enemy.radius):
                    tick_damage = (enemy.damage_rate / 30.0) * (1.0 + (self.difficulty - 1.0) * 0.1)
                    self.agent_health[idx] -= tick_damage
                    damage_taken += tick_damage
                
        # 7.5 Collisions: Enemy Bullet vs Agent
        eb_damage_taken = 0.0
        for eb in self.enemy_bullets:
            if not eb.active:
                continue
            for idx in range(self.num_agents):
                ax = self.agent_x[idx]
                ay = self.agent_y[idx]
                dist = np.sqrt((eb.x - ax)**2 + (eb.y - ay)**2)
                if dist < (self.agent_radius + eb.radius):
                    eb.active = False
                    bullet_damage = 15.0
                    self.agent_health[idx] -= bullet_damage
                    eb_damage_taken += bullet_damage
                    break # Bullet hit this agent, stop checking other agents
 
        # --- P_edge: Edge/Corner penalty to prevent camping ---
        edge_margin = 100.0
        edge_penalty = 0.0
        for idx in range(self.num_agents):
            ax = self.agent_x[idx]
            ay = self.agent_y[idx]
            if ax < edge_margin:
                edge_penalty += (edge_margin - ax) / edge_margin
            elif ax > self.width - edge_margin:
                edge_penalty += (ax - (self.width - edge_margin)) / edge_margin
                
            if ay < edge_margin:
                edge_penalty += (edge_margin - ay) / edge_margin
            elif ay > self.height - edge_margin:
                edge_penalty += (ay - (self.height - edge_margin)) / edge_margin
            
        # --- Proximity danger: continuous penalty for being near hostile enemies ---
        proximity_danger = 0.0
        danger_radius = 150.0
        for idx in range(self.num_agents):
            ax = self.agent_x[idx]
            ay = self.agent_y[idx]
            for enemy in self.enemies:
                if not enemy.alive or getattr(enemy, 'is_ally', False):
                    continue
                dist = np.sqrt((enemy.x - ax)**2 + (enemy.y - ay)**2)
                if dist < danger_radius:
                    # Linear ramp: 0.0 at edge of danger zone, 1.0 at contact
                    proximity_danger += (1.0 - dist / danger_radius)
        
        # 8. Check Termination (Death)
        terminated = False
        truncated = False
        
        any_dead = False
        for idx in range(self.num_agents):
            if self.agent_health[idx] <= 0:
                self.agent_health[idx] = 0.0
                any_dead = True
                
        # Terminate only on death — ally hit penalties are sufficient punishment
        if any_dead:
            terminated = True
            
        # Max steps limit
        if self.steps_survived >= 3600:  # 1 minute at 60 fps
            truncated = True
            
        # Compile step metrics for reward calculation
        step_metrics = {
            "steps_survived": self.steps_survived,
            "difficulty": self.difficulty,
            "agent_healths": self.agent_health,
            "num_agents": self.num_agents,
            "enemies_count": len([e for e in self.enemies if e.alive]),
            "bullets_count": len(self.bullets),
            "enemy_bullets_count": len(self.enemy_bullets),
            "score": self.score,
            "alignment_reward": alignment_reward,
            "missed_bullets_count": missed_bullets_count,
            "well_aimed_misses_count": well_aimed_misses_count,
            "episode_missed_bullets": self.episode_missed_bullets,
            "bullets_fired": bullets_fired,
            "enemy_hit_correct": enemy_hit_correct,
            "enemy_hit_incorrect": enemy_hit_incorrect,
            "enemy_killed": enemy_killed_count,
            "ally_hit": ally_hit_count,
            "ally_killed": ally_killed_count,
            "damage_taken": damage_taken,
            "eb_damage_taken": eb_damage_taken,
            "edge_penalty": edge_penalty,
            "survival_bonus": 0.1,
            "any_dead": any_dead,
            "ally_aim_penalty": ally_aim_penalty,
            "ally_shoot_penalty": ally_shoot_penalty,
            "trigger_discipline_reward": trigger_discipline_reward,
            "precision_hits": precision_hits,
            "proximity_danger": proximity_danger
        }
        
        # Calculate reward
        if calculate_reward is not None:
            try:
                reward = calculate_reward(self, step_metrics)
            except Exception as e:
                # Fallback to default reward logic on function error to prevent crash
                reward = alignment_reward
                reward -= missed_bullets_count * 2.0
                reward -= well_aimed_misses_count * 0.1
                reward -= ally_aim_penalty + ally_shoot_penalty
                reward += enemy_hit_correct * 3.0 + enemy_killed_count * 10.0
                reward -= ally_hit_count * 15.0 + ally_killed_count * 25.0
                reward -= damage_taken * 1.1 + eb_damage_taken * 1.5
                reward -= edge_penalty * 0.08
                reward += 0.05
                if any_dead:
                    reward -= 10.0
        else:
            reward = alignment_reward
            reward -= missed_bullets_count * 2.0
            reward -= well_aimed_misses_count * 0.1
            reward -= ally_aim_penalty + ally_shoot_penalty
            reward += enemy_hit_correct * 3.0 + enemy_killed_count * 10.0
            reward -= ally_hit_count * 15.0 + ally_killed_count * 25.0
            reward -= damage_taken * 1.1 + eb_damage_taken * 1.5
            reward -= edge_penalty * 0.08
            reward += 0.05
            if any_dead:
                reward -= 10.0
                
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        self.current_episode_reward += reward
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.renderer is None:
            from env.renderer import PygameRenderer
            self.renderer = PygameRenderer(self.width, self.height)
            
        return self.renderer.render(
            self.agent_x, self.agent_y, self.agent_health, self.max_health,
            self.enemies, self.bullets, self.score, self.steps_survived,
            self.last_fired_weapon, self.shoot_cooldown, self.difficulty,
            self.last_aim_x, self.last_aim_y,
            self.render_mode,
            enemy_bullets=self.enemy_bullets
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
