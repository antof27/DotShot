import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.generator import EnemySpawner, Enemy

class Bullet:
    def __init__(self, x, y, dx, dy, bullet_type):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.bullet_type = bullet_type  # 0: Water, 1: Grass, 2: Fire, 3: Wind/Flying
        self.speed = 10.0
        self.radius = 5.0
        self.active = True
        
        if bullet_type == 0:
            self.color = (0, 100, 255)
            self.name = "Water Bullet"
        elif bullet_type == 1:
            self.color = (0, 255, 100)
            self.name = "Grass Bullet"
        elif bullet_type == 2:
            self.color = (255, 100, 0)
            self.name = "Fire Bullet"
        else:
            self.color = (240, 240, 240)
            self.name = "Wind Bullet"

    def update(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed


class ElementShooterEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, width=800, height=800, multidiscrete=False):
        super(ElementShooterEnv, self).__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.multidiscrete = multidiscrete
        
        # Game constants
        self.agent_speed = 4.0
        self.agent_radius = 15.0
        self.max_health = 100.0
        self.shoot_cooldown_max = 12  # ticks between shots
        self.max_enemies_on_screen = 5
        self.max_enemies_tracked = 5  # For RL observation
        
        if self.multidiscrete:
            # Action space:
            # action[0]: movement dx (0: Left, 1: None, 2: Right)
            # action[1]: movement dy (0: Up, 1: None, 2: Down)
            # action[2]: aim direction (0..15 -> 16 discrete angles)
            # action[3]: weapon selection (0: None, 1: Water, 2: Grass, 3: Fire, 4: Wind)
            self.action_space = spaces.MultiDiscrete([3, 3, 16, 5])
        else:
            # Action space: 
            # action[0], action[1]: movement dx, dy in [-1, 1]
            # action[2], action[3]: aim ax, ay in [-1, 1] (defines shooting direction vector)
            # action[4..7]: weapon shoot triggers (Water, Grass, Fire, Wind).
            # We check the highest trigger > 0.2 to fire.
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(8,), dtype=np.float32
            )
        
        # Observation space size:
        # Agent status: [x, y, vel_x, vel_y, health, cooldown, last_fired_weapon_one_hot (4)] -> 10 features
        # 5 Closest enemies: 5 * [rel_x, rel_y, distance, health, type_one_hot (4), active_flag] -> 5 * 9 = 45 features
        # Total = 55 features
        obs_size = 10 + (self.max_enemies_tracked * 9)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Element effectiveness counters: weapon_type -> effective_against_enemy_type
        # Weapon: 0: Water, 1: Grass, 2: Fire, 3: Wind
        # Enemy: 0: Water, 1: Grass, 2: Fire, 3: Flying
        self.counters = {
            0: 2,  # Water beats Fire
            1: 0,  # Grass beats Water
            2: 1,  # Fire beats Grass
            3: 3   # Wind beats Flying
        }
        
        # Reverse counters (which enemy is countered by which weapon)
        self.enemy_weakness = {v: k for k, v in self.counters.items()}
        
        # Setup Spawner
        self.spawner = EnemySpawner(width=self.width, height=self.height, spawn_cooldown=60)
        
        # Game State Variables
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_vel_x = 0.0
        self.agent_vel_y = 0.0
        self.agent_health = 0.0
        self.shoot_cooldown = 0
        self.last_fired_weapon = -1  # -1 means none, 0..3 for types
        
        self.enemies = []
        self.bullets = []
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        
        # --- Anti-spin tracking ---
        self.last_aim_x = 1.0   # Default facing right
        self.last_aim_y = 0.0
        self.prev_aim_potential = 0.0  # For PBRS aim alignment
        
        # Pygame renderer placeholder
        self.renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize agent at center
        self.agent_x = self.width / 2.0
        self.agent_y = self.height / 2.0
        self.agent_vel_x = 0.0
        self.agent_vel_y = 0.0
        self.agent_health = self.max_health
        self.shoot_cooldown = 0
        self.last_fired_weapon = -1
        
        self.enemies = []
        self.bullets = []
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        self.spawner.reset()
        
        # Reset anti-spin state
        self.last_aim_x = 1.0
        self.last_aim_y = 0.0
        self.prev_aim_potential = 0.0
        
        # Initial enemies spawn
        for _ in range(2):
            enemy = self.spawner.step(self.enemies, self.agent_x, self.agent_y, self.max_enemies_on_screen, self.difficulty)
            if enemy is not None:
                self.enemies.append(enemy)
                
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, info

    def _get_obs(self):
        # 1. Agent status (10 items)
        agent_obs = [
            self.agent_x / self.width,
            self.agent_y / self.height,
            self.agent_vel_x / self.agent_speed,
            self.agent_vel_y / self.agent_speed,
            self.agent_health / self.max_health,
            self.shoot_cooldown / self.shoot_cooldown_max,
        ]
        # Weapon one-hot (4 items)
        weapon_one_hot = [0.0] * 4
        if self.last_fired_weapon != -1:
            weapon_one_hot[self.last_fired_weapon] = 1.0
        agent_obs.extend(weapon_one_hot)
        
        # 2. Track closest enemies
        enemies_info = []
        for enemy in self.enemies:
            if enemy.alive:
                dx = enemy.x - self.agent_x
                dy = enemy.y - self.agent_y
                dist = np.sqrt(dx**2 + dy**2)
                enemies_info.append((dist, dx, dy, enemy))
                
        # Sort by distance
        enemies_info.sort(key=lambda x: x[0])
        
        # Build enemy observation vector
        enemy_obs = []
        for i in range(self.max_enemies_tracked):
            if i < len(enemies_info):
                dist, dx, dy, enemy = enemies_info[i]
                # Normalize values
                rel_x = dx / self.width
                rel_y = dy / self.height
                norm_dist = dist / (np.sqrt(self.width**2 + self.height**2))
                norm_health = enemy.health / enemy.max_health
                
                type_one_hot = [0.0] * 4
                type_one_hot[enemy.enemy_type] = 1.0
                
                enemy_obs.extend([
                    rel_x,
                    rel_y,
                    norm_dist,
                    norm_health,
                    type_one_hot[0],
                    type_one_hot[1],
                    type_one_hot[2],
                    type_one_hot[3],
                    1.0  # active flag
                ])
            else:
                # Padding
                enemy_obs.extend([0.0] * 8 + [0.0]) # rel_x, rel_y, dist, health, type_one_hot(4), active=0
                
        total_obs = np.array(agent_obs + enemy_obs, dtype=np.float32)
        # Clip to make sure it's within bounds
        return np.clip(total_obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "score": self.score,
            "health": self.agent_health,
            "steps_survived": self.steps_survived,
            "difficulty": self.difficulty,
            "enemies_count": len([e for e in self.enemies if e.alive])
        }

    def get_text_description(self):
        """
        Generates a text description of the current environment state.
        Perfect for prompting LLM agents to make a choice.
        """
        # Description of map and rules
        desc = []
        desc.append("=== ELEMENT SHOOTER ENVIRONMENT ===")
        desc.append(f"Grid size: {self.width}x{self.height} (Continuous 2D space).")
        desc.append(f"Agent Status: Health={self.agent_health:.1f}/100.0, Position=({self.agent_x:.1f}, {self.agent_y:.1f}), Weapon Cooldown={self.shoot_cooldown} ticks.")
        desc.append("Weapon rules: Water (0) beats Fire (2); Grass (1) beats Water (0); Fire (2) beats Grass (1); Wind (3) beats Flying (3).")
        
        # Enemies
        active_enemies = [e for e in self.enemies if e.alive]
        desc.append(f"Active Enemies: {len(active_enemies)} (Maximum allowed is {self.max_enemies_on_screen})")
        
        if active_enemies:
            # Sort by distance
            enemies_with_dist = []
            for e in active_enemies:
                dx = e.x - self.agent_x
                dy = e.y - self.agent_y
                dist = np.sqrt(dx**2 + dy**2)
                enemies_with_dist.append((dist, dx, dy, e))
            enemies_with_dist.sort(key=lambda x: x[0])
            
            for idx, (dist, dx, dy, enemy) in enumerate(enemies_with_dist):
                # Calculate angle/direction description
                angle = np.arctan2(-dy, dx) * 180 / np.pi  # -dy because grid y goes down
                if angle < 0:
                    angle += 360
                    
                # Map angle to cardinal/intercardinal direction
                directions = ["East", "North-East", "North", "North-West", "West", "South-West", "South", "South-East", "East"]
                dir_idx = int((angle + 22.5) / 45.0)
                direction = directions[dir_idx]
                
                weakness = ["Grass Weapon", "Fire Weapon", "Water Weapon", "Wind Weapon"][enemy.enemy_type]
                
                desc.append(f"  - Enemy {idx+1}: Type={enemy.name} (Weakness={weakness}), Health={enemy.health:.1f}/{enemy.max_health:.1f}, Distance={dist:.1f} pixels, Direction={direction} (dx={dx:.1f}, dy={dy:.1f})")
        else:
            desc.append("  No enemies detected nearby.")
            
        desc.append("\nYour Action Output format should choose a movement vector dx, dy, aiming vector ax, ay, and weapon to fire.")
        return "\n".join(desc)

    def _find_closest_targetable_enemy(self):
        """Find the closest enemy that the agent has the correct weapon type for."""
        best_dist = float('inf')
        best_enemy = None
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dx = enemy.x - self.agent_x
            dy = enemy.y - self.agent_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < best_dist:
                best_dist = dist
                best_enemy = enemy
        return best_enemy, best_dist

    def step(self, action):
        # 1. Parse actions
        if self.multidiscrete:
            # action is MultiDiscrete of size 4
            # action[0]: movement dx (0 -> -1.0, 1 -> 0.0, 2 -> 1.0)
            # action[1]: movement dy (0 -> -1.0, 1 -> 0.0, 2 -> 1.0)
            move_x = float(action[0]) - 1.0
            move_y = float(action[1]) - 1.0
            
            # Normalize movement vector if diagonal
            move_len = np.sqrt(move_x**2 + move_y**2)
            if move_len > 0:
                move_x /= move_len
                move_y /= move_len
                
            # action[2]: aim direction (0..15 -> 16 discrete angles)
            aim_angle = float(action[2]) * (2.0 * np.pi / 16.0)
            aim_x = np.cos(aim_angle)
            aim_y = -np.sin(aim_angle) # y axis goes down
            
            # action[3]: weapon (0 -> None, 1..4 -> weapons 0..3)
            fired_weapon_type = int(action[3]) - 1
            shoot_triggered = (fired_weapon_type >= 0)
        else:
            # action is Box of size 8
            action = np.clip(action, -1.0, 1.0)
            move_x, move_y = action[0], action[1]
            aim_x, aim_y = action[2], action[3]
            shoot_triggers = action[4:8]
            
            fired_weapon_type = -1
            shoot_triggered = False
            max_trigger_idx = np.argmax(shoot_triggers)
            if shoot_triggers[max_trigger_idx] > 0.2:
                fired_weapon_type = max_trigger_idx
                shoot_triggered = True
        
        self.steps_survived += 1
        
        # Increase difficulty slightly over time
        self.difficulty = 1.0 + (self.steps_survived / 1200.0)  # Difficulty increases every 20 seconds at 60fps
        
        # 2. Update agent velocity & position
        self.agent_vel_x = move_x * self.agent_speed
        self.agent_vel_y = move_y * self.agent_speed
        
        self.agent_x += self.agent_vel_x
        self.agent_y += self.agent_vel_y
        
        # Clamp to bounds
        self.agent_x = np.clip(self.agent_x, self.agent_radius, self.width - self.agent_radius)
        self.agent_y = np.clip(self.agent_y, self.agent_radius, self.height - self.agent_radius)
        
        # =============================================
        # REWARD CALCULATION (overhauled for SAC/PPO)
        # =============================================
        reward = 0.0
        
        # --- 3. Handle Weapon Shooting ---
        self.shoot_cooldown = max(0, self.shoot_cooldown - 1)
        
        fired = False
        
        if shoot_triggered and self.shoot_cooldown <= 0:
            # Normalize aim direction
            aim_len = np.sqrt(aim_x**2 + aim_y**2)
            if aim_len > 0.1:
                bullet_dx = aim_x / aim_len
                bullet_dy = aim_y / aim_len
            else:
                # Shoot in movement direction if aim is empty, or default to facing Right
                move_len = np.sqrt(move_x**2 + move_y**2)
                if move_len > 0.1:
                    bullet_dx = move_x / move_len
                    bullet_dy = move_y / move_len
                else:
                    bullet_dx = 1.0
                    bullet_dy = 0.0
            
            new_bullet = Bullet(self.agent_x, self.agent_y, bullet_dx, bullet_dy, fired_weapon_type)
            self.bullets.append(new_bullet)
            self.shoot_cooldown = self.shoot_cooldown_max
            self.last_fired_weapon = fired_weapon_type
            fired = True
            
            # --- R_aim: Aim Alignment Reward (no range limit) ---
            target_enemy_type = self.counters.get(fired_weapon_type, -1)
            target_enemies = [e for e in self.enemies if e.alive and e.enemy_type == target_enemy_type]
            if target_enemies:
                closest_target = min(target_enemies, key=lambda e: np.sqrt((e.x - self.agent_x)**2 + (e.y - self.agent_y)**2))
                ex = closest_target.x - self.agent_x
                ey = closest_target.y - self.agent_y
                edist = np.sqrt(ex**2 + ey**2)
                if edist > 0:
                    ex /= edist
                    ey /= edist
                    # Dot product of bullet direction and enemy direction
                    alignment = bullet_dx * ex + bullet_dy * ey
                    if alignment > 0.85:
                        reward += 1.0   # Good aim
                        if alignment > 0.95:
                            reward += 1.5  # Excellent aim bonus
                    else:
                        reward -= 0.3   # Firing in wrong direction at correct enemy type
            
        # 4. Update Bullets
        active_bullets = []
        for bullet in self.bullets:
            bullet.update()
            if bullet.active:
                # Bound checking
                if 0 <= bullet.x <= self.width and 0 <= bullet.y <= self.height:
                    active_bullets.append(bullet)
        self.bullets = active_bullets
        
        # 5. Spawning & Updating Enemies
        # Call spawner
        new_enemy = self.spawner.step(
            self.enemies, self.agent_x, self.agent_y, 
            self.max_enemies_on_screen, self.difficulty
        )
        if new_enemy is not None:
            self.enemies.append(new_enemy)
            
        # Update enemies positions
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.agent_x, self.agent_y)
                
        # 6. Collisions: Bullet vs Enemy
        for bullet in self.bullets:
            if not bullet.active:
                continue
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                # Distance bullet to enemy
                dist = np.sqrt((bullet.x - enemy.x)**2 + (bullet.y - enemy.y)**2)
                if dist < (bullet.radius + enemy.radius):
                    bullet.active = False
                    
                    # Check element effectiveness
                    correct_weapon = self.enemy_weakness.get(enemy.enemy_type, -1)
                    if bullet.bullet_type == correct_weapon:
                        # --- R_hit: Effective hit! ---
                        damage = 50.0
                        enemy.take_damage(damage)
                        reward += 5.0  # Hit with correct weapon
                        
                        if not enemy.alive:
                            # --- R_kill: Enemy killed! ---
                            self.score += 10
                            reward += 25.0  # Large kill reward
                            
                            # --- R_snipe: Distance bonus for long-range kills ---
                            kill_dist = np.sqrt((enemy.x - self.agent_x)**2 + (enemy.y - self.agent_y)**2)
                            max_dist = np.sqrt(self.width**2 + self.height**2)
                            dist_bonus = (kill_dist / max_dist) * 10.0  # Up to +10 for max range
                            reward += dist_bonus
                    else:
                        # Ineffective hit — wrong weapon type
                        reward -= 0.5
                    break  # Bullet hit this enemy, stop checking others
                    
        # Filter dead enemies
        self.enemies = [e for e in self.enemies if e.alive]
        
        # 7. Collisions: Enemy vs Agent
        damage_taken = 0.0
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            dist = np.sqrt((enemy.x - self.agent_x)**2 + (enemy.y - self.agent_y)**2)
            if dist < (self.agent_radius + enemy.radius):
                # Contact! Deal damage to agent
                tick_damage = (enemy.damage_rate / 30.0) * (1.0 + (self.difficulty - 1.0) * 0.1)
                self.agent_health -= tick_damage
                damage_taken += tick_damage
                
        # --- P_damage: Damage taken penalty ---
        reward -= damage_taken * 2.0
        
        # --- P_wrong_fire: Wrong weapon / no target penalties ---
        if fired:
            active_enemies = [e for e in self.enemies if e.alive]
            if len(active_enemies) == 0:
                # P_idle_fire: No enemies at all, wasting ammo
                reward -= 0.5
            else:
                matching_enemy_exists = False
                for enemy in active_enemies:
                    correct_weapon = self.enemy_weakness.get(enemy.enemy_type, -1)
                    if fired_weapon_type == correct_weapon:
                        matching_enemy_exists = True
                        break
                if not matching_enemy_exists:
                    # Fired wrong weapon type — no matching target on screen
                    reward -= 1.5
                    
        # --- P_wall: Wall avoidance penalty (stronger) ---
        wall_margin = 100.0
        wall_dist = min(
            self.agent_x, self.width - self.agent_x, 
            self.agent_y, self.height - self.agent_y
        )
        if wall_dist < wall_margin:
            # Quadratic penalty — gets very strong near walls
            wall_penalty = 0.5 * ((1.0 - wall_dist / wall_margin) ** 2)
            reward -= wall_penalty

        # --- P_spin: Anti-spin penalty (only for continuous SAC, disabled for discrete PPO) ---
        aim_len_step = np.sqrt(aim_x**2 + aim_y**2)
        if not self.multidiscrete:
            if aim_len_step > 0.1:
                norm_ax = aim_x / aim_len_step
                norm_ay = aim_y / aim_len_step
                # Angular change = 1 - dot product with last frame's aim
                dot_with_last = norm_ax * self.last_aim_x + norm_ay * self.last_aim_y
                angular_change = 1.0 - dot_with_last  # 0 = no rotation, 2 = full 180° flip
                if angular_change > 0.3:  # Threshold for "spinning"
                    reward -= angular_change * 1.0  # Strong penalty proportional to spin speed
                self.last_aim_x = norm_ax
                self.last_aim_y = norm_ay

        # --- PBRS: Aim alignment potential (every step) ---
        if aim_len_step > 0.1:
            norm_aim_x = aim_x / aim_len_step
            norm_aim_y = aim_y / aim_len_step
            best_alignment = -1.0
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                ex = enemy.x - self.agent_x
                ey = enemy.y - self.agent_y
                edist = np.sqrt(ex**2 + ey**2)
                if edist > 0:
                    alignment = (norm_aim_x * ex + norm_aim_y * ey) / edist
                    best_alignment = max(best_alignment, alignment)
            current_potential = max(0.0, best_alignment)
            pbrs_reward = 0.99 * current_potential - self.prev_aim_potential
            reward += pbrs_reward * 0.3  # Scale factor
            self.prev_aim_potential = current_potential
        else:
            self.prev_aim_potential = 0.0

        # --- R_survival: Tiny survival bonus ---
        reward += 0.01
        
        # 8. Check Termination (Death)
        terminated = False
        truncated = False
        
        if self.agent_health <= 0:
            self.agent_health = 0.0
            terminated = True
            reward -= 50.0  # Death penalty
            
        # Optional: Max steps limit to prevent infinite loops during training
        if self.steps_survived >= 3600:  # 1 minute at 60 fps
            truncated = True
            
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
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
            self.render_mode
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
