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
        3: {  # Level 3: Full game — all 4 types, normal speed
            "max_enemies": 6,           # (was 5)
            "enemy_types": [0, 1, 2, 3],
            "spawn_cooldown": 40,       # (was 60)
            "enemy_speed_mult": 1.25,   # (was 1.0)
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
        self.shoot_cooldown_max = 12  # ticks between shots
        self.max_enemies_on_screen = preset["max_enemies"]
        self.max_enemies_tracked = 5  # For RL observation (always 5 slots)
        
        # Action space:
        # action[0]: movement dx (0: Left, 1: None, 2: Right)
        # action[1]: movement dy (0: Up, 1: None, 2: Down)
        # action[2]: aim direction (0..127 -> 128 discrete angles)
        # action[3]: weapon selection (0: None, 1: Water, 2: Grass, 3: Fire, 4: Wind)
        self.action_space = spaces.MultiDiscrete([3, 3, 128, 5])
        
        # Observation space size:
        # Agent status: [x, y, vel_x, vel_y, health, cooldown, last_fired_weapon_one_hot (4)] -> 10 features
        # 5 Closest enemies: 5 * [rel_x, rel_y, distance, health, type_one_hot (4), correct_weapon_hint, active_flag] -> 5 * 10 = 50 features
        # Total = 60 features
        obs_size = 10 + (self.max_enemies_tracked * 10)
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
        self.spawner = EnemySpawner(
            width=self.width, height=self.height,
            spawn_cooldown=preset["spawn_cooldown"],
            allowed_types=self.allowed_enemy_types
        )
        
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
        self.enemy_bullets = []
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        
        # Aim direction for renderer
        self.last_aim_x = 1.0   # Default facing right
        self.last_aim_y = 0.0
        
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
        self.enemy_bullets = []
        self.score = 0
        self.steps_survived = 0
        self.difficulty = 1.0
        self.spawner.reset()
        
        # Reset aim state
        self.last_aim_x = 1.0
        self.last_aim_y = 0.0
        
        # Initial enemies spawn
        for _ in range(2):
            enemy = self.spawner.step(self.enemies, self.agent_x, self.agent_y, self.max_enemies_on_screen, self.difficulty)
            if enemy is not None:
                enemy.speed *= self.enemy_speed_mult
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
                
                # Weapon-counter hint: which weapon index is effective against this enemy
                correct_weapon = self.enemy_weakness.get(enemy.enemy_type, 0)
                weapon_hint = correct_weapon / 3.0  # Normalize to [0, 1]
                
                enemy_obs.extend([
                    rel_x,
                    rel_y,
                    norm_dist,
                    norm_health,
                    type_one_hot[0],
                    type_one_hot[1],
                    type_one_hot[2],
                    type_one_hot[3],
                    weapon_hint,
                    1.0  # active flag
                ])
            else:
                # Padding (10 zeros matching active slot: rel_x, rel_y, dist, health, type_one_hot(4), weapon_hint, active=0)
                enemy_obs.extend([0.0] * 10)
                
        total_obs = np.array(agent_obs + enemy_obs, dtype=np.float32)
        # Clip to make sure it's within bounds
        return np.clip(total_obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "score": self.score,
            "health": self.agent_health,
            "steps_survived": self.steps_survived,
            "difficulty": self.difficulty,
            "enemies_count": len([e for e in self.enemies if e.alive]),
            "curriculum_level": self.curriculum_level
        }


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
        move_x = float(action[0]) - 1.0
        move_y = float(action[1]) - 1.0
        
        # Normalize movement vector if diagonal
        move_len = np.sqrt(move_x**2 + move_y**2)
        if move_len > 0:
            move_x /= move_len
            move_y /= move_len
            
        # action[2]: aim direction (0..127 -> 128 discrete angles)
        aim_angle = float(action[2]) * (2.0 * np.pi / 128.0)
        aim_x = np.cos(aim_angle)
        aim_y = -np.sin(aim_angle) # y axis goes down
        
        # action[3]: weapon (0 -> None, 1..4 -> weapons 0..3)
        fired_weapon_type = int(action[3]) - 1
        # Prevent shooting if there are no active enemies on the map
        active_enemies_count = len([e for e in self.enemies if e.alive])
        shoot_triggered = (fired_weapon_type >= 0) and (active_enemies_count > 0)
        
        # Keep track of last aiming vector for renderer
        self.last_aim_x = aim_x
        self.last_aim_y = aim_y
        
        self.steps_survived += 1
        
        # Increase difficulty slightly over time
        self.difficulty = 1.0 + (self.steps_survived / 1200.0)
        
        # 2. Update agent velocity & position
        self.agent_vel_x = move_x * self.agent_speed
        self.agent_vel_y = move_y * self.agent_speed
        
        self.agent_x += self.agent_vel_x
        self.agent_y += self.agent_vel_y
        
        # Clamp to bounds
        self.agent_x = np.clip(self.agent_x, self.agent_radius, self.width - self.agent_radius)
        self.agent_y = np.clip(self.agent_y, self.agent_radius, self.height - self.agent_radius)
        
        # =============================================
        # SIMPLIFIED REWARD (Option A)
        # Only 5 clear signals: kill, hit, aim, damage, death, survival
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
            
            # --- R_aim: Decoupled Alignment Reward ---
            # Reward aiming at the ABSOLUTE closest active enemy, regardless of weapon type
            active_enemies = [e for e in self.enemies if e.alive]
            if active_enemies:
                closest_target = min(active_enemies, key=lambda e: np.sqrt((e.x - self.agent_x)**2 + (e.y - self.agent_y)**2))
                ex = closest_target.x - self.agent_x
                ey = closest_target.y - self.agent_y
                edist = np.sqrt(ex**2 + ey**2)
                if edist > 0:
                    ex /= edist
                    ey /= edist
                    alignment = bullet_dx * ex + bullet_dy * ey
                    if alignment > 0.7:
                        reward += 1.5   # Decent aim
                        if alignment > 0.85:
                            reward += 2.8  # Great aim (total +4.0)
                    
        # 4. Update Bullets
        active_bullets = []
        missed_bullets_count = 0
        for bullet in self.bullets:
            bullet.update()
            if bullet.active:
                if 0 <= bullet.x <= self.width and 0 <= bullet.y <= self.height:
                    active_bullets.append(bullet)
                else:
                    missed_bullets_count += 1
        self.bullets = active_bullets
        
        # --- P_miss: Penalty for bullets that go off-screen (missed shots) ---
        reward -= missed_bullets_count * 1.7
        
        # 5. Spawning & Updating Enemies
        new_enemy = self.spawner.step(
            self.enemies, self.agent_x, self.agent_y, 
            self.max_enemies_on_screen, self.difficulty
        )
        if new_enemy is not None:
            new_enemy.speed *= self.enemy_speed_mult
            self.enemies.append(new_enemy)
            
        # Update enemies positions
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.agent_x, self.agent_y)
                
        # Enemy Shooting (Curriculum Level 3 only)
        if self.curriculum_level == 3:
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                if not hasattr(enemy, 'shoot_cooldown'):
                    enemy.shoot_cooldown = np.random.randint(40, 120)
                
                enemy.shoot_cooldown -= 1
                if enemy.shoot_cooldown <= 0:
                    # Fire directly towards agent
                    dx = self.agent_x - enemy.x
                    dy = self.agent_y - enemy.y
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
                    
                    correct_weapon = self.enemy_weakness.get(enemy.enemy_type, -1)
                    if bullet.bullet_type == correct_weapon:
                        # --- R_hit: Effective hit! ---
                        damage = 50.0
                        enemy.take_damage(damage)
                        reward += 3.0  # Correct weapon hit
                        
                        if not enemy.alive:
                            # --- R_kill: Enemy killed! ---
                            self.score += 10
                            reward += 10.0  # Kill reward
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
                tick_damage = (enemy.damage_rate / 30.0) * (1.0 + (self.difficulty - 1.0) * 0.1)
                self.agent_health -= tick_damage
                damage_taken += tick_damage
                
        # --- P_damage: Damage taken penalty ---
        reward -= damage_taken * 1.1
        
        # 7.5 Collisions: Enemy Bullet vs Agent
        eb_damage_taken = 0.0
        for eb in self.enemy_bullets:
            if not eb.active:
                continue
            dist = np.sqrt((eb.x - self.agent_x)**2 + (eb.y - self.agent_y)**2)
            if dist < (self.agent_radius + eb.radius):
                eb.active = False
                bullet_damage = 15.0
                self.agent_health -= bullet_damage
                eb_damage_taken += bullet_damage
                
        # --- P_bullet: Penalty for getting hit by enemy bullet ---
        reward -= eb_damage_taken * 1.5

        # --- P_edge: Edge/Corner penalty to prevent camping ---
        edge_margin = 100.0
        edge_penalty = 0.0
        if self.agent_x < edge_margin:
            edge_penalty += (edge_margin - self.agent_x) / edge_margin
        elif self.agent_x > self.width - edge_margin:
            edge_penalty += (self.agent_x - (self.width - edge_margin)) / edge_margin
            
        if self.agent_y < edge_margin:
            edge_penalty += (edge_margin - self.agent_y) / edge_margin
        elif self.agent_y > self.height - edge_margin:
            edge_penalty += (self.agent_y - (self.height - edge_margin)) / edge_margin
            
        reward -= edge_penalty * 0.08  # Max corner penalty = -0.16/step (exceeds +0.05 survival bonus)

        # --- R_survival: Survival bonus per step ---
        reward += 0.05
        
        # 8. Check Termination (Death)
        terminated = False
        truncated = False
        
        if self.agent_health <= 0:
            self.agent_health = 0.0
            terminated = True
            reward -= 10.0  # Death penalty
            
        # Max steps limit
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
            self.render_mode,
            enemy_bullets=self.enemy_bullets
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
