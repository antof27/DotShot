import numpy as np

class Enemy:
    def __init__(self, x, y, enemy_type):
        self.x = x
        self.y = y
        self.enemy_type = enemy_type  # 0: Water, 1: Grass, 2: Fire, 3: Flying
        self.radius = 15.0
        
        # Configure stats based on type
        if enemy_type == 0:  # Water (Blue)
            self.max_health = 100.0
            self.speed = 1.8
            self.damage_rate = 20.0  # Damage dealt per second on contact (was 15)
            self.color = (0, 150, 255)
            self.name = "Water"
        elif enemy_type == 1:  # Grass (Green)
            self.max_health = 150.0
            self.speed = 1.0
            self.damage_rate = 32.0  # (was 25)
            self.color = (50, 200, 50)
            self.name = "Grass"
        elif enemy_type == 2:  # Fire (Red)
            self.max_health = 75.0
            self.speed = 2.8
            self.damage_rate = 16.0  # (was 12)
            self.color = (255, 50, 50)
            self.name = "Fire"
        else:  # Flying (Purple)
            self.max_health = 50.0
            self.speed = 3.2
            self.damage_rate = 11.0  # (was 8)
            self.color = (200, 50, 255)
            self.name = "Flying"
            
        self.health = self.max_health
        self.alive = True

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False

    def update(self, agent_x, agent_y):
        # Move towards the agent
        dx = agent_x - self.x
        dy = agent_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

class EnemySpawner:
    def __init__(self, width=800, height=800, spawn_cooldown=60, allowed_types=None):
        self.width = width
        self.height = height
        self.spawn_cooldown = spawn_cooldown  # in steps/ticks
        self.cooldown_timer = 0
        self.allowed_types = allowed_types if allowed_types is not None else [0, 1, 2, 3]
        
    def reset(self):
        self.cooldown_timer = 0

    def step(self, current_enemies, agent_x, agent_y, max_enemies=5, difficulty=1.0):
        self.cooldown_timer -= 1
        
        # Check if we can spawn a new enemy
        if len(current_enemies) < max_enemies and self.cooldown_timer <= 0:
            # Reset cooldown, scaled by difficulty (faster spawning at higher difficulty)
            self.cooldown_timer = max(20, int(self.spawn_cooldown / difficulty))
            
            # Choose a spawn edge (0: Top, 1: Right, 2: Bottom, 3: Left)
            edge = np.random.randint(0, 4)
            if edge == 0:  # Top
                x = np.random.uniform(0, self.width)
                y = 0
            elif edge == 1:  # Right
                x = self.width
                y = np.random.uniform(0, self.height)
            elif edge == 2:  # Bottom
                x = np.random.uniform(0, self.width)
                y = self.height
            else:  # Left
                x = 0
                y = np.random.uniform(0, self.height)
                
            # Double check spawn distance to agent to prevent instant kills
            dx = x - agent_x
            dy = y - agent_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 200.0:
                return None
                
            # Randomly select enemy type from allowed types only
            enemy_type = self.allowed_types[np.random.randint(0, len(self.allowed_types))]
            new_enemy = Enemy(x, y, enemy_type)
            
            # Scale enemy stats with difficulty (e.g. slight speed boost)
            new_enemy.speed *= (1.0 + (difficulty - 1.0) * 0.1)
            new_enemy.max_health *= (1.0 + (difficulty - 1.0) * 0.15)
            new_enemy.health = new_enemy.max_health
            
            return new_enemy
        return None
