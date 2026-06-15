import pygame
import numpy as np

class PygameRenderer:
    def __init__(self, width=800, height=800):
        pygame.init()
        pygame.font.init()
        
        self.width = width
        self.height = height
        
        # Initialize screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Element Shooter RL Environment")
        
        self.clock = pygame.time.Clock()
        
        # Font setup (fall back to default if font fails)
        try:
            self.font_title = pygame.font.SysFont("Outfit", 26, bold=True)
            self.font_hud = pygame.font.SysFont("Inter", 16, bold=False)
            self.font_ammo = pygame.font.SysFont("Inter", 14, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 32)
            self.font_hud = pygame.font.Font(None, 20)
            self.font_ammo = pygame.font.Font(None, 18)
            
        # Particles for hit effects and trails
        self.particles = []  # list of dicts: {"x", "y", "vx", "vy", "color", "life", "max_life", "size"}
        
        # Map Grid
        self.grid_size = 40
        
        # Ambient starfield for premium visuals
        self.stars = [
            {
                "x": np.random.uniform(0, width),
                "y": np.random.uniform(0, height),
                "speed": np.random.uniform(0.15, 0.45),
                "size": np.random.uniform(1.0, 2.5),
                "brightness": np.random.randint(90, 190)
            } for _ in range(60)
        ]
        
    def add_hit_particles(self, x, y, color, count=10):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(1.0, 4.0)
            self.particles.append({
                "x": x,
                "y": y,
                "vx": np.cos(angle) * speed,
                "vy": np.sin(angle) * speed,
                "color": color,
                "life": np.random.randint(15, 30),
                "max_life": 30,
                "size": np.random.uniform(2, 5)
            })

    def draw_glow_circle(self, surface, color, center, radius, glow_radius=15, alpha=50):
        """Draws a circle with a neon glow effect using translucent surfaces."""
        # Create a temp surface for blending
        glow_surf = pygame.Surface((radius * 2 + glow_radius * 2, radius * 2 + glow_radius * 2), pygame.SRCALPHA)
        cx = radius + glow_radius
        cy = radius + glow_radius
        
        # Draw layers of glow
        for r in range(glow_radius, 0, -2):
            g_alpha = int(alpha * (1.0 - r / glow_radius))
            pygame.draw.circle(glow_surf, (*color, g_alpha), (cx, cy), radius + r)
            
        # Draw core circle
        pygame.draw.circle(glow_surf, color, (cx, cy), radius)
        
        # Blit to target surface
        surface.blit(glow_surf, (center[0] - cx, center[1] - cy))

    def render(self, agent_x, agent_y, agent_health, max_health,
               enemies, bullets, score, steps_survived,
               last_fired_weapon, shoot_cooldown, difficulty,
               aim_x, aim_y,
               render_mode="human"):
               
        # Process Pygame event queue to avoid "Application not responding"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
                
        # 1. Background Fill (Dark theme)
        bg_color = (11, 14, 20)  # Deeper dark space theme
        self.screen.fill(bg_color)
        
        # Draw ambient starfield
        for star in self.stars:
            star["y"] += star["speed"]
            if star["y"] > self.height:
                star["y"] = 0
                star["x"] = np.random.uniform(0, self.width)
            star_color = (star["brightness"], star["brightness"], int(star["brightness"] * 1.15))
            pygame.draw.circle(self.screen, star_color, (int(star["x"]), int(star["y"])), int(star["size"]))
            
        # 2. Draw Grid Pattern for spatial context
        grid_color = (20, 27, 38)
        for x in range(0, self.width, self.grid_size):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, self.grid_size):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y), 1)
            
        # 3. Update and Draw Particles
        new_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] > 0:
                # Shrink and fade
                ratio = p["life"] / p["max_life"]
                size = max(1, int(p["size"] * ratio))
                alpha = int(255 * ratio)
                # Create particle color with alpha
                p_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(p_surf, (*p["color"], alpha), (size, size), size)
                self.screen.blit(p_surf, (int(p["x"]) - size, int(p["y"]) - size))
                new_particles.append(p)
        self.particles = new_particles
        
        # 4. Draw Bullets (Glowing laser trails)
        for b in bullets:
            # Add micro particles as bullet trails
            if np.random.rand() < 0.5:
                self.particles.append({
                    "x": b.x - b.dx * 10,
                    "y": b.y - b.dy * 10,
                    "vx": -b.dx * np.random.uniform(0.5, 1.5),
                    "vy": -b.dy * np.random.uniform(0.5, 1.5),
                    "color": b.color,
                    "life": 8,
                    "max_life": 8,
                    "size": 2.0
                })
            # Draw bullet with glow
            self.draw_glow_circle(self.screen, b.color, (int(b.x), int(b.y)), int(b.radius), glow_radius=6, alpha=150)
            
        # 5. Draw Enemies
        for enemy in enemies:
            if enemy.alive:
                ex, ey = int(enemy.x), int(enemy.y)
                r = int(enemy.radius)
                
                # Draw main base glow
                self.draw_glow_circle(self.screen, enemy.color, (ex, ey), r, glow_radius=15, alpha=80)
                
                # --- TYPE SPECIFIC ENEMY DRAWING ---
                if enemy.enemy_type == 0:  # Water: Draw concentric ripples
                    pygame.draw.circle(self.screen, enemy.color, (ex, ey), r, width=2)
                    pygame.draw.circle(self.screen, (100, 200, 255), (ex, ey), r - 5, width=1)
                    pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 4)
                elif enemy.enemy_type == 1:  # Grass: Draw leaf/clover shape
                    # 4 leaves
                    leaf_dist = 6
                    pygame.draw.circle(self.screen, enemy.color, (ex - leaf_dist, ey), 8)
                    pygame.draw.circle(self.screen, enemy.color, (ex + leaf_dist, ey), 8)
                    pygame.draw.circle(self.screen, enemy.color, (ex, ey - leaf_dist), 8)
                    pygame.draw.circle(self.screen, enemy.color, (ex, ey + leaf_dist), 8)
                    pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 3)
                elif enemy.enemy_type == 2:  # Fire: Draw fire embers rising
                    if np.random.rand() < 0.2:
                        self.particles.append({
                            "x": enemy.x + np.random.uniform(-10, 10),
                            "y": enemy.y,
                            "vx": np.random.uniform(-0.5, 0.5),
                            "vy": np.random.uniform(-1.5, -0.5),
                            "color": (255, 120, 0),
                            "life": 12,
                            "max_life": 12,
                            "size": np.random.uniform(1.5, 3.0)
                        })
                    pygame.draw.circle(self.screen, enemy.color, (ex, ey), r)
                    # Inner yellow flame core
                    pygame.draw.circle(self.screen, (255, 200, 0), (ex, ey), r - 5)
                    pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 4)
                else:  # Flying: Draw purple wing triangles
                    # Wings expansion/contraction based on game frames
                    wing_offset = int(4 * np.sin(pygame.time.get_ticks() / 100.0))
                    # Left wing
                    pygame.draw.polygon(self.screen, enemy.color, [
                        (ex, ey),
                        (ex - r - wing_offset, ey - 6),
                        (ex - 4, ey + 8)
                    ])
                    # Right wing
                    pygame.draw.polygon(self.screen, enemy.color, [
                        (ex, ey),
                        (ex + r + wing_offset, ey - 6),
                        (ex + 4, ey + 8)
                    ])
                    pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 5)
                
                # Enemy Health Bar
                health_ratio = enemy.health / enemy.max_health
                bar_width = 30
                bar_height = 4
                bar_x = enemy.x - bar_width / 2
                bar_y = enemy.y - enemy.radius - 8
                # Background (red/dark)
                pygame.draw.rect(self.screen, (40, 15, 15), (bar_x, bar_y, bar_width, bar_height), border_radius=2)
                # Fill (green)
                pygame.draw.rect(self.screen, (50, 220, 50), (bar_x, bar_y, int(bar_width * health_ratio), bar_height), border_radius=2)

        # 6. Draw Agent
        agent_color = (255, 255, 255)
        self.draw_glow_circle(self.screen, agent_color, (int(agent_x), int(agent_y)), 15, glow_radius=15, alpha=80)
        
        # Draw dynamic mechanical weapon turret/aiming gun
        aim_len = np.sqrt(aim_x**2 + aim_y**2)
        if aim_len > 0.01:
            ax = aim_x / aim_len
            ay = aim_y / aim_len
            # Determine gun color based on selected weapon
            gun_colors = [
                (0, 120, 255),    # Water (Blue)
                (50, 220, 50),    # Grass (Green)
                (255, 50, 50),    # Fire (Red)
                (220, 220, 220),  # Wind (White)
            ]
            gun_color = gun_colors[last_fired_weapon] if (0 <= last_fired_weapon < 4) else (200, 200, 200)
            
            # Draw gun barrel
            gun_end_x = agent_x + ax * 26
            gun_end_y = agent_y + ay * 26
            pygame.draw.line(self.screen, (30, 30, 30), (int(agent_x), int(agent_y)), (int(gun_end_x), int(gun_end_y)), 7)
            pygame.draw.line(self.screen, gun_color, (int(agent_x), int(agent_y)), (int(gun_end_x), int(gun_end_y)), 3)
            
            # Gun muzzle glow
            pygame.draw.circle(self.screen, gun_color, (int(gun_end_x), int(gun_end_y)), 4)
            
        # Draw central white core of the agent
        pygame.draw.circle(self.screen, (255, 255, 255), (int(agent_x), int(agent_y)), 8)
        
        # Add visual shield/health ring around agent
        health_color = (50, 225, 50)
        if agent_health < 30.0:
            health_color = (255, 50, 50)
        elif agent_health < 70.0:
            health_color = (255, 165, 0)
            
        health_arc_ratio = agent_health / max_health
        if health_arc_ratio > 0:
            pygame.draw.circle(self.screen, health_color, (int(agent_x), int(agent_y)), 20, width=2)
            
        # Draw gun cooldown arc
        if shoot_cooldown > 0:
            cooldown_ratio = shoot_cooldown / 12.0
            rect = pygame.Rect(int(agent_x) - 24, int(agent_y) - 24, 48, 48)
            pygame.draw.arc(self.screen, (255, 255, 255), rect, 0, cooldown_ratio * 2 * np.pi, 2)

        # 7. Draw HUD Overlay
        # Background glassmorphic panel for stats
        hud_panel = pygame.Surface((220, 160), pygame.SRCALPHA)
        hud_panel.fill((20, 30, 45, 180))  # Semi-transparent dark blue
        pygame.draw.rect(hud_panel, (50, 75, 110, 255), (0, 0, 220, 160), width=1, border_radius=8)
        self.screen.blit(hud_panel, (15, 15))
        
        # Text renderings
        title_text = self.font_title.render("ELEMENT SHOOTER", True, (255, 255, 255))
        self.screen.blit(title_text, (25, 22))
        
        score_text = self.font_hud.render(f"SCORE:  {score}", True, (255, 215, 0))
        self.screen.blit(score_text, (25, 55))
        
        time_sec = steps_survived / 60.0
        time_text = self.font_hud.render(f"TIME SURVIVED: {time_sec:.1f}s", True, (200, 220, 255))
        self.screen.blit(time_text, (25, 75))
        
        diff_text = self.font_hud.render(f"DIFFICULTY:  {difficulty:.2f}x", True, (255, 100, 255))
        self.screen.blit(diff_text, (25, 95))
        
        hp_text = self.font_hud.render(f"HEALTH:  {agent_health:.1f}/100.0", True, health_color)
        self.screen.blit(hp_text, (25, 115))
        
        # Health Bar graphic
        hp_bar_width = 190
        hp_bar_height = 8
        pygame.draw.rect(self.screen, (40, 50, 65), (25, 140, hp_bar_width, hp_bar_height), border_radius=4)
        if agent_health > 0:
            pygame.draw.rect(self.screen, health_color, (25, 140, int(hp_bar_width * (agent_health/max_health)), hp_bar_height), border_radius=4)
            
        # Draw Weapon Selection Interface (bottom left panel)
        weapon_panel = pygame.Surface((280, 50), pygame.SRCALPHA)
        weapon_panel.fill((20, 30, 45, 180))
        pygame.draw.rect(weapon_panel, (50, 75, 110, 255), (0, 0, 280, 50), width=1, border_radius=8)
        self.screen.blit(weapon_panel, (15, self.height - 65))
        
        # Draw weapon slots
        weapons_data = [
            ("WATER", (0, 100, 255), "Kills Fire"),
            ("GRASS", (50, 200, 50), "Kills Water"),
            ("FIRE", (255, 50, 50), "Kills Grass"),
            ("WIND", (220, 220, 220), "Kills Fly")
        ]
        
        for idx, (w_name, w_color, counter_desc) in enumerate(weapons_data):
            slot_x = 25 + idx * 65
            slot_y = self.height - 55
            
            # Highlight active slot
            is_active = (idx == last_fired_weapon)
            border_color = (255, 255, 255) if is_active else (60, 80, 100)
            
            pygame.draw.rect(self.screen, border_color, (slot_x, slot_y, 55, 35), width=2 if is_active else 1, border_radius=4)
            
            # Weapon name
            w_text = self.font_ammo.render(w_name, True, w_color if is_active else (120, 140, 160))
            self.screen.blit(w_text, (slot_x + 5, slot_y + 4))
            
            # Weapon subtext
            counter_text = self.font_hud.render(counter_desc.split()[1], True, (160, 180, 200) if is_active else (80, 100, 120))
            self.screen.blit(counter_text, (slot_x + 5, slot_y + 18))

        # 8. Flip Frame Buffer & Limit FPS
        pygame.display.flip()
        self.clock.tick(60)
        
        if render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen)
            
    def close(self):
        pygame.quit()
