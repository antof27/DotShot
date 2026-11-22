---
name: pygame-simulator-craft
description: >-
  Build clean, high-FPS Pygame visualizers decoupled from headless simulation
  physics, featuring tactical arena aesthetics, class silhouettes, MARL
  explainability overlays, and deterministic MP4/GIF replay export.
---

# Pygame Tactical Simulator & Visualization Craft

Use this skill when building or updating `dotshot/render/pygame_renderer.py`, `enjoy.py`, and replay recording utilities.

---

## 1. Strict Decoupling Architecture

- **Headless Training First**: Core physics, collision, line-of-sight, and step logic in `dotshot/core/` must never `import pygame`. Headless training runs at maximum CPU throughput without initializing SDL video drivers.
- **Snapshot-Driven Rendering**: The renderer consumes a pure read-only state snapshot (or environment instance) and draws one frame deterministically. Rendering must never mutate entity positions, RNG state, or reward calculations.

---

## 2. 60 FPS Performance Rules

Legacy v1 allocated a fresh `pygame.Surface(..., pygame.SRCALPHA)` on every bullet and particle every frame. Fix that bottleneck permanently:

- **Pre-bake Glow & Sprite Caches**: On renderer `__init__`, pre-generate radial glow surfaces keyed by `(color, radius, glow_radius)` and store them in a dictionary cache. Render glows via `screen.blit(..., special_flags=pygame.BLEND_RGBA_ADD)` or pre-composited alpha sprites.
- **Cache Fonts & Static Layers**: Build the dark-slate arena background, tactical grid, and static cover drop shadows onto a background `Surface` once per episode reset (only redrawing dynamic shadows or damage flashes as needed).
- **Particle Budget**: Cap active particles (e.g., `<= 400`) using a ring buffer or list comprehension so heavy boss fights stay locked at 60 FPS.

---

## 3. Tactical Visual Language (`1440×960`)

Make state instantly readable both in live play and in paper figures / MP4 clips:

- **Arena Palette**: Deep tactical slate background (`#0B0F17`), muted grid (`#141C2B`), beveled concrete/metal cover blocks (`#1E293B`) with crisp edge highlights (`#38BDF8` / `#475569`) and subtle shadows.
- **Colorblind-Safe Element Coding**: Always pair color with distinct projectile/enemy geometry:
  - **Water**: Cyan-Blue (`#38BDF8`) + circular / droplet ring
  - **Grass**: Emerald-Green (`#22C55E`) + triangle / diamond pulse
  - **Fire**: Ember-Orange/Red (`#F97316`) + sharp star / chevron flare
- **Distinct Agent Class Silhouettes**:
  - **Vanguard (Tank)**: Heavy hexagon + directional frontal shield arc
  - **Striker (DPS)**: Sleek chevron/arrow + thin laser sightline
  - **Support (Tactician)**: Rotating diamond + translucent repair/mark aura ring
- **Combat Feedback**:
  - Central **Energy Core** with pulsing core ring and segmented circular HP arc.
  - **Bosses** carry a distinct outer elemental shield bar above their HP bar; shattering the shield triggers a brief white flash, subtle 4px damped screen shake, and a rising `SHIELD BREAK!` / `COMBO x2` floating combat callout.
- **Clean HUD**: Compact translucent top bar (Stage, Wave, Score, Core HP %, Remaining Enemies, FPS) and bottom agent status pills (HP, cooldown arc, active weapon badge, class icon).

---

## 4. MARL Explainability Overlay & Replay Capture

- **Toggleable (`Tab`) AI Overlay**:
  - Draw faint local vision/LOS cones or 8-ray LIDAR ticks around agents so viewers immediately understand partial observability.
  - Draw target-lock lines from agents to aimed enemies.
  - For `MAT` / `PMAT` / `RoMAT`, render small floating execution-order badges (`#1`, `#2`, `#3`) and dashed teammate attention arcs on top of agents.
- **Deterministic Video Export**:
  - Implement `render_mode="rgb_array"` cleanly using `pygame.surfarray.array3d(self.screen).swapaxes(0, 1)` on an offscreen surface (`SDL_VIDEODRIVER=dummy` supported when headless).
  - Export MP4 clips via `imageio[ffmpeg]` at 30 or 60 FPS directly into `docs/assets/videos/` for embedding in the handbook.
