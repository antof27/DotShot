---
name: game-visual-documentation
description: >-
  Capture deterministic offscreen 1440x960 in-game screenshots across all four
  DotShot stages, curate annotated gameplay replays, and write qualitative
  visual analyses for papers and handbooks.
---

# Game Visual Documentation and Screenshot Capture

Use this skill when capturing visual evidence of **DotShot Game v2** mechanics, UI/HUD clarity, and MARL policy overlays for `docs/` and `docs/paper/`.

---

## 1. Deterministic Offscreen Screenshot Standards

- Always run headless capture with `SDL_VIDEODRIVER=dummy` and `DotShotPygameRenderer(..., render_mode="rgb_array")` so captures reproduce identically on CI or headless servers.
- Set a fixed simulation seed before capturing each stage so entities, projectiles, cover geometry, and callouts appear consistently.
- Stage scenes that display real tactical mechanics rather than empty spawn frames:
  - **Stage 1**: Striker maneuvering around cover pillars, firing blue Water projectiles at advancing Fire triangles.
  - **Stage 2**: Mixed Water/Grass/Fire threats, incoming orange enemy shots, and a white neutral ally crossing the arena.
  - **Stage 3**: Two Strikers defending the pulsing Central Energy Core against a Shielded Boss with `Tab` MARL overlay (`#1/#2` priority badges and LIDAR rays) enabled.
  - **Stage 4**: Heterogeneous squad showing `Vanguard` with active cyan Aegis shield, `Striker` firing on a marked enemy, and `Support` emitting a green Repair/Mark pulse near the Core.

---

## 2. Qualitative Captioning & Analysis

Every embedded screenshot in Markdown or LaTeX must be accompanied by:
1. A concise caption identifying stage, active mechanics, and visual encodings (colors + geometric silhouettes).
2. A short qualitative analysis explaining how the visual layout connects to the Dec-POMDP challenges (LOS occlusion, Boss shield break, role specialization).
