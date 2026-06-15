import os
import json
import time
import gymnasium as gym
import numpy as np
import google.generativeai as genai
import env  # Register environment

# Load API key from environment
api_key = os.environ.get("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = """
You are an AI playing a 2D top-down shooter game called 'Element Shooter'.
Your goal is to survive as long as possible, kill enemies, and avoid dying.

Game Rules:
- The arena is 800x800. Center is (400, 400). You take damage if enemies touch you.
- There are 4 enemy types and 4 corresponding weapon types. To damage/kill an enemy, you MUST shoot it with the correct countering weapon:
  1. Water Weapon (0) counters Fire Enemy (Red)
  2. Grass Weapon (1) counters Water Enemy (Blue)
  3. Fire Weapon (2) counters Grass Enemy (Green)
  4. Wind Weapon (3) counters Flying Enemy (Purple)
- Shooting the WRONG weapon deals 0 damage and penalizes you.
- There is a weapon cooldown. You cannot shoot constantly.
- You must aim by providing an aiming vector (ax, ay). The bullet will travel in this direction.
- You must move by providing a movement vector (dx, dy).

You must analyze the environment state and output a JSON response matching this exact schema:
{
  "reasoning": "A concise single-sentence explanation of your tactical decision (e.g. 'A Fire enemy is closing in from the East, so I will move West and shoot it with my Water gun').",
  "move_x": -1.0 to 1.0 (float, movement direction X),
  "move_y": -1.0 to 1.0 (float, movement direction Y),
  "aim_x": -1.0 to 1.0 (float, shooting direction X),
  "aim_y": -1.0 to 1.0 (float, shooting direction Y),
  "shoot_weapon": 0 (Water), 1 (Grass), 2 (Fire), 3 (Wind), or -1 (Do not shoot)
}

Be strategic:
- If enemies are too close and you cannot shoot due to cooldown or wrong gun, prioritize evading (moving away).
- Aim directly at the relative position (dx, dy) of the target enemy.
- Keep your output response as pure JSON.
"""

def query_gemini(model, state_description):
    prompt = f"{state_description}\n\nProvide your next action in the specified JSON format."
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Clean markdown formatting if present
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        data = json.loads(text)
        return data
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

def main():
    if not api_key:
        print("WARNING: GEMINI_API_KEY environment variable is not set.")
        print("Please export GEMINI_API_KEY='your-api-key' before running.")
        # We will proceed but it will fail unless they set it.
        user_key = input("Enter Gemini API Key (or press enter to skip if already set/handled): ").strip()
        if user_key:
            genai.configure(api_key=user_key)
        else:
            return
    else:
        genai.configure(api_key=api_key)

    # Use gemini-2.5-flash as the default model
    print("Initializing Gemini model (gemini-2.5-flash)...")
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config={"response_mime_type": "application/json"},
        system_instruction=SYSTEM_INSTRUCTION
    )

    print("Creating environment...")
    env_instance = gym.make("ElementShooter-v0", render_mode="human")
    obs, info = env_instance.reset()
    
    # We want a repeat action rate: LLM decides, we repeat the action for N steps (e.g. 12 steps, which is 0.2s of gametime)
    # This keeps simulation smooth and reduces API calls
    ACTION_REPEAT = 15 
    
    print("Starting LLM Game Loop. Press Ctrl+C in terminal to stop.")
    
    try:
        while True:
            # 1. Get current text representation
            state_desc = env_instance.unwrapped.get_text_description()
            
            # 2. Query LLM
            t0 = time.time()
            decision = query_gemini(model, state_desc)
            latency = time.time() - t0
            
            if decision is None:
                # Default safety action: move center, don't shoot
                action = np.zeros(8, dtype=np.float32)
                reasoning = "API Call Failed. Safely idling."
            else:
                reasoning = decision.get("reasoning", "")
                move_x = float(decision.get("move_x", 0.0))
                move_y = float(decision.get("move_y", 0.0))
                aim_x = float(decision.get("aim_x", 0.0))
                aim_y = float(decision.get("aim_y", 0.0))
                shoot_weapon = int(decision.get("shoot_weapon", -1))
                
                # Print decisions
                print(f"[Gemini - Latency: {latency:.2f}s]")
                print(f"Thought: {reasoning}")
                print(f"Action: Move({move_x:.2f}, {move_y:.2f}) | Aim({aim_x:.2f}, {aim_y:.2f}) | Shoot: {shoot_weapon}\n")
                
                # Convert to Box action space of size 8
                action = np.zeros(8, dtype=np.float32)
                action[0] = move_x
                action[1] = move_y
                action[2] = aim_x
                action[3] = aim_y
                
                if shoot_weapon >= 0 and shoot_weapon < 4:
                    action[4 + shoot_weapon] = 1.0  # Trigger weapon
            
            # 3. Apply action for multiple steps (Action Repeat)
            for _ in range(ACTION_REPEAT):
                # Overlay reasoning on Pygame window if renderer exists
                if env_instance.unwrapped.renderer is not None:
                    # Render the screen
                    obs, reward, terminated, truncated, info = env_instance.step(action)
                    
                    # Draw LLM reasoning on screen
                    screen = env_instance.unwrapped.renderer.screen
                    font = env_instance.unwrapped.renderer.font_hud
                    
                    # Draw glassmorphic background for LLM bubble
                    reasoning_surf = pygame_text_bubble(font, reasoning, max_width=400)
                    screen.blit(reasoning_surf, (env_instance.unwrapped.width - 420, 15))
                    
                    import pygame
                    pygame.display.flip()
                else:
                    obs, reward, terminated, truncated, info = env_instance.step(action)
                    
                if terminated or truncated:
                    print(f"Game Over! Score: {info['score']}, Survived: {info['steps_survived']} steps.")
                    obs, info = env_instance.reset()
                    break
                    
    except KeyboardInterrupt:
        print("\nExiting LLM Agent loop.")
    finally:
        env_instance.close()

def pygame_text_bubble(font, text, max_width=400):
    import pygame
    # Wrap text to fit max_width
    words = text.split(' ')
    lines = []
    current_line = []
    
    for word in words:
        current_line.append(word)
        test_line = ' '.join(current_line)
        if font.size(test_line)[0] > (max_width - 20):
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    
    line_height = font.get_linesize()
    height = len(lines) * line_height + 24
    
    bubble = pygame.Surface((max_width, height), pygame.SRCALPHA)
    # Draw bubble background
    pygame.draw.rect(bubble, (45, 20, 50, 180), (0, 0, max_width, height), border_radius=8)  # purple tinted glass
    pygame.draw.rect(bubble, (160, 80, 220, 255), (0, 0, max_width, height), width=1, border_radius=8)
    
    # Render text lines
    for i, line in enumerate(lines):
        line_surf = font.render(line, True, (240, 220, 255))
        bubble.blit(line_surf, (10, 12 + i * line_height))
        
    return bubble

if __name__ == "__main__":
    main()
