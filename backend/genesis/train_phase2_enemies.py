"""
Phase 2: Enemy Avoidance Training

Building on Phase 1 (food seeking), now add enemies to learn survival.

Fear circuit → Boost reflex enables escape behavior.
Enemy Eye → Fear Circuit → Boost Motor

Key behaviors to learn:
1. Continue seeking food (Phase 1 skill)
2. Detect enemies via Enemy Eye
3. Boost away when enemies are near
4. Balance risk vs reward
"""

import sys
sys.path.insert(0, r"C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis")

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Headless pygame

from pathlib import Path
from slither_snn_agent import SlitherBrainConfig, SlitherBrain, CHECKPOINT_DIR
from slither_gym import SlitherGym, SlitherConfig

print("=" * 60)
print("Phase 2: Enemy Avoidance Training")
print("=" * 60)

# Phase 2 config: 3 enemies, smaller brain for faster iteration
N_ENEMIES = 3
N_EPISODES = 30

# Use smaller brain (15K neurons) for faster training
brain_config = SlitherBrainConfig(
    n_food_eye=1000, n_enemy_eye=1000, n_body_eye=500,
    n_integration_1=5000, n_integration_2=5000,
    n_motor_left=500, n_motor_right=500, n_motor_boost=300,
    n_fear_circuit=1000, n_hunger_circuit=1000,
    sparsity=0.02
)

# Environment with enemies
env_config = SlitherConfig(n_enemies=N_ENEMIES)

print(f"Enemies: {N_ENEMIES}")
print(f"Episodes: {N_EPISODES}")
print()

brain = SlitherBrain(brain_config)
env = SlitherGym(env_config, render_mode="none")

# Try to load Phase 1 best model (smaller config)
# Note: May not load if Phase 1 used different config
phase1_path = CHECKPOINT_DIR / "best.pt"
if phase1_path.exists():
    try:
        brain.load(phase1_path)
        print("Loaded Phase 1 model as starting point")
    except Exception as e:
        print(f"Could not load Phase 1 model (different config?): {e}")
        print("Starting fresh")
else:
    print("No Phase 1 model found, starting fresh")

# Tracking
scores = []
deaths_by_enemy = 0
deaths_by_wall = 0
best_score = 0

for ep in range(N_EPISODES):
    obs = env.reset()
    brain.reset()
    brain.stats = {
        'food_eaten': 0, 'boosts': 0, 'fear_triggers': 0,
        'left_turns': 0, 'right_turns': 0, 'fatigue_switches': 0
    }

    step = 0
    max_steps = 2000

    while step < max_steps:
        brain.current_heading = obs.get('heading', 0.0)
        sensor = env.get_sensor_input(brain.config.n_rays)
        target_x, target_y, boost = brain.process(sensor)
        obs, reward, done, info = env.step((target_x, target_y, boost))

        if reward != 0:
            brain.process(sensor, reward)

        step += 1

        if done:
            # Determine death cause
            head = env.agent.head
            if head.x < 20 or head.x > env.config.width - 20 or \
               head.y < 20 or head.y > env.config.height - 20:
                deaths_by_wall += 1
            else:
                deaths_by_enemy += 1
            break

    score = info['length']
    food = info.get('foods_eaten', 0)
    boosts = brain.stats['boosts']
    fear = brain.stats['fear_triggers']

    scores.append(score)
    high = max(scores)
    avg = sum(scores[-10:]) / min(len(scores), 10)

    # Save best
    if score > best_score:
        best_score = score
        brain.save(CHECKPOINT_DIR / f"phase2_best_{score}.pt")
        brain.save(CHECKPOINT_DIR / "phase2_best.pt")
        print(f"Episode {ep+1}: Length={score} | Food={food} | Boosts={boosts} | Fear={fear} | [NEW BEST]")
    else:
        print(f"Episode {ep+1}: Length={score} | Food={food} | Boosts={boosts} | Fear={fear} | High={high} Avg={avg:.0f}")

env.close()

print()
print("=" * 60)
print("Phase 2 Training Complete")
print("=" * 60)
print(f"Best Length: {best_score}")
print(f"Average: {sum(scores)/len(scores):.1f}")
print(f"Deaths by enemy: {deaths_by_enemy}")
print(f"Deaths by wall: {deaths_by_wall}")
print(f"Saved: {CHECKPOINT_DIR / 'phase2_best.pt'}")
