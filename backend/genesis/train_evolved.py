"""
Training with Evolved Innate Reflexes

The snake is now born with survival instincts:
- Enemy avoidance reflex (pre-wired synapses)
- Still learnable via DA-STDP

This gives the agent TIME to learn more complex strategies.
"""
import sys
sys.path.insert(0, r"C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis")

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from slither_snn_agent import SlitherBrainConfig, SlitherBrain, CHECKPOINT_DIR
from slither_gym import SlitherGym, SlitherConfig
import time

print("=" * 60)
print("Training with EVOLVED Innate Reflexes")
print("Snake is born with survival instincts!")
print("=" * 60)

N_ENEMIES = 5
N_EPISODES = 100
MAX_STEPS = 2000

brain_config = SlitherBrainConfig(
    n_food_eye=1000, n_enemy_eye=1000, n_body_eye=500,
    n_integration_1=5000, n_integration_2=5000,
    n_motor_left=500, n_motor_right=500, n_motor_boost=300,
    n_fear_circuit=1000, n_hunger_circuit=1000,
    sparsity=0.02
)

# Fresh brain with innate reflexes (no loading old model)
brain = SlitherBrain(brain_config)
print("\nStarting with FRESH brain + innate avoidance reflex")

env = SlitherGym(SlitherConfig(n_enemies=N_ENEMIES), render_mode="none")

scores = []
kills_total = 0
best_score = 0
start_time = time.time()

print(f"\nTraining {N_EPISODES} episodes with {N_ENEMIES} enemies...")
print("-" * 60)

for ep in range(N_EPISODES):
    obs = env.reset()
    brain.reset()

    step = 0
    ep_kills = 0

    while step < MAX_STEPS:
        brain.current_heading = obs.get('heading', 0.0)
        sensor = env.get_sensor_input(brain.config.n_rays)
        target_x, target_y, boost = brain.process(sensor)
        obs, reward, done, info = env.step((target_x, target_y, boost))

        if reward > 1:
            ep_kills += int(reward / 5)

        if reward != 0:
            brain.process(sensor, reward)

        step += 1
        if done:
            break

    score = info['length']
    scores.append(score)
    kills_total += ep_kills

    if score > best_score:
        best_score = score
        brain.save(CHECKPOINT_DIR / f"evolved_best_{score}.pt")
        brain.save(CHECKPOINT_DIR / "evolved_best.pt")
        marker = " [NEW BEST]"
    else:
        marker = ""

    if (ep + 1) % 10 == 0 or marker:
        elapsed = time.time() - start_time
        eps_per_sec = (ep + 1) / elapsed
        avg = sum(scores[-10:]) / min(len(scores), 10)
        print(f"Ep {ep+1:3d} | Score: {score:2d} | Kills: {ep_kills} | "
              f"Avg10: {avg:.1f} | Best: {best_score} | "
              f"{eps_per_sec:.2f} ep/s{marker}")

env.close()

elapsed = time.time() - start_time
print("-" * 60)
print(f"Training complete in {elapsed/60:.1f} minutes")
print(f"  Best: {best_score}")
print(f"  Average: {sum(scores)/len(scores):.1f}")
print(f"  Total kills: {kills_total}")
print(f"  Saved: {CHECKPOINT_DIR / 'evolved_best.pt'}")
