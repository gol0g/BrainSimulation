"""
Evolved Innate Reflexes + Curriculum Learning

Best of both worlds:
- Innate avoidance reflex (진화된 본능)
- Progressive difficulty (커리큘럼)
"""
import sys
sys.path.insert(0, r"C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis")

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from slither_snn_agent import SlitherBrainConfig, SlitherBrain, CHECKPOINT_DIR
from slither_gym import SlitherGym, SlitherConfig
import time

print("=" * 60)
print("EVOLVED + CURRICULUM Training")
print("Innate reflexes + Progressive difficulty")
print("=" * 60)

CURRICULUM = [
    (1, 30, 40),   # Stage 1: 1 enemy, target avg 30
    (2, 25, 40),   # Stage 2: 2 enemies
    (3, 20, 40),   # Stage 3: 3 enemies
    (5, 15, 80),   # Stage 4: 5 enemies (longer)
]

MAX_STEPS = 2000

brain_config = SlitherBrainConfig(
    n_food_eye=1000, n_enemy_eye=1000, n_body_eye=500,
    n_integration_1=5000, n_integration_2=5000,
    n_motor_left=500, n_motor_right=500, n_motor_boost=300,
    n_fear_circuit=1000, n_hunger_circuit=1000,
    sparsity=0.02
)

# Fresh brain WITH innate reflexes
brain = SlitherBrain(brain_config)
print("\nFresh brain with EVOLVED innate avoidance reflex")

best_score_overall = 0
total_episodes = 0
total_kills = 0
start_time = time.time()

for stage_idx, (n_enemies, target_avg, n_episodes) in enumerate(CURRICULUM):
    print(f"\n{'='*60}")
    print(f"STAGE {stage_idx + 1}: {n_enemies} enemies, target: {target_avg}")
    print(f"{'='*60}")

    env = SlitherGym(SlitherConfig(n_enemies=n_enemies), render_mode="none")

    scores = []
    stage_kills = 0
    stage_best = 0

    for ep in range(n_episodes):
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
        stage_kills += ep_kills
        total_kills += ep_kills
        total_episodes += 1

        if score > stage_best:
            stage_best = score

        if score > best_score_overall:
            best_score_overall = score
            brain.save(CHECKPOINT_DIR / f"evolved_cur_best_{score}.pt")
            brain.save(CHECKPOINT_DIR / "evolved_cur_best.pt")
            marker = " [BEST!]"
        else:
            marker = ""

        if (ep + 1) % 10 == 0 or marker:
            avg10 = sum(scores[-10:]) / min(len(scores), 10)
            print(f"  Ep {ep+1:3d} | Score: {score:2d} | Kills: {ep_kills} | "
                  f"Avg10: {avg10:.1f} | Best: {stage_best}{marker}")

    stage_avg = sum(scores) / len(scores)
    print(f"\n  Stage {stage_idx + 1} Complete: Avg={stage_avg:.1f}, Best={stage_best}, Kills={stage_kills}")

    env.close()

elapsed = time.time() - start_time
print(f"\n{'='*60}")
print("EVOLVED + CURRICULUM COMPLETE")
print(f"{'='*60}")
print(f"  Total episodes: {total_episodes}")
print(f"  Best overall: {best_score_overall}")
print(f"  Total kills: {total_kills}")
print(f"  Time: {elapsed/60:.1f} minutes")
print(f"  Saved: {CHECKPOINT_DIR / 'evolved_cur_best.pt'}")
