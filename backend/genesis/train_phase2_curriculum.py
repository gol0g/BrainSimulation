"""
Phase 2: Curriculum Learning

Start easy (1 enemy) → gradually increase difficulty
Better learning signal = more survival time = more STDP updates
"""
import sys
sys.path.insert(0, r"C:\Users\JungHyun\Desktop\brain\BrainSimulation\backend\genesis")

import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from slither_snn_agent import SlitherBrainConfig, SlitherBrain, CHECKPOINT_DIR
from slither_gym import SlitherGym, SlitherConfig
import time

print("=" * 60)
print("Phase 2: Curriculum Learning")
print("Start easy, gradually increase difficulty")
print("=" * 60)

# Curriculum stages: (n_enemies, target_avg_score, episodes_per_stage)
CURRICULUM = [
    (1, 25, 50),   # Stage 1: 1 enemy, target avg 25
    (2, 20, 50),   # Stage 2: 2 enemies, target avg 20
    (3, 15, 50),   # Stage 3: 3 enemies, target avg 15
    (5, 12, 100),  # Stage 4: 5 enemies, target avg 12
]

MAX_STEPS = 2000

# 15K neuron brain
brain_config = SlitherBrainConfig(
    n_food_eye=1000, n_enemy_eye=1000, n_body_eye=500,
    n_integration_1=5000, n_integration_2=5000,
    n_motor_left=500, n_motor_right=500, n_motor_boost=300,
    n_fear_circuit=1000, n_hunger_circuit=1000,
    sparsity=0.02
)

brain = SlitherBrain(brain_config)

# Start from Phase 1 model (knows how to eat food!)
phase1_path = CHECKPOINT_DIR / "best.pt"
if phase1_path.exists():
    try:
        brain.load(phase1_path)
        print("Starting from Phase 1 best model (food seeking expert)")
    except:
        print("Could not load Phase 1, starting fresh")
else:
    print("No Phase 1 model, starting fresh")

best_score_overall = 0
total_episodes = 0
total_kills = 0
start_time = time.time()

for stage_idx, (n_enemies, target_avg, n_episodes) in enumerate(CURRICULUM):
    print(f"\n{'='*60}")
    print(f"STAGE {stage_idx + 1}: {n_enemies} enemies, target avg: {target_avg}")
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
            brain.save(CHECKPOINT_DIR / f"curriculum_best_{score}.pt")
            brain.save(CHECKPOINT_DIR / "curriculum_best.pt")
            marker = " [BEST!]"
        else:
            marker = ""

        # Progress every 10 episodes
        if (ep + 1) % 10 == 0 or marker:
            avg10 = sum(scores[-10:]) / min(len(scores), 10)
            elapsed = time.time() - start_time
            print(f"  Ep {ep+1:3d} | Score: {score:2d} | Kills: {ep_kills} | "
                  f"Avg10: {avg10:.1f} | StageBest: {stage_best}{marker}")

    # Stage summary
    stage_avg = sum(scores) / len(scores)
    print(f"\n  Stage {stage_idx + 1} Complete:")
    print(f"    Avg: {stage_avg:.1f} (target: {target_avg})")
    print(f"    Best: {stage_best}")
    print(f"    Kills: {stage_kills}")

    env.close()

    # Check if passed stage
    if stage_avg >= target_avg:
        print(f"    [PASSED] Moving to next stage!")
    else:
        print(f"    [BELOW TARGET] Continuing anyway...")

# Final summary
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print("CURRICULUM TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Total episodes: {total_episodes}")
print(f"  Best overall: {best_score_overall}")
print(f"  Total kills: {total_kills}")
print(f"  Time: {elapsed/60:.1f} minutes")
print(f"  Saved: {CHECKPOINT_DIR / 'curriculum_best.pt'}")
