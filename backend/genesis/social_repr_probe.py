#!/usr/bin/env python3
"""
사회 표상 프로브 (C8) — "표상 단계 부재" 추정을 측정으로 확정.

C7까지: 사회 개념 9접근 실패, 관찰회로 열면 행동만 개선. "표상 단계가 없다"고 추정만 함.
이 프로브: NPC 단서(호출 좌/우) 제시 시 사회 표상 집단(mirror_food, social_memory, tom_intention,
vicarious_reward)이 **차등 반응**하는지 스파이크로 직접 측정.
- 좌/우 단서에 표상 활동이 다르면 → 표상은 있고 read-out(행동 연결)이 문제.
- 단서 유무/좌우에 반응 자체가 없으면 → 표상 단계 부재 확정(= 진짜 아키텍처 결손).
WM 포화를 강제 프로브로 규명한 것과 동형 방법.
"""
import argparse
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

POPS = ["mirror_food", "social_memory", "tom_intention", "social_observation", "sts_social", "motor_left", "motor_right", "good_food_eye_left", "good_food_eye_right"]


def rate(brain, name):
    pop = getattr(brain, name, None)
    if pop is None:
        return None
    try:
        n = len(pop.spike_recording_data[0][0])
        return n / max(1, pop.num_neurons if hasattr(pop, "num_neurons") else 100)
    except Exception:
        return None


def run(brain, env, trials=40):
    obs = env.reset(); brain.reset()
    for _ in range(30):
        a, i = brain.process(obs); obs, _, d, _ = env.step((a,))
        if d: obs = env.reset()

    acc = {p: {"left": [], "right": [], "none": []} for p in POPS}
    for t in range(trials):
        for cond in ("left", "right", "none"):
            o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
            nh = env.config.n_rays // 2
            for key in ("food_rays_left", "food_rays_right", "good_food_rays_left",
                        "good_food_rays_right", "bad_food_rays_left", "bad_food_rays_right"):
                o[key] = np.zeros(nh)
            o["npc_call_food_left"] = 0.8 if cond == "left" else 0.0
            o["npc_call_food_right"] = 0.8 if cond == "right" else 0.0
            o["npc_eating_left"] = 0.8 if cond == "left" else 0.0
            o["npc_eating_right"] = 0.8 if cond == "right" else 0.0
            o["npc_near_food"] = 0.0 if cond == "none" else 0.8
            o["social_proximity"] = 0.0 if cond == "none" else 0.7
            # ★수정: agent_rays(NPC 시각)가 사회표상 주입력 — 초판 프로브가 빠뜨림
            # 측정 검증용: 음식 시각도 좌/우로 줌(반응 확실한 집단 대조)
            o["good_food_rays_left"] = np.ones(nh) * (0.9 if cond == "left" else 0.0)
            o["good_food_rays_right"] = np.ones(nh) * (0.9 if cond == "right" else 0.0)
            o["agent_rays_left"] = np.ones(nh) * (0.8 if cond == "left" else 0.0)
            o["agent_rays_right"] = np.ones(nh) * (0.8 if cond == "right" else 0.0)
            for _ in range(3):
                brain.process(o)
            for p in POPS:
                r = rate(brain, p)
                if r is not None:
                    acc[p][cond].append(r)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-weights", default=None)
    ap.add_argument("--mirror-motor", type=float, default=None)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    cfg = ForagerBrainConfig()
    if args.mirror_motor is not None:
        cfg.mirror_to_motor_weight = args.mirror_motor
        cfg.tom_to_motor_weight = args.mirror_motor
    brain = ForagerBrain(cfg)
    if args.load_weights:
        brain.load_all_weights(args.load_weights)
    env = ForagerGym(ForagerConfig())
    acc = run(brain, env, args.trials)
    print("\n=== 사회 표상 프로브 (C8) ===")
    print(f"{'population':>18} {'none':>8} {'left':>8} {'right':>8} {'cue-effect':>11} {'LR-diff':>9}")
    for p in POPS:
        d = acc[p]
        if not d["none"]:
            print(f"{p:>18}   (집단 없음/기록 불가)")
            continue
        n_, l_, r_ = np.mean(d["none"]), np.mean(d["left"]), np.mean(d["right"])
        cue = (l_ + r_) / 2 - n_
        lr = l_ - r_
        print(f"{p:>18} {n_:>8.3f} {l_:>8.3f} {r_:>8.3f} {cue:>+11.3f} {lr:>+9.3f}")
    print("cue-effect>0 = NPC단서에 반응(표상 있음) / LR-diff≠0 = 방향 구분(사용가능 표상)")


if __name__ == "__main__":
    main()
