#!/usr/bin/env python3
"""
맥락 의존 개념 프로브 (C22) — DESIGN_RECOVERY 97행 🔴 미상 항목.

기록: `--context-select`(Zone A 정상 / Zone B에서 good↔bad 의미 반전)는 4월 "M4 v9 hard gate 돌파
(PI 0.17→0.25)"로 작동 보고됐으나 **이번 세션 미검증**. 조합 맥락(C2)은 🔴 미상으로 남아 있음.

측정: **동일한 시각 자극**을 Zone A(왼쪽)와 Zone B(오른쪽)에서 제시하고 조향이 **반대로** 뒤집히는지.
- Zone A: good-typed 쪽으로 접근해야
- Zone B: 같은 good-typed가 실제로는 bad → 반대쪽으로 가야
맥락 개념이 있으면 두 조건에서 선택이 반전(= context-dependent meaning). 무작위=50%.
기준 >60%. 이건 "같은 자극, 다른 의미"를 뇌가 맥락으로 구분하는가 = 개념 형성의 상위 층위.
"""
import argparse, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def run(brain, env, trials=60):
    obs = env.reset(); brain.reset()
    for _ in range(30):
        a, i = brain.process(obs); obs, _, d, _ = env.step((a,))
        if d: obs = env.reset()
    nh = env.config.n_rays // 2
    W, H = env.config.width, env.config.height

    ok_A = ok_B = n = 0
    for t in range(trials):
        good_side = "left" if np.random.random() > 0.5 else "right"

        for zone in ("A", "B"):
            # 에이전트를 해당 zone에 배치(A=왼쪽 절반, B=오른쪽 절반)
            env.agent_x = (0.25 if zone == "A" else 0.75) * W
            env.agent_y = 0.5 * H
            o = env._get_observation()
            o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in o.items()}
            # 동일 자극: good-typed를 good_side에, bad-typed를 반대쪽에
            gl = np.ones(nh) * (0.9 if good_side == "left" else 0.0)
            gr = np.ones(nh) * (0.9 if good_side == "right" else 0.0)
            o["good_food_rays_left"], o["good_food_rays_right"] = gl, gr
            o["bad_food_rays_left"], o["bad_food_rays_right"] = gr, gl
            o["food_rays_left"] = np.ones(nh) * 0.7
            o["food_rays_right"] = np.ones(nh) * 0.7

            tot = 0.0
            for _ in range(5):
                a, i = brain.process(o)
                tot += a
            went_left = tot < -0.02
            went_right = tot > 0.02
            if zone == "A":
                # 정상: good_side로 가야
                if (good_side == "left" and went_left) or (good_side == "right" and went_right):
                    ok_A += 1
            else:
                # 반전: good_side의 반대로 가야(그쪽이 실제 good)
                if (good_side == "left" and went_right) or (good_side == "right" and went_left):
                    ok_B += 1
        n += 1

    a_ = ok_A / max(n, 1) * 100
    b_ = ok_B / max(n, 1) * 100
    return a_, b_, (a_ + b_) / 2, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-weights", default=None)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--hard-gate", action="store_true", help="M4 v9 context HARD gate 활성")
    args = ap.parse_args()
    cfg = ForagerBrainConfig()
    if args.hard_gate:
        cfg.context_hard_gate_enabled = True
    brain = ForagerBrain(cfg)
    if args.load_weights:
        brain.load_all_weights(args.load_weights)
    ecfg = ForagerConfig(); ecfg.context_rules_enabled = True
    env = ForagerGym(ecfg)
    a_, b_, comb, n = run(brain, env, args.trials)
    # combined은 ill-posed(고정매핑만 있어도 A가 천장이라 평균이 올라감).
    # 진짜 지표 = 맥락 선택성: P(good_side로 감|A) - P(good_side로 감|B). 맥락무관=0, 완전반전=+1.
    p_good_A = a_ / 100.0
    p_good_B = 1.0 - b_ / 100.0
    csi = p_good_A - p_good_B
    print(f"CONTEXT: ZoneA(정상) {a_:.1f}% | ZoneB(반전) {b_:.1f}% | combined {comb:.1f}%(ill-posed) "
          f"| CSI={csi:+.3f} (맥락무관=0, 완전반전=+1, n={n}) "
          f"[{'REVERSAL' if b_ > 50 else 'NO-REVERSAL'}]")


if __name__ == "__main__":
    main()
