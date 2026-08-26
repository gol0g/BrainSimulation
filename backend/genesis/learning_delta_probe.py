#!/usr/bin/env python3
"""학습 델타 프로브 (C25) — 학습이 일어나기는 하는가.

C24 이후 핵심 질문이 "성능이 있느냐"에서 **"학습이 기여하느냐"**로 바뀌었다.
이 프로브는 **로드를 거치지 않는다** — 한 프로세스 안에서 훈련 전/후 가중치를 직접 비교하므로
C24(로드 손상)와 무관하게 판정된다.

측정: 학습 경로 시냅스별로 (1)변화량 |Δg| 평균 (2)변화한 시냅스 비율 (3)분포 표준편차 변화.
- 변화량 ≈ 0 → 학습이 아예 안 일어남(성능은 전부 선천 배선)
- 변화량 크지만 std 그대로 → 전역 스케일만 변함(변별 구조 없음)
- std 증가 → 시냅스별 분화 = 진짜 구조 학습
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

# 학습 경로(C24에서 파괴 대상이던 바로 그 시냅스들)
TARGETS = [
    "food_to_d1_l", "food_to_d1_r", "food_to_d2_l", "food_to_d2_r",
    "good_food_to_d1_l", "good_food_to_d1_r", "bad_food_to_d1_l", "bad_food_to_d1_r",
    "it_food_to_d1_l", "it_food_to_d1_r", "food_to_nac_l", "food_to_nac_r",
]


def snap(brain):
    out = {}
    for name in TARGETS:
        syn = getattr(brain, name, None)
        if syn is None:
            continue
        try:
            syn.vars["g"].pull_from_device()
            v = syn.vars["g"].values
            if v is None:
                v = syn.vars["g"].view
            out[name] = np.array(v, dtype=np.float64).copy()
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    brain = ForagerBrain(ForagerBrainConfig())
    env = ForagerGym(ForagerConfig())

    before = snap(brain)
    print("[snap] 학습경로 시냅스 %d개 기록" % len(before))

    obs = env.reset()
    eaten = 0
    for ep in range(args.episodes):
        obs = env.reset()
        for _ in range(args.steps):
            a, info = brain.process(obs)
            obs, rew, done, inf = env.step((a,))
            if rew and rew > 0:
                brain.release_dopamine(reward_magnitude=float(rew), primary_reward=True)
                eaten += 1
            if done:
                break
    print("[train] %dep 완료, 보상 %d회" % (args.episodes, eaten))

    after = snap(brain)
    print("\n%-24s %10s %10s %12s %12s" % ("시냅스", "|Δ|평균", "변화비율", "std(전)", "std(후)"))
    print("-" * 74)
    tot_d, tot_frac, n = 0.0, 0.0, 0
    for k in before:
        if k not in after or before[k].shape != after[k].shape:
            continue
        d = np.abs(after[k] - before[k])
        frac = float(np.mean(d > 1e-6))
        print("%-24s %10.4f %9.1f%% %12.4f %12.4f"
              % (k, float(np.mean(d)), frac * 100, float(np.std(before[k])), float(np.std(after[k]))))
        tot_d += float(np.mean(d)); tot_frac += frac; n += 1
    if n:
        print("-" * 74)
        print("평균: |Δ|=%.4f | 변화시냅스 %.1f%% (보상 %d회)" % (tot_d / n, tot_frac / n * 100, eaten))
        print("판정: %s" % ("학습 없음(성능=선천배선)" if tot_d / n < 1e-4
                            else "가중치 변화 있음 — std 증감으로 구조학습 여부 판단"))


if __name__ == "__main__":
    main()
