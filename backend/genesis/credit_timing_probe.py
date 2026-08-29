#!/usr/bin/env python3
"""시간 신용 할당 프로브 (C51) — 자격흔적과 도파민의 시점이 맞는가.

C50까지: 학습기계·교차경로·탐색·포화해소를 다 갖추고 가중치가 시냅스별로 분화해도 행동이 안 변한다.
남은 가설: **시점 어긋남**. 자격흔적 e는 pre-post 상관으로 쌓이고 tau_e=200ms로 감쇠하는데,
도파민이 그보다 늦게(혹은 이르게) 도달하면 흔적이 엉뚱한 순간에 굳는다.

측정: 자극 제시 → N스텝 뒤 도파민 투여, N을 바꿔가며 가중치 변화량을 본다.
  - 특정 지연에서만 |Δg|가 크면 → 시점 창이 좁다(맞추면 학습 가능)
  - 지연과 무관하게 |Δg|가 비슷하면 → 시점 문제가 아니다(흔적이 자극과 무관하게 쌓임)
후자면 **자격흔적이 자극 정보를 담지 못한다**는 뜻이고, 그게 진짜 병목이다.
"""
import argparse, sys, os, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def stim(obs, nh, good_side):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    L = 0.9 if good_side == "left" else 0.0
    R = 0.9 if good_side == "right" else 0.0
    o["good_food_rays_left"] = np.ones(nh) * L
    o["good_food_rays_right"] = np.ones(nh) * R
    o["food_rays_left"] = np.ones(nh) * L
    o["food_rays_right"] = np.ones(nh) * R
    return o


def snap(brain):
    out = {}
    for nm in ("food_to_d1_l", "food_to_d1_r"):
        s = getattr(brain, nm, None)
        if s is None:
            continue
        try:
            s.pull_connectivity_from_device()
            s.vars["g"].pull_from_device()
            v = s.vars["g"].values
            if v is None or (hasattr(v, "size") and v.size == 0):
                v = s.vars["g"].view
            out[nm] = np.array(v, dtype=np.float64).copy()
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()

    print("%-8s %14s %14s %10s" % ("지연(step)", "|Δg| 좌", "|Δg| 우", "좌우 비대칭"))
    print("-" * 52)
    for delay in (0, 2, 5, 10, 20, 50):
        random.seed(args.seed)
        np.random.seed(args.seed)
        cfg = ForagerBrainConfig()
        cfg.real_rstdp = True
        cfg.d1_inhibition = -200.0
        cfg.d1_to_direct_weight = 1.0
        brain = ForagerBrain(cfg)
        env = ForagerGym(ForagerConfig())
        obs = env.reset()
        for _ in range(20):
            a, _ = brain.process(obs)
            obs, _, d, _ = env.step((a,))
            if d:
                obs = env.reset()
        nh = env.config.n_rays // 2

        before = snap(brain)
        # 항상 good=좌 자극만 제시 → 좌 경로에만 흔적이 쌓여야 정상(신용 할당의 최소 조건)
        for _ in range(args.trials):
            o = stim(obs, nh, "left")
            for _ in range(3):
                brain.process(o)
            for _ in range(delay):
                brain.process(o)
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
            brain.process(o)
        after = snap(brain)

        dl = float(np.mean(np.abs(after["food_to_d1_l"] - before["food_to_d1_l"]))) \
            if "food_to_d1_l" in before else float("nan")
        dr = float(np.mean(np.abs(after["food_to_d1_r"] - before["food_to_d1_r"]))) \
            if "food_to_d1_r" in before else float("nan")
        asym = dl - dr
        print("%-8d %14.5f %14.5f %10.5f" % (delay, dl, dr, asym))

    print("\n해석: good=좌 자극만 줬으므로 **좌 경로만** 변해야 정상(비대칭 > 0).")
    print("비대칭이 0에 가까우면 자격흔적이 자극과 무관하게 쌓인다 = 신용 할당 실패.")


if __name__ == "__main__":
    main()
