#!/usr/bin/env python3
"""선천 경로 프로브 (C28) — 무학습 뇌가 왜 79~98%를 내는가.

C26/C27에서 소리·맥락 모두 "무학습 ≥ 학습"으로 나왔다. 그렇다면 진짜 질문은
**과제가 학습 없이 배선만으로 풀리게 설계돼 있는가**이다.

가설: `good_food_rays_*`가 접근 회로(motor)로 **직결**돼 있으면 "좋은 음식 쪽으로 조향"은
학습이 아니라 반사다. 개념 프로브 대부분이 바로 이 판별을 요구하므로, 점수는 배선의 성능이 된다.

측정: 무학습 뇌에서 (a)good만 제시 (b)bad만 제시 (c)둘 다 제시할 때 조향이 어디로 향하는가.
학습 없이 good→접근 / bad→회피가 나오면 = 선천 반사 확인.
추가로 시냅스 배선을 직접 조회해 good_food→motor 직결 경로의 존재·강도를 보고한다.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

DIRECT = ["good_food_to_motor_l", "good_food_to_motor_r",
          "food_explore_motor_l", "food_explore_motor_r",
          "food_memory_left_to_motor", "food_memory_right_to_motor"]


def steer(brain, obs, nh, gl, gr, bl, br, steps=5):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    o["good_food_rays_left"] = np.ones(nh) * gl
    o["good_food_rays_right"] = np.ones(nh) * gr
    o["bad_food_rays_left"] = np.ones(nh) * bl
    o["bad_food_rays_right"] = np.ones(nh) * br
    o["food_rays_left"] = np.ones(nh) * max(gl, bl)
    o["food_rays_right"] = np.ones(nh) * max(gr, br)
    tot = 0.0
    for _ in range(steps):
        a, _i = brain.process(o)
        tot += a
    return tot


def main():
    b = ForagerBrain(ForagerBrainConfig())   # 학습 전혀 없음
    env = ForagerGym(ForagerConfig())
    obs = env.reset()
    for _ in range(20):
        a, _ = b.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()
    nh = env.config.n_rays // 2

    print("=== 선천 배선 직결 경로 ===")
    for name in DIRECT:
        syn = getattr(b, name, None)
        if syn is None:
            print("  %-30s 없음" % name)
            continue
        try:
            # SPARSE는 연결을 먼저 pull해야 values가 채워진다(빠뜨리면 n=0으로 오독).
            try:
                syn.pull_connectivity_from_device()
            except Exception:
                pass
            syn.vars["g"].pull_from_device()
            v = syn.vars["g"].values
            if v is None or (hasattr(v, "size") and v.size == 0):
                v = syn.vars["g"].view
            v = np.array(v, dtype=np.float64)
            if v.size == 0:
                print("  %-30s n=0 (값 조회 불가 — 판정 보류)" % name)
            else:
                print("  %-30s n=%-6d 평균g=%.3f" % (name, v.size, float(np.mean(v))))
        except Exception as e:
            print("  %-30s 조회실패(%s)" % (name, e))

    print("\n=== 무학습 뇌 조향 (음수=좌, 양수=우) ===")
    trials = 20
    res = {}
    for label, (gl, gr, bl, br) in {
        "good만 좌": (0.9, 0.0, 0.0, 0.0),
        "good만 우": (0.0, 0.9, 0.0, 0.0),
        "bad만 좌": (0.0, 0.0, 0.9, 0.0),
        "bad만 우": (0.0, 0.0, 0.0, 0.9),
        "good좌+bad우": (0.9, 0.0, 0.0, 0.9),
        "good우+bad좌": (0.0, 0.9, 0.9, 0.0),
    }.items():
        vals = [steer(b, obs, nh, gl, gr, bl, br) for _ in range(trials)]
        m, s = float(np.mean(vals)), float(np.std(vals))
        res[label] = m
        print("  %-14s 조향 %+.3f ± %.3f" % (label, m, s))

    print("\n=== 판정 ===")
    good_ok = (res["good만 좌"] < -0.02) and (res["good만 우"] > 0.02)
    bad_ok = (res["bad만 좌"] > 0.02) and (res["bad만 우"] < 0.02)
    conf_ok = (res["good좌+bad우"] < -0.02) and (res["good우+bad좌"] > 0.02)
    print("  good→접근 반사(학습 없이): %s" % ("YES" if good_ok else "NO"))
    print("  bad→회피 반사(학습 없이):  %s" % ("YES" if bad_ok else "NO"))
    print("  경합시 good 우선(학습 없이): %s" % ("YES" if conf_ok else "NO"))
    if good_ok:
        print("  → 개념 프로브 점수는 **선천 반사**로 설명됨(학습 불필요).")


if __name__ == "__main__":
    main()
