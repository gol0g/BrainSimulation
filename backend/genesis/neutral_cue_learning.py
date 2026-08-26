#!/usr/bin/env python3
"""중성 단서 학습 과제 (C29) — 선천 반사로 풀 수 없는 첫 개념 과제.

C28: 개념 4층 점수는 `good_food_to_motor`(가중치 25) 직결 반사의 성능이었다. 그 반사는
**good 태그가 붙은 자극에만** 반응한다. 따라서 개념 형성을 물으려면 good 태그를 쓰지 않아야 한다.

과제: 좌/우에 **중성 단서**(소리 채널)를 제시. 한쪽 단서만 보상과 짝지어져 있다.
- 학습 전: 단서는 아무 의미 없음 → 우연(50%)이어야 정상
- 학습 후: 보상 짝 단서 쪽으로 조향해야 = **학습으로만 획득 가능한 변별**
`good_food_rays`는 양쪽 모두 0으로 두어 직결 반사를 원천 차단한다.

또한 C28의 **좌편향(4.5배 비대칭)** 을 보정: 보상 단서의 좌/우 배정을 균형시키고,
좌편향 기준선을 함께 측정해 편향분을 빼고 판정한다.
"""
import argparse, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def present(obs, nh, cue_side, rewarded_cue_left):
    """중성 단서만 제시. good/bad food 채널은 0(반사 차단)."""
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    for k in ("good_food_rays_left", "good_food_rays_right",
              "bad_food_rays_left", "bad_food_rays_right",
              "food_rays_left", "food_rays_right"):
        if k in o:
            o[k] = np.zeros(nh)
    # 중성 단서 = 소리 채널(gym 실제 채널명: sound_food_*, sound_danger_*).
    # A단서(보상 짝) = 강, B단서 = 약. 스칼라 채널이므로 배열이 아님.
    a_str, b_str = 0.9, 0.3
    left_is_A = (cue_side == "left")
    o["sound_food_left"] = a_str if left_is_A else b_str
    o["sound_food_right"] = b_str if left_is_A else a_str
    # danger 채널은 중립 유지(교란 차단)
    o["sound_danger_left"] = 0.0
    o["sound_danger_right"] = 0.0
    return o


def decide(brain, o, steps=5):
    tot = 0.0
    for _ in range(steps):
        a, _i = brain.process(o)
        tot += a
    return tot


def evaluate(brain, env, obs, nh, trials=100, bias=0.0):
    """A단서(강)가 있는 쪽으로 조향하는가. 좌편향 보정(bias)을 빼고 판정."""
    ok = n = 0
    for t in range(trials):
        side = "left" if (t % 2 == 0) else "right"   # 좌/우 균형 배정
        o = present(obs, nh, side, True)
        tot = decide(brain, o) - bias
        if side == "left" and tot < -0.02:
            ok += 1
        elif side == "right" and tot > 0.02:
            ok += 1
        n += 1
    return ok / max(n, 1) * 100, n


def measure_bias(brain, env, obs, nh, trials=30):
    """좌우 동일 자극에서의 계통 편향(C28: 4.5배 좌편향 확인됨)."""
    vals = []
    for _ in range(trials):
        o = present(obs, nh, "left", True)
        o2 = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in o.items()}
        # 좌우 동일 강도로 중립화 → 남는 조향값이 곧 계통 편향
        o2["sound_food_left"] = 0.6
        o2["sound_food_right"] = 0.6
        vals.append(decide(brain, o2))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    brain = ForagerBrain(ForagerBrainConfig())
    env = ForagerGym(ForagerConfig())
    obs = env.reset()
    for _ in range(20):
        a, _ = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()
    nh = env.config.n_rays // 2

    bias0 = measure_bias(brain, env, obs, nh)
    pre, n = evaluate(brain, env, obs, nh, args.trials, bias0)
    print("[사전] 좌편향 기준선 %+.3f | 학습전 정답률 %.1f%% (우연=50, n=%d)" % (bias0, pre, n))

    # 단서 민감도: 소리 단서가 조향에 도달하기는 하는가.
    # 도달하지 않으면 학습 호출 여부와 무관하게 과제가 원천적으로 풀리지 않는다(가불가 판별).
    sl = np.mean([decide(brain, present(obs, nh, "left", True)) for _ in range(15)])
    sr = np.mean([decide(brain, present(obs, nh, "right", True)) for _ in range(15)])
    print("[민감도] 단서좌 조향 %+.3f | 단서우 조향 %+.3f | 차이 %.4f → 소리→운동 경로 %s"
          % (sl, sr, abs(sl - sr), "있음" if abs(sl - sr) > 0.02 else "**없음(과제 불가)**"))

    # 학습: A단서(강) 쪽을 선택하면 도파민 보상
    rew = 0
    for ep in range(args.episodes):
        for t in range(args.steps):
            side = "left" if (np.random.random() > 0.5) else "right"
            o = present(obs, nh, side, True)
            tot = decide(brain, o, steps=3) - bias0
            chose_left = tot < -0.02
            chose_right = tot > 0.02
            correct = (side == "left" and chose_left) or (side == "right" and chose_right)
            if correct:
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                rew += 1
                # C25에서 확인: 가중치 갱신은 process()가 아니라 **명시 호출**로 일어난다
                # (run_v2_tasks.py 560~565). 이 호출이 없으면 "학습 없음"은 프로브의 산물이 된다.
                cfg = brain.config
                try:
                    if getattr(cfg, "perceptual_learning_enabled", False) and getattr(cfg, "it_enabled", False):
                        brain.update_cortical_rstdp("good_food")
                    if getattr(cfg, "prediction_error_enabled", False):
                        brain.update_prediction_error_rstdp("food")
                except Exception:
                    pass
            else:
                brain.decay_dopamine()
    print("[학습] %dep 완료, 보상 %d회" % (args.episodes, rew))

    bias1 = measure_bias(brain, env, obs, nh)
    post, n = evaluate(brain, env, obs, nh, args.trials, bias1)
    print("[사후] 좌편향 기준선 %+.3f | 학습후 정답률 %.1f%% (우연=50, n=%d)" % (bias1, post, n))
    print("=> 학습 효과 %+.1f%%p | 판정: %s"
          % (post - pre, "학습 있음" if (post - pre) > 10 and post > 60 else "학습 없음/불충분"))


if __name__ == "__main__":
    main()
