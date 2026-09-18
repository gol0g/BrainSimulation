#!/usr/bin/env python3
"""평가 중에 학습이 정말 멈추는가 (외부 검토 2026-09-18 지적 #1 검증).

주장: `reset()`은 파이썬 변수 `dopamine_level`만 0으로 두고 장치 파라미터를 갱신하지 않는다
(`_push_dopamine_to_rstdp()` 미호출). 그리고 reflex 과제는 `decay_dopamine()`을 한 번도
부르지 않는다 → 마지막 보상 때 장치에 밀어넣은 도파민이 평가 내내 남아 R-STDP가 계속 돈다.

그러면 evaluate()의 전제("남는 차이는 가중치뿐", C64 주석)가 깨진다. 측정 중에 가중치가 변하기 때문이다.

측정: 보상을 준 뒤 → 가중치 스냅샷 → evaluate()와 같은 절차 실행 → 다시 스냅샷.
가중치가 변하면 **평가가 뇌를 바꾸고 있다**.
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kc-rstdp", action="store_true", default=True)
ap.add_argument("--kc-d1-w", type=float, default=None)
ap.add_argument("--kc-w-max", type=float, default=None)
# INV-A4/A5 기본값 고정 (규약 P15)
ap.add_argument("--d1-inhib", type=float, default=-400.0)
ap.add_argument("--direct-inhib", type=float, default=-100.0)
ap.add_argument("--fix", action="store_true",
                help="평가 전에 도파민을 장치까지 0으로 밀어넣고 같은 측정을 반복(양성 대조).")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed)
cfg = ForagerBrainConfig()
cfg.kc_rstdp = True
cfg.d1_inhibition = a.d1_inhib
cfg.direct_inhibition = a.direct_inhib
if a.kc_d1_w is not None: cfg.kc_to_d1_init_w = a.kc_d1_w
if a.kc_w_max is not None: cfg.kc_real_rstdp_w_max = a.kc_w_max

brain = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig()); obs = env.reset()
for _ in range(20):
    act, _ = brain.process(obs); obs, _, d, _ = env.step((act,))
    if d: obs = env.reset()
nh = env.config.n_rays // 2

# 도파민만 막고 다른 가소성이 살아 있으면 같은 문제다. 학습 시냅스를 전부 본다.
SYNS = tuple(n for n in ("kc_to_d1_l", "kc_to_d1_r", "food_to_d1_l", "food_to_d1_r",
                         "good_food_to_motor_l", "good_food_to_motor_r") )


def snap():
    out = {}
    for nm in SYNS:
        s = getattr(brain, nm, None)
        if s is None: continue
        s.vars["g"].pull_from_device()
        v = s.vars["g"].values
        if v is None or (hasattr(v, "size") and v.size == 0):
            v = s.vars["g"].view
        out[nm] = np.array(v, dtype=np.float64).ravel().copy()
    return out


def report(before, after, label):
    print("  %s" % label)
    for nm in SYNS:
        if nm not in before: continue
        b, c = before[nm], after[nm]
        n = min(len(b), len(c))
        d = c[:n] - b[:n]
        changed = float((np.abs(d) > 1e-9).mean() * 100.0)
        print("    %-12s 평균 %.4f→%.4f  |Δ|평균 %.5f  최대 %.5f  **변한 시냅스 %.1f%%**"
              % (nm, b[:n].mean(), c[:n].mean(), np.abs(d).mean(), np.abs(d).max(), changed))


print("\n=== 1) 학습(보상 100회) ===")
for i in range(100):
    brain.process(obs)
    brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
print("  파이썬 dopamine_level = %.4f" % brain.dopamine_level)

print("\n=== 2) 평가 절차를 그대로 실행 (reset + 안정화 30 + 조향 측정) ===")
if a.fix:
    brain.dopamine_level = 0.0
    brain._push_dopamine_to_rstdp()
    print("  [양성대조] 평가 전에 도파민을 장치까지 0으로 밀어넣음")

pre = snap()
brain.reset()
print("  reset() 직후 파이썬 dopamine_level = %.4f" % brain.dopamine_level)
mid = snap()
for _ in range(30):
    brain.process(obs)
post_stab = snap()
# evaluate()의 조향 측정과 같은 분량의 process 호출
for _ in range(100 * 5):
    brain.process(obs)
post = snap()

print("\n=== 3) 결과 ===")
report(pre, mid, "reset() 자체:")
report(mid, post_stab, "안정화 30스텝:")
report(post_stab, post, "조향 측정 500스텝:")
report(pre, post, "**평가 전체 합계**:")
print("\n해석: 평가 구간에서 가중치가 변하면, evaluate()가 재는 것은 '훈련이 끝난 뇌'가 아니다.")
