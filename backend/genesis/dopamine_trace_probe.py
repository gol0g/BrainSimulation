#!/usr/bin/env python3
"""전체 모델의 도파민 궤적을 잰다 (E098 전제 확인).

최소 회로에서 확립된 것:
- **수반성이 없으면 학습이 0%다**(K32, E092: learn 25.0% vs shuffled 0.0%).
- **tau_e ≈ 보상 지연**이 설계 규칙이다(K42, E097).

전체 모델 코드에서 본 것:
- `steer(steps=3)` 직후 보상 → **행동 창 3스텝, 지연 ≈ 0**. 그런데 `tau_e = 200`이다.
- `release_dopamine`은 `dopamine_level += magnitude` 후 ±1로 clip만 한다.
  `decay_dopamine()`은 이 과제에서 **호출되지 않는다**(forager_brain.py:12489는 run_training 전용).
  → 도파민이 누적돼 상한에 머물면 **보상 수반성이 사라진다.**

추측하지 않고 **실제 궤적을 잰다.** 시행마다 정답 여부와 도파민 값을 기록한다.
"""
import argparse, os, random, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerConfig, ForagerGym

ap = argparse.ArgumentParser()
ap.add_argument("--trials", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--d1-inhib", type=float, default=-400.0)      # INV-A4 (규약 P15)
ap.add_argument("--direct-inhib", type=float, default=-100.0)  # INV-A5
ap.add_argument("--real-rstdp", action="store_true", default=True)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed)
cfg = ForagerBrainConfig()
cfg.d1_inhibition = a.d1_inhib
cfg.direct_inhibition = a.direct_inhib
cfg.real_rstdp = True
cfg.genn_seed = 12345 + a.seed        # INV-A1
brain = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig()); obs = env.reset()
for _ in range(20):
    act, _ = brain.process(obs); obs, _, d, _ = env.step((act,))
    if d: obs = env.reset()

print("")
print("=== 도파민 궤적 (%d시행) ===" % a.trials)
print("시행  정답  도파민(파이썬)  |  구간 요약")
vals, corr = [], []
for t in range(a.trials):
    for _ in range(3):                      # steer(steps=3) 와 같은 길이
        brain.process(obs)
    ok = (random.random() < 0.5)            # 정답률 50% 를 모사(실제 과제도 ~50%)
    brain.release_dopamine(reward_magnitude=1.0, primary_reward=True) if ok else \
        brain.release_dopamine(reward_magnitude=-0.5)
    vals.append(brain.dopamine_level); corr.append(ok)
    if t < 12 or t % 50 == 0:
        print("%4d   %s    %+.4f" % (t, "O" if ok else "X", brain.dopamine_level))

v = np.array(vals); c = np.array(corr)
print("")
print("=== 요약 ===")
print("  도파민 범위 %.4f ~ %.4f | 평균 %.4f | std %.4f" % (v.min(), v.max(), v.mean(), v.std()))
print("  0 에 가까운(|d|<0.05) 시행: %d/%d (%.1f%%)" % ((np.abs(v) < 0.05).sum(), len(v),
                                                      (np.abs(v) < 0.05).mean()*100))
print("  후반 50시행 평균 %.4f (std %.4f)" % (v[-50:].mean(), v[-50:].std()))
print("")
print("  정답 시행 평균 도파민 %.4f" % v[c].mean())
print("  오답 시행 평균 도파민 %.4f" % v[~c].mean())
print("  **대비 %.4f**  (0 에 가까우면 수반성이 사라진 것)" % (v[c].mean() - v[~c].mean()))
print("")
print("=> DATRACE min=%.4f max=%.4f mean=%.4f late=%.4f contrast=%.4f"
      % (v.min(), v.max(), v.mean(), v[-50:].mean(), v[c].mean() - v[~c].mean()))
