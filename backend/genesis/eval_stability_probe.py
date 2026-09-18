#!/usr/bin/env python3
"""같은 뇌를 여러 번 평가하면 같은 값이 나오는가 (검토 제안 0단계, 규약 P11).

E083의 사전 판정 기준은 ±0.008이다. 그런데 **같은 뇌를 두 번 재서 0.008 이상 흔들리면
그 기준은 의미가 없다.** 검출력을 먼저 확인하라는 P11이 요구하는 것이 이것이다.

여기서는 `reflex_override_task.evaluate()`를 **그대로 import해서** 쓴다.
프로브가 평가 절차를 복사하면 원본과 갈라진다 — 실제로 INV-B4 누락이 그렇게 숨어 있었다.
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig
import reflex_override_task as T   # 평가 함수를 공유한다 (복사하지 않는다)

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--stab", type=int, default=30,
                help="평가 전 안정화 스텝. 기본 30은 부족함이 확인됐다.")
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--kc-d1-w", type=float, default=None)
ap.add_argument("--kc-rstdp", action="store_true")
ap.add_argument("--kc-w-max", type=float, default=750.0)
ap.add_argument("--d1-inhib", type=float, default=-400.0)    # INV-A4 (규약 P15)
ap.add_argument("--direct-inhib", type=float, default=-100.0)  # INV-A5
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed)
cfg = ForagerBrainConfig()
cfg.d1_inhibition = a.d1_inhib
cfg.direct_inhibition = a.direct_inhib
cfg.kc_real_rstdp_w_max = a.kc_w_max
if a.kc_rstdp: cfg.kc_rstdp = True
if a.kc_d1_w is not None: cfg.kc_to_d1_init_w = a.kc_d1_w

brain = ForagerBrain(cfg)
env = ForagerGym(ForagerConfig()); obs = env.reset()
for _ in range(20):
    act, _ = brain.process(obs); obs, _, d, _ = env.step((act,))
    if d: obs = env.reset()
nh = env.config.n_rays // 2

print("\n=== 같은 뇌를 %d번 연속 평가 (학습 없음) ===" % a.repeats)
mods, offs, accs = [], [], []
for i in range(a.repeats):
    acc, off, mod = T.evaluate(brain, obs, nh, a.trials, stab=a.stab)
    mods.append(mod); offs.append(off); accs.append(acc)
    print("  %d회차: 오프셋 %+.4f | 정답률 %5.1f%% | **변조폭 %+.6f**" % (i + 1, off, acc, mod))

m = np.array(mods)
print("\n=== 평가 재현성 ===")
print("  변조폭 평균 %+.6f  std **%.6f**  최대-최소 **%.6f**" % (m.mean(), m.std(), m.max() - m.min()))
print("  오프셋 std %.6f" % np.std(offs))
print("\n판정 기준 대비:")
for th in (0.003, 0.008):
    print("  기준 %.3f vs 평가잡음(최대-최소) %.6f → %s"
          % (th, m.max() - m.min(),
             "**기준이 잡음보다 작다 = 검출 불가**" if (m.max() - m.min()) >= th else "기준이 잡음보다 큼 (사용 가능)"))
