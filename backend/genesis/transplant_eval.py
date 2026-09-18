#!/usr/bin/env python3
"""가중치 이식 평가 — "남는 차이는 가중치뿐"을 실제로 성립시킨다.

## 왜 필요한가
2026-09-18 측정:
- **런 간 잡음은 0**이다 (같은 커맨드 3회 = 소수점까지 동일).
- 그러나 평가값은 **측정 시작 상태에 결정론적으로 민감**하다.
  안정화 30/100/300/1000 → 0.5546 / 0.5767 / 0.5543 / 0.5774 (폭 0.028).
  에피소드 수만 3→6으로 바꾸면 사후 변조폭 0.0903~0.1614 (폭 **0.0711**, 검출 목표의 8.9배).
- `brain.reset()` + 안정화로는 이 의존성이 지워지지 않는다(C64가 시도했으나 실패).

이력 의존성은 **결정론적**이므로 반복 평균으로 지워지지 않는다. 축을 바꿔야 한다.

## 방법
1. 훈련 뇌에서 학습 시냅스 가중치를 뽑는다.
2. **같은 시드로 새 뇌를 만든다** — 훈련 이력이 0인 뇌.
3. 거기에 가중치를 이식한다.
4. 평가한다.

모든 조건이 **동일한 동역학 이력**(갓 만든 뇌 + 같은 워밍업)을 지나므로,
조건 간 차이로 남는 것은 **가중치뿐**이다. 결정론이 확인됐으므로 이 절차는 재현된다.

## 검증 (이 파일이 스스로 하는 것)
- **A. 항등성**: 이식할 가중치가 초기값과 같으면(=학습 안 한 뇌), 이식 평가값이
  갓 만든 뇌의 평가값과 **정확히** 같아야 한다. 다르면 이식 자체가 뇌를 바꾼 것이다.
- **B. 이력 무관성**: 같은 가중치를 서로 다른 훈련 길이의 뇌에서 뽑아 이식했을 때
  평가값이 같아야 한다. → 이력 의존성이 제거됐는지의 직접 증거.
"""
import sys, os, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig
import reflex_override_task as T   # 평가 함수를 공유한다 (복사하지 않는다)

# 이식 대상 = 학습으로 변할 수 있는 시냅스. 여기 없는 경로가 학습하면 이식이 불완전해진다.
LEARNED = ("kc_to_d1_l", "kc_to_d1_r", "food_to_d1_l", "food_to_d1_r",
           "good_food_to_motor_l", "good_food_to_motor_r")


def make_cfg(a):
    cfg = ForagerBrainConfig()
    cfg.d1_inhibition = a.d1_inhib
    cfg.direct_inhibition = a.direct_inhib
    cfg.kc_real_rstdp_w_max = a.kc_w_max
    if a.kc_rstdp:
        cfg.kc_rstdp = True
    if a.kc_d1_w is not None:
        cfg.kc_to_d1_init_w = a.kc_d1_w
    return cfg


def build(a):
    """시드를 고정해 뇌를 만들고 **고정 길이** 워밍업만 준다. 모든 조건이 같은 이력을 갖는다."""
    random.seed(a.seed); np.random.seed(a.seed)
    brain = ForagerBrain(make_cfg(a))
    env = ForagerGym(ForagerConfig()); obs = env.reset()
    for _ in range(20):
        act, _ = brain.process(obs); obs, _, d, _ = env.step((act,))
        if d:
            obs = env.reset()
    return brain, env, obs


def pull(brain):
    out = {}
    for nm in LEARNED:
        s = getattr(brain, nm, None)
        if s is None:
            continue
        s.vars["g"].pull_from_device()
        v = s.vars["g"].values
        if v is None or (hasattr(v, "size") and v.size == 0):
            v = s.vars["g"].view
        out[nm] = np.array(v, dtype=np.float64).ravel().copy()
    return out


def push(brain, w):
    """이식. 길이가 다르면 **연결 구조가 다른 뇌**라는 뜻이므로 중단한다(조용히 자르지 않는다)."""
    for nm, arr in w.items():
        s = getattr(brain, nm, None)
        if s is None:
            raise RuntimeError("이식 대상 없음: %s" % nm)
        s.vars["g"].pull_from_device()
        v = s.vars["g"].values
        if v is None or (hasattr(v, "size") and v.size == 0):
            v = s.vars["g"].view
        cur = np.array(v, dtype=np.float64)
        if cur.size != arr.size:
            raise RuntimeError("%s 크기 불일치 %d vs %d — 같은 시드로 만든 뇌가 아니다"
                               % (nm, cur.size, arr.size))
        cur[:] = arr
        s.vars["g"].values = cur
        s.vars["g"].push_to_device()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--stab", type=int, default=30)
    ap.add_argument("--kc-d1-w", type=float, default=None)
    ap.add_argument("--kc-rstdp", action="store_true")
    ap.add_argument("--kc-w-max", type=float, default=750.0)
    ap.add_argument("--d1-inhib", type=float, default=-400.0)      # INV-A4 (규약 P15)
    ap.add_argument("--direct-inhib", type=float, default=-100.0)  # INV-A5
    ap.add_argument("--churn", type=int, nargs="*", default=[0, 200, 600],
                    help="검증 B: 이식 **전에** 뇌를 이만큼 더 돌려 이력을 벌린다. "
                         "이식 평가가 이력에 무관하면 세 값이 같아야 한다.")
    a = ap.parse_args()

    print("\n=== A. 항등성: 초기 가중치를 그대로 이식하면 갓 만든 뇌와 같은가 ===")
    b0, env0, obs0 = build(a)
    nh = env0.config.n_rays // 2
    base_w = pull(b0)
    acc0, off0, mod0 = T.evaluate(b0, obs0, nh, a.trials, stab=a.stab)
    print("  갓 만든 뇌            : 변조폭 %+.6f" % mod0)

    b1, env1, obs1 = build(a)
    push(b1, base_w)
    acc1, off1, mod1 = T.evaluate(b1, obs1, nh, a.trials, stab=a.stab)
    print("  초기 가중치 이식      : 변조폭 %+.6f   차이 %.8f %s"
          % (mod1, abs(mod1 - mod0), "**일치**" if abs(mod1 - mod0) < 1e-9 else "**불일치 — 이식이 뇌를 바꿨다**"))

    print("\n=== B. 이력 무관성: 이력을 벌린 뇌에서 뽑은 **같은** 가중치를 이식 ===")
    print("  (가중치는 동일하므로, 평가값이 다르면 그 차이는 전부 이력 탓이다)")
    got = []
    for c in a.churn:
        bs, envs, obss = build(a)
        for _ in range(c):                      # 이력만 벌린다. 보상 없음 = 가중치 불변
            bs.process(obss)
        w = pull(bs)
        same = all(np.array_equal(w[k], base_w[k]) for k in base_w)
        # 직접 평가 (이식 없음) — 이력 의존성이 그대로 드러나는 경로
        accd, offd, modd = T.evaluate(bs, obss, nh, a.trials, stab=a.stab)
        # 이식 평가 — 갓 만든 뇌로 옮겨서 평가
        bt, envt, obst = build(a)
        push(bt, w)
        acct, offt, modt = T.evaluate(bt, obst, nh, a.trials, stab=a.stab)
        got.append((c, same, modd, modt))
        print("  churn %-4d 가중치동일=%-5s | 직접평가 %+.6f | **이식평가 %+.6f**"
              % (c, same, modd, modt))

    dd = [g[2] for g in got]; tt = [g[3] for g in got]
    print("\n=== 결과 ===")
    print("  직접평가   최대-최소 **%.6f**" % (max(dd) - min(dd)))
    print("  이식평가   최대-최소 **%.6f**" % (max(tt) - min(tt)))
    print("\n판정: 이식평가 폭이 검출 목표(0.008)보다 작아야 이 축을 쓸 수 있다.")


if __name__ == "__main__":
    main()
