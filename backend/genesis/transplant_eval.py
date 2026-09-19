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

# 이식 대상은 **뇌에서 유도한다**. 손으로 목록을 관리하면 반드시 빠진다.
#
# 2026-09-19 사고: 하드코딩 목록 6종이 `food_to_d1_cross_lr/rl`(--crossed 로 생기는 R-STDP 경로)을
# **빠뜨렸고**, 대신 학습하지 않는 `good_food_to_motor_l/r`(StaticPulse)를 넣고 있었다.
# 그 결과 E084는 "학습된 뇌"가 아니라 **선택된 6종만 옮긴 뇌**를 평가했다.
# E084 사전등록 6번에 "목록 완전성 미검증"이라고 **직접 적어놓고** 확인하지 않은 채 20런을 돌렸다.
#
# `_rstdp_synapses`는 뇌가 도파민 갱신 대상으로 등록한 진짜 학습 시냅스 목록이다
# (forager_brain.py: food_to_d1 L/R + cross_lr/rl + kc_to_d1 L/R).
FALLBACK = ("kc_to_d1_l", "kc_to_d1_r", "food_to_d1_l", "food_to_d1_r",
            "food_to_d1_cross_lr", "food_to_d1_cross_rl")


def learned_names(brain):
    """이 뇌에서 실제로 학습하는 시냅스의 **속성 이름**을 찾아낸다."""
    syns = getattr(brain, "_rstdp_synapses", None)
    if not syns:
        return tuple(n for n in FALLBACK if getattr(brain, n, None) is not None)
    ids = {id(s) for s in syns}
    names = []
    for nm in dir(brain):
        if nm.startswith("__"):
            continue
        try:
            v = getattr(brain, nm)
        except Exception:
            continue
        if id(v) in ids and nm not in names:
            names.append(nm)
    missing = len(ids) - len(names)
    if missing:
        raise RuntimeError("학습 시냅스 %d개의 이름을 찾지 못했다 — 이식이 불완전해진다" % missing)
    return tuple(sorted(names))


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


def build_from_cfg(cfg, seed, env_seed=None, env_cfg=None):
    """**훈련에 쓴 cfg 객체 그대로** 이식 대상 뇌를 만든다.

    2026-09-19 사고: 어댑터가 6개 필드만 새 `ForagerBrainConfig`에 옮겨 담았다.
    그러면 `genn_seed`·`rstdp_crossed`·`food_approach_init_w` 등이 기본값으로 돌아가
    **구조가 다른 뇌**가 만들어진다. 이식은 의미를 잃고, GeNN은 CODE를 재빌드하다
    `cuda error 1: invalid argument`로 죽는다(증상이 원인을 가렸다).

    같은 cfg를 쓰면 (i) 구조가 동일하고 (ii) 재빌드가 없어 충돌도 없다.

    `env_seed`/`env_cfg`도 **훈련에 쓴 것을 그대로** 넘겨야 한다. 안 넘기면 사전·사후가
    서로 다른 환경에서 측정된다(2026-09-19 사고).
    """
    if env_seed is None:
        env_seed = seed
    random.seed(seed); np.random.seed(seed)
    brain = ForagerBrain(cfg)
    # 2026-09-19 사고: 여기서 환경을 **뇌 시드**로 만들고 기본 ForagerConfig를 썼다.
    # 훈련 경로는 환경을 **환경 시드**(--env-seed)와 자기 _ecfg 로 만든다.
    # brain_seed != env_seed 인 뇌(b1~b4)에서 사전·사후가 **다른 환경**을 평가했다는 뜻이다.
    # 먹이 배치·초기 방향이 달라지고, stim()은 음식 채널만 덮어쓰므로 그 차이가 지워지지 않는다.
    random.seed(env_seed); np.random.seed(env_seed)
    env = ForagerGym(env_cfg if env_cfg is not None else ForagerConfig()); obs = env.reset()
    for _ in range(20):
        act, _ = brain.process(obs); obs, _, d, _ = env.step((act,))
        if d:
            obs = env.reset()
    return brain, env, obs


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


def _read_g(syn, nm):
    """가중치 배열을 읽는다.

    2026-09-19: 예전 코드는 `values`가 비면 `view`로 폴백했다. SPARSE 시냅스에서 `view`는
    예외를 던진다(`Only variables associated with DENSE or KERNEL ... use 'values'`).
    하드코딩 6종에서는 우연히 안 걸렸고, 이식 목록을 뇌에서 유도해 교차 경로(SPARSE)가
    들어오자 즉시 터졌다. **폴백을 쓰지 않는다** — 읽히지 않으면 조용히 넘기지 말고 멈춘다.
    """
    # SPARSE 시냅스는 **연결을 먼저 장치에서 가져와야** `values`가 채워진다.
    # 2026-09-19: 이것을 빠뜨려 `food_to_d1_cross_lr 가중치가 비었다`로 다섯 시드 전부 실패했다.
    # (`pathway_transfer_probe.py`는 처음부터 이 순서로 하고 있었다 — 나만 빠뜨린 것이다.)
    try:
        syn.pull_connectivity_from_device()
    except Exception:
        pass   # DENSE 는 이 호출이 없다
    syn.vars["g"].pull_from_device()
    v = syn.vars["g"].values
    if v is None:
        raise RuntimeError("%s 가중치를 읽지 못했다" % nm)
    a = np.asarray(v, dtype=np.float64).ravel()
    if a.size == 0:
        raise RuntimeError("%s 가중치가 비었다 — 이식이 무의미해진다" % nm)
    return a


def pull(brain):
    return {nm: _read_g(getattr(brain, nm), nm).copy() for nm in learned_names(brain)}


def push(brain, w):
    """이식. 대상이 없거나 길이가 다르면 **중단한다**(조용히 자르지 않는다)."""
    for nm, arr in w.items():
        s = getattr(brain, nm, None)
        if s is None:
            raise RuntimeError("이식 대상 없음: %s — 구조가 다른 뇌다" % nm)
        cur = _read_g(s, nm)
        if cur.size != arr.size:
            raise RuntimeError("%s 크기 불일치 %d vs %d — 같은 구조의 뇌가 아니다"
                               % (nm, cur.size, arr.size))
        cur[:] = arr
        s.vars["g"].values = cur
        s.vars["g"].push_to_device()


def verify(src, dst, w):
    """이식 후 목적지 가중치가 원본과 **정확히** 같은지 확인한다. 이식이 조용히 실패하지 않게."""
    bad = []
    for nm, arr in w.items():
        got = _read_g(getattr(dst, nm), nm)
        if not np.array_equal(got, arr):
            bad.append("%s(최대차 %.3g)" % (nm, float(np.abs(got - arr).max())))
    if bad:
        raise RuntimeError("이식 검증 실패: %s" % ", ".join(bad))
    return sorted(w.keys())


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
