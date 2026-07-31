#!/usr/bin/env python3
"""
WM 래치 강제 프로브 (D14) — 개념 forced-choice 방법론을 WM에 적용.

D13 블로커: 자유항법서 WM_latch 음수 = "래치 미작동" 추정이나, (a)에이전트가 A를 떠나는
교란과 (b)막전위 V readout 깨짐(std=0)으로 기계적 규명 미검증.

이 프로브가 둘 다 우회:
  (a) 입력을 명시적으로 통제 — A 적재(게이트 open) 후 게이트 close + 중립관측 = 입력 0.
      "에이전트가 A 떠남"이 아니라 "입력 끊은 뒤 recurrent가 유지하나"를 순수 격리.
  (b) 깨진 V 대신 뉴런별 스파이크카운트 벡터로 WM 패턴 readout (신뢰 가능).

측정: 적재군(A+) vs 대조군(A없음) 지연구간 발화율 궤적 + 적재패턴 유지 상관.
  - A+ 발화율이 대조 위로 유지 → 래치 작동(bistable). D10 음성 반증.
  - A+가 대조로 감쇠 → not-bistable 확정(이번엔 깨끗한 격리+신뢰 계측 위에서).
"""
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def wm_vector(brain):
    """working_memory 뉴런별 스파이크카운트 벡터 (이번 스텝)."""
    ids = np.asarray(brain.working_memory.spike_recording_data[0][1], dtype=np.int64)
    n = brain.config.n_working_memory
    if ids.size == 0:
        return np.zeros(n, dtype=np.float32)
    return np.bincount(ids, minlength=n).astype(np.float32)


def obs_at(env, nx, ny):
    """정규화 좌표 (nx,ny)에 에이전트를 놓고 관측 생성 (place cells 위치 인코딩)."""
    env.agent_x = nx * env.config.width
    env.agent_y = ny * env.config.height
    return env._get_observation()


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a @ b) / d) if d > 1e-9 else 0.0


def run_trial(brain, env, load, K=8, M=25, ax=0.3, ay=0.3, neutral=(0.5, 0.5)):
    obs = env.reset()
    brain.reset()
    for _ in range(20):  # 워밍업
        a, info = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()

    pat_load = None
    if load:
        oA = obs_at(env, ax, ay)
        for _ in range(K):                       # 적재: 게이트 open + A + 도파민
            brain.gate_wm_input(1.0)
            a, info = brain.process(oA)
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
        # 적재패턴 = 마지막 몇 스텝 누적 (단일스텝은 희소·잡음 → 창 누적)
        acc = np.zeros(brain.config.n_working_memory, dtype=np.float32)
        for _ in range(4):
            brain.gate_wm_input(1.0)
            a, info = brain.process(oA)
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
            acc += wm_vector(brain)
        pat_load = acc

    oN = obs_at(env, *neutral)                   # 지연: 게이트 close + 중립(입력0)
    rates, corrs = [], []
    seg = np.zeros(brain.config.n_working_memory, dtype=np.float32)
    delay_acc = np.zeros(brain.config.n_working_memory, dtype=np.float32)  # 지연 전체 누적
    for t in range(M):
        brain.gate_wm_input(0.0)
        a, info = brain.process(oN)
        brain.decay_dopamine()
        rates.append(info.get("working_memory_rate", 0.0))
        v = wm_vector(brain)
        seg += v
        delay_acc += v
        if (t + 1) % 4 == 0:                      # 4스텝 창 누적 패턴 상관 (자기일치)
            if pat_load is not None and pat_load.std() > 0 and seg.std() > 0:
                corrs.append(_corr(seg, pat_load))
            seg = np.zeros(brain.config.n_working_memory, dtype=np.float32)
    return np.array(rates), np.array(corrs), delay_acc, pat_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-weights", required=True)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--delay", type=int, default=25)
    ap.add_argument("--inhib-wm", type=float, default=None,
                    help="wm_inhibitory→WM 되먹임 억제강도 override (음수, 기본 -5.0). 희소성 스윕용.")
    ap.add_argument("--wm-to-inhib", type=float, default=None,
                    help="WM→wm_inhibitory 구동강도 override (양수, 기본 6.0).")
    args = ap.parse_args()

    cfg = ForagerBrainConfig()
    if args.inhib_wm is not None:
        cfg.inhibitory_to_wm_weight = args.inhib_wm
        print(f"[OVERRIDE] inhibitory_to_wm_weight = {args.inhib_wm}")
    if args.wm_to_inhib is not None:
        cfg.wm_to_inhibitory_weight = args.wm_to_inhib
        print(f"[OVERRIDE] wm_to_inhibitory_weight = {args.wm_to_inhib}")
    brain = ForagerBrain(cfg)
    brain.load_all_weights(args.load_weights)
    print(f"Loaded {args.load_weights}")
    env = ForagerGym(ForagerConfig())

    # --- 진단: WM 진짜 rest 발화율 + 벡터 구조 (포화/균일 여부) ---
    obs = env.reset(); brain.reset()
    rest_rates = []
    for _ in range(15):
        brain.gate_wm_input(0.0)
        a, info = brain.process(obs)
        rest_rates.append(info.get("working_memory_rate", 0.0))
    v = wm_vector(brain)
    nz = int((v > 0).sum())
    active_frac = nz / v.size
    print(f"[DIAG] true-rest WM rate (no warmup/load) mean={np.mean(rest_rates):.4f} "
          f"last={rest_rates[-1]:.4f}")
    print(f"[DIAG] WM vector: n={v.size} active_neurons={nz} active_frac={active_frac:.2f} "
          f"std={v.std():.3f} max={v.max():.0f}")
    print(f"[DIAG] wm_inhibitory_rate={info.get('wm_inhibitory_rate', 0.0):.4f} "
          f"(억제뉴런 발화 — 0이면 억제 미작동)")
    print(f"[SPARSE] {'OK 희소' if 0.05 <= active_frac <= 0.35 else ('포화saturated' if active_frac > 0.35 else '과억제silent')} "
          f"(목표 활성비율 0.10~0.20)")

    A_rates, C_rates, A_corrs = [], [], []
    disc_Aplus, disc_ctrl = [], []   # A-기준패턴과의 상관: A+ 지연 vs 대조 지연
    for i in range(args.trials):
        ra, ca, dA, patA = run_trial(brain, env, load=True, M=args.delay)
        rc, _, dC, _ = run_trial(brain, env, load=False, M=args.delay)
        A_rates.append(ra); C_rates.append(rc); A_corrs.append(ca)
        # 핵심 판별: A+ 지연패턴/대조 지연패턴을 같은 A-적재 기준패턴에 상관
        if patA is not None and patA.std() > 0:
            if dA.std() > 0:
                disc_Aplus.append(_corr(dA, patA))
            if dC.std() > 0:
                disc_ctrl.append(_corr(dC, patA))

    A = np.array(A_rates); C = np.array(C_rates)     # (trials, M)
    # 지연 초반/후반 평균
    early = slice(0, max(1, args.delay // 5))
    late = slice(args.delay - max(1, args.delay // 5), args.delay)
    a_early, a_late = A[:, early].mean(), A[:, late].mean()
    c_early, c_late = C[:, early].mean(), C[:, late].mean()
    corr_mean = np.concatenate(A_corrs).mean() if any(len(x) for x in A_corrs) else 0.0

    # 유지도: A+ 지연 발화율이 대조 위로 유지되나 (후반)
    diff_late = a_late - c_late
    sustain = diff_late > 0.02 and a_late > a_early * 0.6   # 대조초과 + 자체 붕괴X

    dAp = float(np.mean(disc_Aplus)) if disc_Aplus else 0.0
    dCt = float(np.mean(disc_ctrl)) if disc_ctrl else 0.0
    discrimination = dAp - dCt
    # 진짜 A-특이 래치: A+ 지연이 A-기준과 유의히 더 닮고, 대조는 덜 닮아야
    specific = discrimination > 0.15 and dAp > 0.4

    print(f"\n=== WM Latch Forced Probe (D14) ===")
    print(f"trials={args.trials} delay={args.delay}")
    print(f"[A+ loaded]  rate early={a_early:.4f}  late={a_late:.4f}")
    print(f"[control]    rate early={c_early:.4f}  late={c_late:.4f}")
    print(f"delay-late A+ minus control (rate) = {diff_late:+.4f}")
    print(f"load-pattern persistence corr (window) = {corr_mean:.3f}")
    print(f"--- 핵심 판별 (A-기준패턴 상관) ---")
    print(f"corr(A+ delay, A-ref)   = {dAp:.3f}")
    print(f"corr(control delay, A-ref) = {dCt:.3f}")
    print(f"DISCRIMINATION (A+ minus ctrl) = {discrimination:+.3f}")
    print(f"A-SPECIFIC LATCH: {'YES' if specific else 'NO'} "
          f"(A+ 지연이 A-기준에 특이적으로 더 닮음)")


if __name__ == "__main__":
    main()
