#!/usr/bin/env python3
"""경로 전달 프로브 (C39) — 학습된 D1 가중치가 운동까지 전달되는가.

C38에서 `food_eye→D1` 가중치가 1.0→30.0(30배)으로 변했는데 **조향이 전혀 안 바뀌었다.**
대비 부족만으로는 설명되지 않는다. 단계별로 신호가 어디서 끊기는지 측정한다:

  자극(good 좌/우) → food_eye 발화 → D1 좌/우 발화 → direct 좌/우 → motor 좌/우 → 조향

각 단계에서 **좌 자극 vs 우 자극의 차이**를 재고, 어느 단계에서 차이가 소멸하는지 특정한다.
차이가 D1까지는 살아 있는데 motor에서 사라지면 → D1→motor 전달이 병목.
D1에서 이미 없으면 → 가중치가 커져도 자극 변별이 안 되는 것(포화 등).
"""
import argparse, sys, os, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig

STAGES = ["food_eye_left", "food_eye_right", "d1_left", "d1_right",
          "direct_left", "direct_right", "indirect_left", "indirect_right",
          "motor_left", "motor_right"]


def spikes(brain, name):
    p = getattr(brain, name, None)
    if p is None:
        return None
    try:
        return len(p.spike_recording_data[0][0])
    except Exception:
        return None


def present(obs, nh, good_side):
    o = {k: (np.copy(v) if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    L = 0.9 if good_side == "left" else 0.0
    R = 0.9 if good_side == "right" else 0.0
    o["good_food_rays_left"] = np.ones(nh) * L
    o["good_food_rays_right"] = np.ones(nh) * R
    o["bad_food_rays_left"] = np.zeros(nh)
    o["bad_food_rays_right"] = np.zeros(nh)
    o["food_rays_left"] = np.ones(nh) * L
    o["food_rays_right"] = np.ones(nh) * R
    return o


def measure(brain, obs, nh, good_side, steps=5, trials=10):
    acc = {s: [] for s in STAGES}
    ang = []
    for _ in range(trials):
        o = present(obs, nh, good_side)
        tot = 0.0
        for _ in range(steps):
            a, _i = brain.process(o)
            tot += a
            for s in STAGES:
                v = spikes(brain, s)
                if v is not None:
                    acc[s].append(v)
        ang.append(tot)
    return {s: (float(np.mean(v)) if v else float("nan")) for s, v in acc.items()}, float(np.mean(ang))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-rstdp", action="store_true")
    ap.add_argument("--crossed", action="store_true")
    ap.add_argument("--seed", type=int, default=0,
                    help="C46: 환경·워밍업 난수 시드. 같은 시드면 같은 결과여야 한다(결정론 검증용).")
    ap.add_argument("--direct-motor-w", type=float, default=None,
                    help="C44: direct→motor 가중치(기본 25.0). C42에서 학습 전후 오프셋이 소수점 셋째자리까지 "
                         "동일했다 → 기저핵 출력이 운동에 도달하는 기여가 사실상 0일 가능성 검증.")
    ap.add_argument("--reflex-w", type=float, default=None,
                    help="선천반사 가중치(기본 25.0).")
    ap.add_argument("--direct-inhib", type=float, default=None,
                    help="C56: direct E/I 억제. d1→direct를 낮추면 탈포화되나 D1 영향력이 0이 되므로"
                         "(보상 381→0), 가중치는 유지하고 억제로 포화를 푼다.")
    ap.add_argument("--d1-inhib", type=float, default=None,
                    help="C41: d1 E/I 억제강도(C14 배선). d1이 자극과 무관하게 ~667로 고정 발화(포화)해 "
                         "정보를 담지 못한다. 억제를 넣어 변별이 살아나는지 시험.")
    ap.add_argument("--d1-direct-w", type=float, default=None,
                    help="C40: D1→direct 가중치(기본 20.0, DENSE). 이 값이 커서 direct가 포화되고 "
                         "D1의 좌/우 변별(31.3)이 direct에서 2.0으로 소멸한다.")
    ap.add_argument("--set-d1-weight", type=float, default=None,
                    help="학습을 흉내내어 food_to_d1 가중치를 이 값으로 직접 설정(전달 여부만 격리 검사).")
    args = ap.parse_args()

    cfg = ForagerBrainConfig()
    if args.real_rstdp:
        cfg.real_rstdp = True
    if args.crossed:
        cfg.rstdp_crossed = True
    if args.d1_direct_w is not None:
        cfg.d1_to_direct_weight = args.d1_direct_w
    if args.direct_motor_w is not None:
        cfg.direct_to_motor_weight = args.direct_motor_w
    if args.reflex_w is not None:
        cfg.food_approach_init_w = args.reflex_w
    if args.direct_inhib is not None and args.direct_inhib != 0:
        cfg.direct_inhibition = args.direct_inhib
    if args.d1_inhib is not None and args.d1_inhib != 0:
        # d1_inhibition != 0 이면 뇌가 자동으로 억제뉴런·배선을 만든다(forager_brain.py 1764).
        # 별도 활성 플래그는 없다 — `d1_inhib`은 뉴런집단 속성명이므로 건드리면 안 된다.
        cfg.d1_inhibition = args.d1_inhib
    # C46: 측정 결정론화.
    # GeNN 시드(12345)를 고정했는데도 **같은 설정의 두 런이 다른 부호**를 냈다(C45: direct +10.1 vs −10.3).
    # 원인은 연결이 아니라 **환경**: ForagerGym이 매 런 먹이를 무작위 배치하고, 워밍업 20스텝이
    # 매번 다른 뇌 상태를 만든다. 이것이 이 세션 내내 모든 측정을 흔든 "런 오프셋"의 정체다
    # (C22b 이체제 요동, C28b 부호 뒤집힘, C40 단일런 30.9, C43 사전 72%).
    random.seed(args.seed)
    np.random.seed(args.seed)
    brain = ForagerBrain(cfg)
    env = ForagerGym(ForagerConfig())
    obs = env.reset()
    for _ in range(20):
        a, _ = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()
    nh = env.config.n_rays // 2

    if args.set_d1_weight is not None:
        # 학습 과정을 거치지 않고 가중치만 바꿔 **전달 여부**를 격리 검사한다.
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
                arr = np.array(v, dtype=np.float64)
                arr[:] = args.set_d1_weight
                s.vars["g"].values = arr
                s.vars["g"].push_to_device()
            except Exception as e:
                print("  [WARN] %s 설정 실패: %s" % (nm, e))
        print("[설정] food_to_d1 가중치 = %.1f 로 직접 지정" % args.set_d1_weight)

    L, angL = measure(brain, obs, nh, "left")
    R, angR = measure(brain, obs, nh, "right")

    # 절대 발화량을 함께 본다. 측성차이만 보면 "집단이 아예 침묵해서 차이가 0"인 경우와
    # "활발한데 변별만 없는" 경우를 구분할 수 없다(C39 1차에서 direct가 그랬다).
    print("\n%-14s %11s %11s %11s %11s" % ("단계", "절대(좌자극)", "절대(우자극)", "측성차이", "판정"))
    print("-" * 64)
    for i in range(0, len(STAGES), 2):
        a, b = STAGES[i], STAGES[i + 1]
        absL = L[a] + L[b]
        absR = R[a] + R[b]
        lat = (L[a] - L[b]) - (R[a] - R[b])
        if not (absL == absL):          # nan
            verdict = "측정불가"
        elif absL + absR < 5:
            verdict = "**침묵**"
        elif abs(lat) < 5:
            verdict = "활동O 변별X"
        else:
            verdict = "변별O"
        print("%-14s %11.1f %11.1f %11.1f %11s"
              % (a.replace("_left", ""), absL, absR, lat, verdict))
    print("-" * 64)
    print("%-14s %11.3f %11.3f %11.3f" % ("조향", angL, angR, angL - angR))
    print("\n해석: 각 행의 '차이'가 자극 변별 신호. 위 단계에서 크고 아래에서 0이면 그 사이가 병목.")


if __name__ == "__main__":
    main()
