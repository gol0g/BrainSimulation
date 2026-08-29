#!/usr/bin/env python3
"""반사 역전 학습 과제 (C34) — 학습이 선천 반사를 이길 수 있는가.

C29 실패에서 배운 것: "중성 단서"로 고른 `sound_food_*`도 선천 경로가 있어(변조폭 0.415)
훈련 0회에 100%가 나왔다. 반사가 없는 채널을 찾는 접근은 취약하다.

대신 **반사와 반대 방향을 보상**한다. good_food가 보이는 쪽의 **반대쪽**으로 조향해야 보상.
- 선천 반사(good→접근, C28b 변조폭 0.835)가 **정답을 방해**한다
- 따라서 성적이 오르려면 학습이 반사를 **이겨야만** 한다 → 학습 능력의 직접 시험
- 학습 전 기대값: 반사 때문에 우연 이하(0~30%)
- 학습이 작동하면: 상승. 작동 안 하면: 그대로

C28b에서 확인된 **런마다 변하는 상수 오프셋**을 매 평가마다 측정해 빼고 판정한다.
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
    o["bad_food_rays_left"] = np.zeros(nh)
    o["bad_food_rays_right"] = np.zeros(nh)
    o["food_rays_left"] = np.ones(nh) * L
    o["food_rays_right"] = np.ones(nh) * R
    return o


def steer(brain, o, steps=5, bias_side=None, bias_strength=0.0, bias_at_d1=False):
    """bias_side가 주어지면 **매 스텝** 운동 집단에 편향을 주입한다.
    C37 1차 실패 원인: 편향을 호출 전 한 번만 넣었더니 감각 입력이 다음 스텝에 덮어써서
    탐색 5975회에 정답 표본이 0개였다. 탐색은 실제로 행동이 바뀔 만큼 주입해야 의미가 있다."""
    tot = 0.0
    for _ in range(steps):
        if bias_side is not None and bias_strength > 0.0:
            # C52: 탐색 주입 지점을 **학습 시냅스 상류(D1)** 로 옮긴다.
            # 기존엔 motor에 주입했는데, 학습 시냅스는 food_eye→D1로 그보다 상류다.
            # 강제된 행동이 D1을 거치지 않으므로 자격흔적에는 **자극이 만든 원래(반사정렬)
            # D1 패턴**이 기록되고, 보상이 그걸 강화한다 → 실측: 변조폭이 반사 방향으로 +0.036.
            # D1에 주입하면 흔적이 탐색한 상태를 담아 보상이 그 연합을 강화할 수 있다.
            _targets = (("d1_left", bias_side == "left"), ("d1_right", bias_side == "right")) \
                if bias_at_d1 else \
                (("motor_left", bias_side == "left"), ("motor_right", bias_side == "right"))
            for nm, want in _targets:
                p = getattr(brain, nm, None)
                if p is None:
                    continue
                try:
                    p.vars["V"].pull_from_device()
                    v_ = p.vars["V"].view
                    v_[:] += (bias_strength if want else -bias_strength)
                    p.vars["V"].push_to_device()
                except Exception:
                    pass
        a, _i = brain.process(o)
        tot += a
    return tot


def measure_offset(brain, obs, nh, n=20):
    """좌우 대칭 자극에서 남는 조향 = 런 상수 오프셋(C28b: 런마다 0.3~0.83으로 요동)."""
    vals = []
    for _ in range(n):
        o = stim(obs, nh, "left")
        o["good_food_rays_left"] = np.ones(nh) * 0.45
        o["good_food_rays_right"] = np.ones(nh) * 0.45
        o["food_rays_left"] = np.ones(nh) * 0.45
        o["food_rays_right"] = np.ones(nh) * 0.45
        vals.append(steer(brain, o))
    return float(np.mean(vals))


def evaluate(brain, obs, nh, trials=100):
    """정답률과 **변조폭**을 함께 반환.

    C49에서 드러난 결함: 반사를 없애면 조향이 거의 0이라 |v|<0.02 임계에 걸려 좌·우 양쪽 다
    오답 처리된다 → 0.0%는 "틀림"이 아니라 **"결정 안 함"**. 이 지표로는 학습의 미세 변화를
    원리적으로 못 잡는다. C28b에서 이미 얻은 교훈(절대부호·임계 판정은 ill-posed, 좌↔우
    **차이값**만 견고)을 이 프로브에 적용하지 않았던 것.

    변조폭 = mean(조향 | good=우) − mean(조향 | good=좌).
      반사(good쪽 접근)면 양수, 반사역전 학습이 성공하면 **음수 방향으로 이동**해야 한다.
    """
    off = measure_offset(brain, obs, nh)
    ok = 0
    vs_left, vs_right = [], []
    for t in range(trials):
        side = "left" if (t % 2 == 0) else "right"     # 좌우 균형
        v = steer(brain, stim(obs, nh, side)) - off     # 오프셋 보정
        (vs_left if side == "left" else vs_right).append(v)
        # 정답 = good의 **반대쪽**
        if side == "left" and v > 0.02:
            ok += 1
        elif side == "right" and v < -0.02:
            ok += 1
    mod = float(np.mean(vs_right)) - float(np.mean(vs_left))
    return ok / trials * 100, off, mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--reflex-w", type=float, default=None,
                    help="선천 반사 가중치(기본 25.0). 낮추면 학습이 이길 여지가 생기는지 시험.")
    ap.add_argument("--real-rstdp", action="store_true",
                    help="C36: food_to_d1을 시냅스별 자격흔적 R-STDP로(기본은 정적=학습불가).")
    ap.add_argument("--rstdp-eta", type=float, default=0.02)
    ap.add_argument("--crossed", action="store_true",
                    help="C36 수리1: 학습가능 교차경로(food_eye_L→D1_R) 신설. 없으면 매핑 재학습 불가.")
    ap.add_argument("--no-reward", action="store_true",
                    help="C63: 도파민을 **한 번도 주지 않고** 동일 횟수만큼 처리만 한다. "
                         "양 조건 공통의 +0.02~0.04 변조폭 표류가 학습 때문인지 "
                         "단순 처리(적응·잔류전류) 때문인지 가리는 대조. 표류가 그대로면 학습과 무관.")
    ap.add_argument("--cortical-eta", type=float, default=None,
                    help="C62: 피질 전역스칼라 학습률(기본 0.0008). 이것이 두 조건 공통으로 "
                         "변조폭을 +0.02~0.04 표류시켜(C60/C61) 측정하려는 효과(~0.003)를 10배로 덮는다. "
                         "0으로 두면 공통항이 제거돼 R-STDP 효과가 드러날 수 있다.")
    ap.add_argument("--direct-inhib", type=float, default=None,
                    help="C56/C60: direct E/I 억제. d1→direct를 낮추면 D1 영향력이 0이 되므로"
                         "(C55), 가중치는 20으로 유지하고 억제로 포화를 푼다.")
    ap.add_argument("--hippo-eta", type=float, default=None,
                    help="C54: 해마 학습률(place→food_memory, 기본 0.15). 0으로 두면 해마 학습을 끈다. "
                         "C50의 학습 신호(+0.013)가 기저핵이 아니라 해마에서 온 것인지 판별용 "
                         "(C53: D1을 300까지 밀어도 행동 변화 0 = 기저핵은 행동 제어 불가).")
    ap.add_argument("--bias-at-d1", action="store_true",
                    help="C52: 탐색 편향을 motor 대신 **D1**(학습 시냅스 하류 첫 단계)에 주입. "
                         "motor 주입은 학습 시냅스보다 하류라 자격흔적이 탐색행동을 담지 못했다.")
    ap.add_argument("--d1-direct-w", type=float, default=None,
                    help="C45: d1→direct 가중치(기본 20.0). 20이면 direct가 666으로 포화해 "
                         "d1의 변별을 통과시키지 못한다. d1억제와 **함께** 낮춰야 신호가 지난다.")
    ap.add_argument("--seed", type=int, default=0,
                    help="C46: 환경·워밍업 시드. 조건 비교는 같은 시드로 짝지어라.")
    ap.add_argument("--d1-inhib", type=float, default=None,
                    help="C41: d1 E/I 억제(-200 권장). 없으면 d1이 ~667로 포화해 자극 정보를 담지 못하고, "
                         "학습·교차·탐색을 다 갖춰도 기저핵을 통과하지 못한다.")
    ap.add_argument("--bias", type=float, default=25.0,
                    help="C37: 탐색 시 운동 편향 세기(매 스텝 주입). 6.0은 반사에 압도돼 정답표본 0개였음.")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="C37: 행동 탐색 확률. 0이면 결정론적 정책이라 정답 표본이 0개 → 학습 불가.")
    ap.add_argument("--w-max", type=float, default=None,
                    help="C36 수리3: 학습 상한(기본 30.0). 선천반사 25.0과 경쟁하려면 그 이상 필요.")
    args = ap.parse_args()

    # C46: 환경·워밍업 난수 고정. 미고정이면 사전 정답률이 런마다 0%~72%로 흔들려
    # 학습 효과가 잡음에 묻힌다(C43에서 실제로 그랬다).
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = ForagerBrainConfig()
    if args.reflex_w is not None:
        cfg.food_approach_init_w = args.reflex_w
    if args.real_rstdp:
        cfg.real_rstdp = True
        cfg.real_rstdp_eta = args.rstdp_eta
    if args.crossed:
        cfg.rstdp_crossed = True
    if args.w_max is not None:
        cfg.real_rstdp_w_max = args.w_max
    if args.d1_inhib is not None and args.d1_inhib != 0:
        cfg.d1_inhibition = args.d1_inhib   # !=0 이면 뇌가 억제뉴런·배선을 자동 생성
    if args.d1_direct_w is not None:
        cfg.d1_to_direct_weight = args.d1_direct_w
    if args.cortical_eta is not None:
        cfg.cortical_rstdp_eta = args.cortical_eta
    if args.direct_inhib is not None and args.direct_inhib != 0:
        cfg.direct_inhibition = args.direct_inhib
    if args.hippo_eta is not None:
        cfg.place_to_food_memory_eta = args.hippo_eta
    brain = ForagerBrain(cfg)
    env = ForagerGym(ForagerConfig())
    obs = env.reset()
    for _ in range(20):
        a, _ = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()
    nh = env.config.n_rays // 2

    def snap_d1():
        """C36 진단: food_to_d1 가중치가 실제로 변하는가.
        '학습이 안 일어남'과 '학습은 됐는데 행동에 안 닿음'을 분리한다."""
        out = {}
        for nm in ("food_to_d1_l", "food_to_d1_r"):
            s = getattr(brain, nm, None)
            if s is None:
                continue
            try:
                try:
                    s.pull_connectivity_from_device()
                except Exception:
                    pass
                s.vars["g"].pull_from_device()
                v = s.vars["g"].values
                if v is None or (hasattr(v, "size") and v.size == 0):
                    v = s.vars["g"].view
                out[nm] = np.array(v, dtype=np.float64).copy()
            except Exception:
                pass
        return out

    d1_before = snap_d1()
    pre, off0, mod0 = evaluate(brain, obs, nh, args.trials)
    print("[사전] 오프셋 %+.3f | 정답률 %.1f%% | **변조폭 %+.4f** (양수=반사방향, 음수=역전)"
          % (off0, pre, mod0))

    def explore_bias(target_side, strength=6.0):
        """C37: **행동 탐색 주입**.
        이 뇌의 조향은 결정론적이라 정답(반사 반대쪽)을 한 번도 내지 않고, 그래서 양성 보상이
        0회가 된다(D/E 조건에서 실측). 보상 기반 학습은 강화할 행동 표본이 있어야 부트스트랩된다.
        운동 집단 막전위에 편향을 넣어 대안 행동을 실제로 발생시키고, 그때의 활동에 자격흔적이
        쌓이게 한다(사후 라벨링이 아니라 실제 행동을 만들어야 STDP가 그 행동을 학습한다)."""
        for nm, want in (("motor_left", target_side == "left"), ("motor_right", target_side == "right")):
            p = getattr(brain, nm, None)
            if p is None:
                continue
            try:
                p.vars["V"].pull_from_device()
                v_ = p.vars["V"].view
                v_[:] += (strength if want else -strength * 0.5)
                p.vars["V"].push_to_device()
            except Exception:
                pass

    rew = 0
    explored = 0
    eps = args.epsilon
    for ep in range(args.episodes):
        off = measure_offset(brain, obs, nh, n=5)
        for t in range(args.steps):
            side = "left" if (np.random.random() > 0.5) else "right"
            # 정답 = good의 반대쪽. ε 확률로 그 행동을 실제로 유도해 표본을 만든다.
            do_explore = (np.random.random() < eps)
            if do_explore:
                # C38 수정: 이전 판은 **항상 정답 방향으로** 유도해 보상률이 98%가 됐다.
                # 도파민이 상수가 되니 전 시냅스가 균일하게 천장까지 자라고 std가 0으로 붕괴
                # (= 변별 소멸). 진짜 ε-탐욕은 **무작위 방향**으로 탐색하고 우연히 맞았을 때만
                # 보상해야 대비가 생겨 시냅스별 변별이 학습된다.
                explored += 1
                probe_side = "left" if (np.random.random() < 0.5) else "right"
                v = steer(brain, stim(obs, nh, side), steps=3,
                          bias_side=probe_side, bias_strength=args.bias,
                          bias_at_d1=args.bias_at_d1) - off
            else:
                v = steer(brain, stim(obs, nh, side), steps=3) - off
            correct = (side == "left" and v > 0.02) or (side == "right" and v < -0.02)
            if args.no_reward:
                continue          # C63: 처리만 하고 도파민·학습 호출을 전혀 하지 않는다
            if correct:
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                rew += 1
                c = brain.config
                try:
                    if getattr(c, "perceptual_learning_enabled", False) and getattr(c, "it_enabled", False):
                        brain.update_cortical_rstdp("good_food")
                    if getattr(c, "prediction_error_enabled", False):
                        brain.update_prediction_error_rstdp("food")
                except Exception:
                    pass
            else:
                brain.release_dopamine(reward_magnitude=-0.5)
    print("[학습] %dep 완료, 보상 %d회 (탐색 주입 %d회, ε=%.2f)" % (args.episodes, rew, explored, eps))

    d1_after = snap_d1()
    for nm in sorted(d1_before):
        if nm in d1_after and d1_before[nm].shape == d1_after[nm].shape:
            b_, a_ = d1_before[nm], d1_after[nm]
            d = np.abs(a_ - b_)
            print("[D1가중치] %-14s |Δ|평균=%.5f 변화율=%.1f%% | 평균 %.3f→%.3f | std %.4f→%.4f"
                  % (nm, d.mean(), (d > 1e-9).mean() * 100, b_.mean(), a_.mean(), b_.std(), a_.std()))

    post, off1, mod1 = evaluate(brain, obs, nh, args.trials)
    print("[사후] 오프셋 %+.3f | 정답률 %.1f%% | **변조폭 %+.4f**" % (off1, post, mod1))
    dmod = mod1 - mod0
    print("=> 정답률 %+.1f%%p | **변조폭 변화 %+.4f** | 판정: %s"
          % (post - pre, dmod,
             "학습이 조향을 역전 방향으로 이동" if dmod < -0.02
             else ("학습이 반사 방향으로 강화" if dmod > 0.02 else "변화 없음")))


if __name__ == "__main__":
    main()
