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
import argparse, sys, os
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


def steer(brain, o, steps=5, bias_side=None, bias_strength=0.0):
    """bias_side가 주어지면 **매 스텝** 운동 집단에 편향을 주입한다.
    C37 1차 실패 원인: 편향을 호출 전 한 번만 넣었더니 감각 입력이 다음 스텝에 덮어써서
    탐색 5975회에 정답 표본이 0개였다. 탐색은 실제로 행동이 바뀔 만큼 주입해야 의미가 있다."""
    tot = 0.0
    for _ in range(steps):
        if bias_side is not None and bias_strength > 0.0:
            for nm, want in (("motor_left", bias_side == "left"),
                             ("motor_right", bias_side == "right")):
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
    off = measure_offset(brain, obs, nh)
    ok = 0
    for t in range(trials):
        side = "left" if (t % 2 == 0) else "right"     # 좌우 균형
        v = steer(brain, stim(obs, nh, side)) - off     # 오프셋 보정
        # 정답 = good의 **반대쪽**
        if side == "left" and v > 0.02:
            ok += 1
        elif side == "right" and v < -0.02:
            ok += 1
    return ok / trials * 100, off


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
    pre, off0 = evaluate(brain, obs, nh, args.trials)
    print("[사전] 오프셋 %+.3f | 반사역전 정답률 %.1f%% (우연=50)" % (off0, pre))

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
                          bias_side=probe_side, bias_strength=args.bias) - off
            else:
                v = steer(brain, stim(obs, nh, side), steps=3) - off
            correct = (side == "left" and v > 0.02) or (side == "right" and v < -0.02)
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

    post, off1 = evaluate(brain, obs, nh, args.trials)
    print("[사후] 오프셋 %+.3f | 반사역전 정답률 %.1f%%" % (off1, post))
    print("=> 학습 효과 %+.1f%%p | 판정: %s"
          % (post - pre, "학습이 반사를 이김" if (post - pre) > 10 else "학습이 반사를 못 이김"))


if __name__ == "__main__":
    main()
