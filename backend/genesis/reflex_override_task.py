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


def act_window_current(brain, o, ex_side, current, steps):
    """E115: 행동 창 — 실행 motor 에 +current, 반대 motor 에 −current 를 **지속 전류**로(Ioffset) 넣고 steps 처리.
    motor Ioffset 이 동적 파라미터여야 한다(cfg.motor_ioffset_dynamic). 끝나면 0으로 되돌린다.
    반환: 창 동안 (실행 motor 발화율 합, 반대 motor 발화율 합)."""
    ml, mr = brain.motor_left, brain.motor_right
    ml.set_dynamic_param_value("Ioffset", current if ex_side == "left" else -current)
    mr.set_dynamic_param_value("Ioffset", current if ex_side == "right" else -current)
    ex_r = ot_r = 0.0
    try:
        for _ in range(steps):
            _a, info = brain.process(o)
            l, r = info["motor_left_rate"], info["motor_right_rate"]
            ex_r += (l if ex_side == "left" else r)
            ot_r += (r if ex_side == "left" else l)
    finally:
        ml.set_dynamic_param_value("Ioffset", 0.0)
        mr.set_dynamic_param_value("Ioffset", 0.0)
    return ex_r, ot_r


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


def evaluate(brain, obs, nh, trials=100, stab=30):
    """정답률과 **변조폭**을 함께 반환.

    C49에서 드러난 결함: 반사를 없애면 조향이 거의 0이라 |v|<0.02 임계에 걸려 좌·우 양쪽 다
    오답 처리된다 → 0.0%는 "틀림"이 아니라 **"결정 안 함"**. 이 지표로는 학습의 미세 변화를
    원리적으로 못 잡는다. C28b에서 이미 얻은 교훈(절대부호·임계 판정은 ill-posed, 좌↔우
    **차이값**만 견고)을 이 프로브에 적용하지 않았던 것.

    변조폭 = mean(조향 | good=우) − mean(조향 | good=좌).
      반사(good쪽 접근)면 양수, 반사역전 학습이 성공하면 **음수 방향으로 이동**해야 한다.
    """
    # C64: **상태 정규화**. C63에서 도파민을 한 번도 주지 않아도 변조폭이 +0.025~0.038 똑같이
    # 표류함이 확인됐다(보상 있음과 차이 0.0002~0.0011). 사후 측정이 수천 스텝 처리 뒤에 일어나
    # 뇌의 동역학 상태(적응·잔류전류·도파민)가 사전과 달랐던 것 = **서로 다른 상태의 뇌를 비교**.
    # 측정 직전에 상태를 초기화하고 동일한 안정화를 거치면, 남는 차이는 **가중치뿐**이다.
    brain.reset()
    # 2026-09-18: 안정화 길이를 인자로 뺐다. 30스텝은 **부족하다** — 같은 뇌를 연속 평가하면
    # 1회차만 튀고(변조폭 0.5609) 2회차부터 0.5524~0.5542로 가라앉는다. 사전 측정이 그 1회차라
    # 모든 조건의 '변화'에 음수 편향이 들어간다(pilot에서 무학습 칸도 -0.007).
    for _ in range(stab):
        brain.process(obs)

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
    ap.add_argument("--kc-gamma", action="store_true", help="E081/H015: KC→D1 가중치 감마분포")
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
    ap.add_argument("--reflex-sparsity", type=float, default=None,
                    help="E070/H004: 반사 대역폭(good_food_eye→motor sparsity, 기본 0.15).")
    ap.add_argument("--learn-sparsity", type=float, default=None,
                    help="E070/H004: 학습경로 대역폭(food_eye→D1 sparsity, 기본 0.08).")
    ap.add_argument("--seed", type=int, default=0,
                    help="C46: 환경·워밍업 시드. 조건 비교는 같은 시드로 짝지어라.")
    ap.add_argument("--d1-lateral", type=float, default=None,
                    help="E080/H014: D1 좌↔우 측면억제. 전역억제와 달리 승자를 가려 변별을 만든다.")
    ap.add_argument("--kc-rstdp", action="store_true",
                    help="E079/H013: KC→D1을 시냅스별 자격흔적 R-STDP로. 기본은 집단 스칼라 갱신이라 신용할당이 없다.")
    ap.add_argument("--kc-d1-w", type=float, default=None,
                    help="E082/H004: KC→D1 초기 가중치(기본 0.5). K19 — 기본값에서 KC가 D1 발화의 "
                         "0.6%%만 움직인다. 학습 경로의 출력 영향력을 키우는 손잡이.")
    ap.add_argument("--transplant-eval", action="store_true",
                    help="INV-B5: 학습된 가중치를 **같은 시드로 새로 만든 뇌**에 이식해 평가한다. "
                         "직접 평가는 뇌의 동역학 이력에 의존해 가중치가 동일해도 0.028~0.045 흔들린다"
                         "(K22). 이식하면 이력이 모든 조건에서 같아져 그 폭이 0.000000이 된다.")
    ap.add_argument("--dump-kc-weights", action="store_true",
                    help="E082 조작검증: 학습 후 kc_to_d1 가중치 통계를 출력. "
                         "--kc-d1-w를 w_max보다 크게 주면 클램프로 깎이는지 확인하는 용도.")
    ap.add_argument("--tau-e", type=float, default=None,
                    help="E098/K42: R-STDP 자격흔적 시정수(기본 200). 이 과제는 시행이 3스텝인데 "
                         "tau 200이면 시행당 0.985만 감쇠해 66시행 뒤에도 37%%가 남는다 — "
                         "현재 보상이 과거 수십 시행의 흔적에 배정된다.")
    ap.add_argument("--kc-w-max", type=float, default=None,
                    help="E082: KC R-STDP 가중치 상한(기본 30). --kc-d1-w를 30보다 크게 주면 "
                         "학습이 도로 상한까지 깎아내리므로 함께 올려야 한다. "
                         "두 조건에서 **같은 값**을 써야 초기 영향력만 비교된다.")
    ap.add_argument("--n-food", type=int, default=None,
                    help="E077: 먹이 개수(기본 45). env6이 n_good=42로 최다였다.")
    ap.add_argument("--food-ratio", type=float, default=None,
                    help="E077: 좋은 먹이 비율. env6이 0.750으로 최고였다.")
    ap.add_argument("--symmetric-env", action="store_true",
                    help="H008/E076: 먹이를 좌우 균등 배치해 비대칭만 제거(다른 특성 보존).")
    ap.add_argument("--brain-seed", type=int, default=None,
                    help="E073: 뇌 연결 난수 시드(미지정시 --seed 사용). 분산 출처 분리용.")
    ap.add_argument("--env-seed", type=int, default=None,
                    help="E073: 환경·워밍업 난수 시드(미지정시 --seed 사용). 분산 출처 분리용.")
    ap.add_argument("--d1-inhib", type=float, default=None,
                    help="C41: d1 E/I 억제(-200 권장). 없으면 d1이 ~667로 포화해 자극 정보를 담지 못하고, "
                         "학습·교차·탐색을 다 갖춰도 기저핵을 통과하지 못한다.")
    ap.add_argument("--bias", type=float, default=25.0,
                    help="C37: 탐색 시 운동 편향 세기(매 스텝 주입). 6.0은 반사에 압도돼 정답표본 0개였음.")
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="C37: 행동 탐색 확률. 0이면 결정론적 정책이라 정답 표본이 0개 → 학습 불가.")
    ap.add_argument("--w-max", type=float, default=None,
                    help="C36 수리3: 학습 상한(기본 30.0). 선천반사 25.0과 경쟁하려면 그 이상 필요.")
    ap.add_argument("--kc-motor", action="store_true",
                    help="E109: KC→motor 4방향 R-STDP 학습 경로(버섯체 MBON 유사). D1 경로 권한 부족(K52) 대안.")
    ap.add_argument("--kc-motor-w-max", type=float, default=2.0)
    ap.add_argument("--kc-motor-init-w", type=float, default=0.5)
    ap.add_argument("--kc-motor-eta", type=float, default=0.001)
    ap.add_argument("--kc-motor-sparsity", type=float, default=0.05)
    ap.add_argument("--calib-kc-motor", action="store_true",
                    help="E109 조작검증: KC→motor 가중치를 '완전 역전 학습'(교차=w_max, 같은쪽=0)과 "
                         "'반사 정렬'(반대)로 직접 넣고 변조폭을 재 종료(학습 없음) — 이 경로의 행동 권한.")
    ap.add_argument("--reward-window", type=int, default=0,
                    help="E109: 보상 후 같은 자극을 유지한 채 처리할 스텝 수(도파민이 이번 시행 흔적에 작용). 0=이전 동작")
    ap.add_argument("--act-window", type=int, default=0,
                    help="E114: 행동 결정 후 자극 유지 + 실행 motor 구동·반대쪽 억제 스텝 수(0=이전 동작)")
    ap.add_argument("--act-drive", type=float, default=25.0,
                    help="E114: 행동 창 구동 세기(실행 motor V +A, 반대쪽 −A, 매 스텝)")
    ap.add_argument("--act-current", type=float, default=0.0,
                    help="E115: 행동 창을 지속 전류(Ioffset ±A)로. >0 이면 --act-drive(막전위 튕기기) 대신 사용. motor Ioffset 동적화 필요(자동 설정)")
    ap.add_argument("--calib-act-current", default=None,
                    help="E115 보정: 쉼표로 전류 목록(예: 0,50,100,200,400). 자극 good=left, 실행=right/left 각 20회 행동 창 발화율 → 종료")
    ap.add_argument("--save-weights", default=None,
                    help="E112: 이식 평가 직전 학습 가중치를 npz로 저장")
    ap.add_argument("--decomp-weights", default=None,
                    help="E112: 저장된 가중치로 부분 이식 분해 평가만 하고 종료(학습 없음)")
    ap.add_argument("--decomp-mode", default="all",
                    choices=("all", "none", "kc_only", "d1_only", "kc_shuffle", "kc_uniform", "kc_cm", "neuron", "kcsets",
                             "kcrate", "kcsel", "kcselonly", "kcpop"))
    ap.add_argument("--kc-rate-file", default=None,
                    help="E138: kcrate 가 KC 별 좌/우 제시·기준선 스파이크 수를 저장하고 kcsel·kcselonly 가 읽는 npz")
    ap.add_argument("--rw-motor-silence", type=float, default=0.0,
                    help="E140: 보상 창 동안 양쪽 motor Ioffset 을 −값으로(침묵). 0 = 이전 동작. motor Ioffset 동적화 자동")
    ap.add_argument("--rw-apm-scale", type=float, default=-1.0,
                    help="E141: 보상 창(도파민 켜짐) 동안 KC→motor R-STDP 의 A_plus·A_minus 배율. 0 = 흔적 생성 동결(흔적은 감쇠만), "
                         "1 = 동적화만(회귀 검사). 음수 = 끔(이전 동작, 생성 코드도 그대로). 보상 창 루프 직후 원래 값으로 되돌린다")
    ap.add_argument("--trace-kc-class", default=None,
                    help="E139(읽기 전용): 학습 중 시행마다 선택 KC(--kc-rate-file 분류) 시냅스의 교차·같은 쪽 Δg 합과 "
                         "도파민 직전 자격흔적 합을 npz 로. 난수 소비 없음 — 학습 경로 불변(경로 검사: 가중치 정확 일치)")
    ap.add_argument("--trace-kc-motor", default=None,
                    help="E111: 매 시행 도파민 직전 KC→motor 4그룹 자격흔적 합·가중치 평균을 CSV로(읽기 전용)")
    ap.add_argument("--reward-stim", default="same", choices=("same", "none"),
                    help="E110: 보상 창 동안 자극. same=유지(E109, 반사 강화로 판명) / none=끔(최소 회로와 같음)")
    ap.add_argument("--trial-gap", type=int, default=0,
                    help="E109: 보상 창 뒤 도파민 0 + 무자극 처리 스텝 수(흔적 소거). 0=이전 동작")
    ap.add_argument("--calib-kc-motor-set", default=None, choices=("zero", "rev", "ali"),
                    help="E109 보정(수정판): 새 뇌에 KC→motor 가중치를 한 상태로 넣고 **한 번만** 평가(이력 교란 제거).")
    ap.add_argument("--calib-d1-sign", type=int, default=0,
                    help="E108 조작검증: N>0이면 D1/motor 좌우 편향별 조향 부호를 N회씩 재고 종료(학습 없음).")
    ap.add_argument("--judge", default="v", choices=("v", "exec"),
                    help="E119: 보상 판정 기준. v=행동 창 이전 조향 v(|v|<=0.02 는 오답, 이전 동작) / "
                         "exec=행동 창에서 실제 실행한 motor(_ex, v 부호) — --act-window 필요")
    ap.add_argument("--kc-bilateral-scale", type=float, default=1.0,
                    help="E121 요소 맞바꾸기: 좌우 공통 KC 입력(it_food·assoc_edible·wernicke·ppc_goal·social·assoc_bind → kc_l·kc_r) "
                         "가중치 배율. 0 = 최소 회로처럼 KC 가 좌/우 눈 입력만 받음. 시냅스 구조(연결)는 그대로 — 가중치만 바뀐다")
    ap.add_argument("--snap-all-syn", action="store_true",
                    help="E121 경로 검사(읽기 전용): 학습 전후 **모든** 시냅스 집단의 g 를 비교해 이식 목록 밖에서 변한 집단을 찾는다")
    args = ap.parse_args()
    if args.judge == "exec" and args.act_window <= 0:
        raise SystemExit("--judge exec 는 --act-window 가 필요하다(실행 행동 _ex 가 행동 창에서만 정해진다)")

    # C46: 환경·워밍업 난수 고정. 미고정이면 사전 정답률이 런마다 0%~72%로 흔들려
    # 학습 효과가 잡음에 묻힌다(C43에서 실제로 그랬다).
    # E073: 뇌 연결과 환경을 **분리**해 분산 출처를 가린다.
    #   시드 간 분산(±0.02)이 효과(-0.005)의 4배라 어떤 가설도 판정 불가였다(K12).
    _bseed = args.brain_seed if args.brain_seed is not None else args.seed
    _eseed = args.env_seed if args.env_seed is not None else args.seed

    # 1) 뇌 생성 직전: 뇌 시드로 고정 (SPARSE 연결 난수가 여기서 뽑힌다)
    random.seed(_bseed)
    np.random.seed(_bseed)

    cfg = ForagerBrainConfig()
    cfg.genn_seed = 12345 + _bseed   # GeNN 연결 시드도 뇌 시드에 종속
    if args.reflex_w is not None:
        cfg.food_approach_init_w = args.reflex_w
    cfg.kc_bilateral_scale = args.kc_bilateral_scale
    if args.d1_lateral is not None:
        cfg.d1_lateral_inhibition = args.d1_lateral
    if args.kc_gamma:
        cfg.kc_weight_gamma = True
    if args.tau_e is not None:
        cfg.rstdp_tau_e = args.tau_e
    if args.kc_rstdp:
        cfg.kc_rstdp = True
    if args.kc_d1_w is not None:
        cfg.kc_to_d1_init_w = args.kc_d1_w
    if args.kc_w_max is not None:
        cfg.kc_real_rstdp_w_max = args.kc_w_max
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
    if args.reflex_sparsity is not None:
        cfg.reflex_sparsity = args.reflex_sparsity
    if args.learn_sparsity is not None:
        cfg.learn_path_sparsity = args.learn_sparsity
    if args.cortical_eta is not None:
        cfg.cortical_rstdp_eta = args.cortical_eta
    if args.direct_inhib is not None and args.direct_inhib != 0:
        cfg.direct_inhibition = args.direct_inhib
    if args.hippo_eta is not None:
        cfg.place_to_food_memory_eta = args.hippo_eta
    if args.act_current > 0 or args.calib_act_current or args.rw_motor_silence > 0:
        cfg.motor_ioffset_dynamic = True
    if args.kc_motor:
        cfg.kc_motor_rstdp = True
        cfg.kc_motor_w_max = args.kc_motor_w_max
        cfg.kc_motor_init_w = args.kc_motor_init_w
        cfg.kc_motor_eta = args.kc_motor_eta
        cfg.kc_motor_sparsity = args.kc_motor_sparsity
    if args.rw_apm_scale >= 0:
        if not args.kc_motor:
            raise SystemExit("--rw-apm-scale 은 --kc-motor 가 필요하다")
        cfg.kc_motor_apm_dynamic = True
    brain = ForagerBrain(cfg)

    # 2) 뇌 생성 후: 환경 시드로 재고정 (먹이 배치·워밍업이 여기서 결정된다)
    random.seed(_eseed)
    np.random.seed(_eseed)
    _ecfg = ForagerConfig()
    if args.n_food is not None:
        _ecfg.n_food = args.n_food
    if args.food_ratio is not None:
        _ecfg.food_type_ratio = args.food_ratio
    if args.symmetric_env:
        _ecfg.force_lr_symmetry = True
    env = ForagerGym(_ecfg)
    obs = env.reset()
    for _ in range(20):
        a, _ = brain.process(obs)
        obs, _, d, _ = env.step((a,))
        if d:
            obs = env.reset()
    nh = env.config.n_rays // 2

    if args.calib_act_current:
        # E115 보정(학습 없음): 행동 창 지속 전류 세기별 실행/반대 motor 발화. 자극 good=left(반사 = 왼쪽 motor).
        # 목표: 실행=right(반사 반대)일 때 반대(왼쪽, 반사) motor 가 거의 침묵하는 최소 전류.
        brain.reset()
        for _ in range(30):
            brain.process(obs)
        for cur in [float(x) for x in args.calib_act_current.split(",")]:
            for ex in ("right", "left"):
                exs, ots = [], []
                for _ in range(20):
                    e_, o_ = act_window_current(brain, stim(obs, nh, "left"), ex, cur, max(1, args.act_window or 3))
                    exs.append(e_); ots.append(o_)
                    for _ in range(10):   # 시행 간격(무자극)
                        _n = stim(obs, nh, "left")
                        for _k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
                            _n[_k] = np.zeros(nh)
                        brain.process(_n)
                print("=> CALIBAC cur=%.1f ex=%s ex_rate=%.4f other_rate=%.4f sel=%.3f"
                      % (cur, ex, float(np.mean(exs)), float(np.mean(ots)),
                         float(np.mean(exs)) / max(float(np.mean(exs)) + float(np.mean(ots)), 1e-9)))
        return

    if args.decomp_weights:
        # E112: 부분 이식 분해(평가만). 이식 평가와 **같은 경로**(TE.build_from_cfg: 같은 시드 새 뇌 + 같은 워밍업)로
        # 저장된 학습 가중치의 일부만 넣거나 변형해 넣고 한 번 평가한다.
        import transplant_eval as TE
        W = dict(np.load(args.decomp_weights))
        kc = sorted(n for n in W if n.startswith("kc_") and "_to_motor_" in n)
        d1 = sorted(n for n in W if n.startswith("food_to_d1"))
        if len(kc) != 4 or not d1:
            raise SystemExit("분해: KC→motor 4개·D1 경로가 필요하다 (kc=%d d1=%d)" % (len(kc), len(d1)))
        mode = args.decomp_mode
        sub = {}
        if mode == "all":
            sub = dict(W)
        elif mode == "none":
            sub = {}
        elif mode == "kc_only":
            sub = {n: W[n] for n in kc}
        elif mode == "d1_only":
            sub = {n: W[n] for n in d1}
        elif mode == "kc_shuffle":
            # 그룹별 평균·분포 보존, 시냅스별 구조(어느 KC→어느 motor 뉴런)만 파괴
            sub = dict(W)
            _rs = np.random.RandomState(777)
            for n in kc:
                sub[n] = W[n][_rs.permutation(W[n].size)]
        elif mode == "kc_uniform":
            # 그룹 평균만 남김(공통 상승 + 매핑 차이 D 보존, 그룹 안 구조 제거)
            sub = dict(W)
            for n in kc:
                # float32 로 표현 가능한 값으로(뇌 가중치는 float32 — float64 평균은 이식 검증(정확 일치)에서 실패한다, E112)
                sub[n] = np.full(W[n].size, float(np.float32(W[n].mean())))
        elif mode == "kc_cm":
            # 네 그룹 전체 평균(공통 상승만 보존, D·구조 제거)
            sub = dict(W)
            # 풀링 평균(사전등록 "전체 평균"), float32 표현 가능 값
            gm = float(np.float32(np.concatenate([W[n] for n in kc]).mean()))
            for n in kc:
                sub[n] = np.full(W[n].size, gm)
        elif mode in ("neuron", "kcsets"):
            sub = dict(W)
        elif mode in ("kcrate", "kcsel", "kcselonly", "kcpop"):
            sub = {}    # E138: 아래에서 새 뇌의 장치 연결(전시냅스 KC 인덱스)로 만든다 — KC→motor 4집단만, D1 등은 초기값
        else:
            raise SystemExit("알 수 없는 분해 모드 %s" % mode)
        _b2, _env2, _obs2 = TE.build_from_cfg(cfg, _bseed, env_seed=_eseed, env_cfg=_ecfg)
        if mode == "kcrate":
            # E138: 발화 **수** 기준 KC 선택성(kcsets 의 ≥1 스파이크 기준은 지속·잔여 발화로 "공유"를 부풀릴 수 있다).
            # 새 뇌(학습 가중치 이식 전 — KC 입력은 감각에서 오므로 반응은 KC→motor 가중치와 무관)에 kcsets 와 같은 제시:
            # good=왼쪽/오른쪽 교대 args.trials 회(각 3처리 스텝), 사이 무자극 10처리 스텝 중 **뒤 5스텝**을 기준선으로 센다.
            # 정의·분류·여유 분해는 kc_selectivity.py(합성 정답 시험 scripts/test_kc_selectivity.py) — 판정 기준 logs/E138/criteria_fixed.txt.
            import kc_selectivity as KS
            if not args.kc_rate_file:
                raise SystemExit("kcrate 는 --kc-rate-file 이 필요하다")
            n_k = int(cfg.n_kc_per_side)
            cnt = {sd: {"l": np.zeros(n_k), "r": np.zeros(n_k)} for sd in ("left", "right")}
            c0 = {"l": np.zeros(n_k), "r": np.zeros(n_k)}
            n_pres = {"left": 0, "right": 0}
            _b2.reset()
            for _ in range(30):
                _b2.process(_obs2)
            neu = stim(_obs2, nh, "left")
            for _k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
                neu[_k] = np.zeros(nh)

            def _count(dst):
                for kn, pop in (("l", _b2.kc_left), ("r", _b2.kc_right)):
                    ids = np.asarray(pop.spike_recording_data[0][1], dtype=np.int64)
                    if ids.size:
                        dst[kn] += np.bincount(ids, minlength=n_k)[:n_k]
            for rep_i in range(args.trials):
                sd = "left" if rep_i % 2 == 0 else "right"
                n_pres[sd] += 1
                for _ in range(3):
                    _b2.process(stim(_obs2, nh, sd))
                    _count(cnt[sd])
                for j in range(10):
                    _b2.process(neu)
                    if j >= 5:
                        _count(c0)
            if n_pres["left"] != n_pres["right"]:
                raise SystemExit("kcrate: 좌우 제시 수가 다르다(--trials 짝수)")
            base_steps = 5 * args.trials
            tot_sp = sum(float(cnt[sd][kn].sum()) for sd in cnt for kn in "lr")
            if tot_sp == 0:
                raise RuntimeError("KC 스파이크 0 — 측정 도구 실패(기록 버퍼 확인)")
            init_w = float(cfg.kc_motor_init_w)
            for kn in ("l", "r"):
                rL, rR, b, SI, cls = KS.classify(cnt["left"][kn], cnt["right"][kn], c0[kn], n_pres["left"], 3, base_steps)
                D = KS.dilution(cnt["left"][kn], cnt["right"][kn], cls)
                g1 = KS.sets_ge1(cnt["left"][kn], cnt["right"][kn])
                dS = {}
                for m in ("l", "r"):
                    nm = "kc_%s_to_motor_%s" % (kn, m)
                    sy = getattr(_b2, nm); sy.pull_connectivity_from_device()
                    pre = np.asarray(sy.get_sparse_pre_inds(), dtype=np.int64)
                    if pre.size != W[nm].size:
                        raise RuntimeError("%s 크기 불일치 %d vs %d" % (nm, pre.size, W[nm].size))
                    dS[m] = KS.per_kc_sum(pre, W[nm] - init_w, n_k)
                mk = KS.margin(rL, rR, dS["r"], dS["l"])
                M = float(mk.sum())
                sh = [float(mk[cls == c].sum() / M) if M != 0 else float("nan") for c in (KS.CLS_L, KS.CLS_R, KS.CLS_NS)]
                ncls = [int((cls == c).sum()) for c in (KS.CLS_L, KS.CLS_R, KS.CLS_NS, KS.CLS_NONE)]
                nsel = {th: int(np.isin(KS.classify(cnt["left"][kn], cnt["right"][kn], c0[kn], n_pres["left"], 3, base_steps, theta=th)[4],
                                        (KS.CLS_L, KS.CLS_R)).sum()) for th in (0.3, 0.7)}
                mds = " ".join("%s(→L %+.1f →R %+.1f)" % (lab, float(dS["l"][cls == c].mean()) if (cls == c).any() else float("nan"),
                                                       float(dS["r"][cls == c].mean()) if (cls == c).any() else float("nan"))
                               for lab, c in (("좌선택", KS.CLS_L), ("우선택", KS.CLS_R), ("비선택", KS.CLS_NS)))
                print("=> KCRATE kc_%s | 좌선택 %d 우선택 %d 비선택 %d 무활동 %d | 희석 %.3f | ≥1스파이크 좌전용 %d 우전용 %d 공유 %d 무반응 %d"
                      " | 여유합 %+.1f 몫 좌선택 %.3f 우선택 %.3f 비선택 %.3f | θ0.3 선택 %d θ0.7 선택 %d | 제시 스파이크 %d 기준선(제시창) 평균 %.4f | KC별 ΔS %s"
                      % (kn, ncls[0], ncls[1], ncls[2], ncls[3], D, g1[0], g1[1], g1[2], g1[3], M, sh[0], sh[1], sh[2],
                         nsel[0.3], nsel[0.7], int(cnt["left"][kn].sum() + cnt["right"][kn].sum()), float(b.mean()), mds))
            np.savez_compressed(args.kc_rate_file, cL_l=cnt["left"]["l"], cR_l=cnt["right"]["l"], c0_l=c0["l"],
                                cL_r=cnt["left"]["r"], cR_r=cnt["right"]["r"], c0_r=c0["r"],
                                n_pres=np.int64(n_pres["left"]), base_steps=np.int64(base_steps))
            print("[E138] KC 발화 수 저장 → %s (좌우 각 %d회, 기준선 %d스텝)" % (args.kc_rate_file, n_pres["left"], base_steps))
            return
        if mode in ("kcsel", "kcselonly", "kcpop"):
            # E138: KC→motor 4집단만 바꿔 이식(D1 등 나머지 학습 경로는 초기값 — kc_only 기준과 같은 범위).
            # kcpop = 모든 KC 집단 교차(E119 P2 rev 와 같은 값) / kcsel = 선택 KC 만 선호 쪽 교차, 나머지 초기 / kcselonly = 선택 KC 만 학습값, 나머지 초기.
            import kc_selectivity as KS
            n_k = int(cfg.n_kc_per_side); init_w = float(cfg.kc_motor_init_w); wmax = float(cfg.kc_motor_w_max)
            cls = {}
            if mode != "kcpop":
                if not args.kc_rate_file or not os.path.exists(args.kc_rate_file):
                    raise SystemExit("%s 는 kcrate 가 저장한 --kc-rate-file 이 필요하다" % mode)
                Z = np.load(args.kc_rate_file)
                for kn in ("l", "r"):
                    cls[kn] = KS.classify(Z["cL_" + kn], Z["cR_" + kn], Z["c0_" + kn], int(Z["n_pres"]), 3, int(Z["base_steps"]))[4]
            for kn in ("l", "r"):
                for m in ("l", "r"):
                    nm = "kc_%s_to_motor_%s" % (kn, m)
                    sy = getattr(_b2, nm); sy.pull_connectivity_from_device()
                    pre = np.asarray(sy.get_sparse_pre_inds(), dtype=np.int64)
                    if pre.size != W[nm].size:
                        raise RuntimeError("%s 크기 불일치 %d vs %d" % (nm, pre.size, W[nm].size))
                    if mode == "kcpop":
                        sub[nm] = KS.ideal_weights(pre, None, kn, m, wmax, init_w, "pop")
                    elif mode == "kcsel":
                        sub[nm] = KS.ideal_weights(pre, cls[kn], kn, m, wmax, init_w, "sel")
                    else:
                        sub[nm] = KS.selonly_weights(pre, cls[kn], W[nm], init_w)
            if cls:
                print("[E138] %s: 선택 KC kc_l 좌 %d 우 %d / kc_r 좌 %d 우 %d (wmax %.0f, init %.0f)"
                      % (mode, int((cls["l"] == KS.CLS_L).sum()), int((cls["l"] == KS.CLS_R).sum()),
                         int((cls["r"] == KS.CLS_L).sum()), int((cls["r"] == KS.CLS_R).sum()), wmax, init_w))
        if mode == "kcsets":
            # E117: KC 반응 집합(자극 good=왼쪽/오른쪽) — 새 뇌(학습 가중치 이식 전)에 steer 와 같은 3스텝 제시 × N, 사이 무자극 10스텝.
            # KC 입력은 감각에서 오므로 반응 집합은 KC→motor 가중치와 무관하다. 집합별로 저장된 학습 Δg 를 분해한다.
            n_k = int(cfg.n_kc_per_side)
            cnt = {sd: {"l": np.zeros(n_k), "r": np.zeros(n_k)} for sd in ("left", "right")}
            _b2.reset()
            for _ in range(30):
                _b2.process(_obs2)
            neu = stim(_obs2, nh, "left")
            for _k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
                neu[_k] = np.zeros(nh)
            for rep_i in range(args.trials):
                sd = "left" if rep_i % 2 == 0 else "right"
                for _ in range(3):
                    _b2.process(stim(_obs2, nh, sd))
                    for kn, pop in (("l", _b2.kc_left), ("r", _b2.kc_right)):
                        ids = np.asarray(pop.spike_recording_data[0][1], dtype=np.int64)
                        if ids.size:
                            cnt[sd][kn] += np.bincount(ids, minlength=n_k)[:n_k]
                for _ in range(10):
                    _b2.process(neu)
            tot_sp = sum(float(cnt[sd][kn].sum()) for sd in cnt for kn in "lr")
            if tot_sp == 0:
                raise RuntimeError("KC 스파이크 0 — 측정 도구 실패(기록 버퍼 확인)")
            init_w = float(cfg.kc_motor_init_w)
            for kn in ("l", "r"):
                aL = cnt["left"][kn] > 0; aR = cnt["right"][kn] > 0
                cls = {"좌전용": aL & ~aR, "우전용": aR & ~aL, "공유": aL & aR, "무반응": ~aL & ~aR}
                pm = {}
                for m in ("l", "r"):
                    nm = "kc_%s_to_motor_%s" % (kn, m)
                    sy = getattr(_b2, nm); sy.pull_connectivity_from_device()
                    pre = np.asarray(sy.get_sparse_pre_inds(), dtype=np.int64)
                    dg = W[nm] - init_w
                    pm[m] = np.bincount(pre, weights=dg, minlength=n_k) / np.maximum(np.bincount(pre, minlength=n_k), 1)
                out = []
                for cn, msk in cls.items():
                    n = int(msk.sum())
                    out.append("%s n=%d →L %+.2f →R %+.2f" % (cn, n, float(pm["l"][msk].mean()) if n else float("nan"),
                                                               float(pm["r"][msk].mean()) if n else float("nan")))
                print("=> KCSETS kc_%s | %s" % (kn, " | ".join(out)))
            return
        if mode == "neuron":
            # E113: 뉴런 수준 신용 누수 — motor 뉴런별 (학습된 KC 입력 Δg 평균) 대 (반사 입력 연결 수).
            # 연결은 같은 시드로 만든 새 뇌에서 읽는다(학습 뇌와 동일 — 이식 검증이 크기·순서를 보장).
            def post_inds(syn):
                syn.pull_connectivity_from_device()
                a = np.asarray(syn.get_sparse_post_inds(), dtype=np.int64)
                if a.size == 0:
                    raise RuntimeError("연결을 읽지 못했다 — 측정 도구 실패")
                return a
            init_w = float(cfg.kc_motor_init_w)
            n_m = {"l": int(cfg.n_motor_left), "r": int(cfg.n_motor_right)}
            refl = {}
            for side in ("l", "r"):
                deg = np.zeros(n_m[side])
                for nm in ("good_food_to_motor_%s" % side, "food_explore_motor_%s" % side):
                    sy = getattr(_b2, nm, None)
                    if sy is None:
                        raise SystemExit("반사 경로 %s 없음" % nm)
                    deg += np.bincount(post_inds(sy), minlength=n_m[side])[:n_m[side]]
                refl[side] = deg
            for k in ("l", "r"):
                for m in ("l", "r"):
                    nm = "kc_%s_to_motor_%s" % (k, m)
                    post = post_inds(getattr(_b2, nm))
                    dg = W[nm] - init_w
                    if dg.size != post.size:
                        raise RuntimeError("%s 크기 불일치 %d vs %d" % (nm, dg.size, post.size))
                    s_ = np.bincount(post, weights=dg, minlength=n_m[m]); c_ = np.bincount(post, minlength=n_m[m])
                    mdg = s_ / np.maximum(c_, 1)
                    d = refl[m]
                    r = float(np.corrcoef(d, mdg)[0, 1]) if d.std() > 0 and mdg.std() > 0 else float("nan")
                    q = np.quantile(d, [0.25, 0.75])
                    lo, hi = mdg[d <= q[0]].mean(), mdg[d >= q[1]].mean()
                    print("=> NEURON group=%s%s r=%+.4f dg_hiRefl=%+.4f dg_loRefl=%+.4f dg_all=%+.4f refl_mean=%.1f"
                          % (k, m, r, hi, lo, float(dg.mean()), float(d.mean())))
            # KC(pre) 쪽: KC 는 한 번에 ~6%만 발화 → 그룹 평균은 비활성 KC 시냅스로 희석된다.
            # KC 별로 같은 쪽 motor(반사: kc_l→motor_l, kc_r→motor_r)와 교차 motor 로 가는 평균 Δg 를 구하고,
            # 가장 많이 변한 KC(활성 KC 근사: 두 방향 평균 Δg 상위 5%)에서 반사 대 교차를 비교한다.
            n_k = int(cfg.n_kc_per_side)
            for k in ("l", "r"):
                same, cross = k, ("r" if k == "l" else "l")
                pm = {}
                for m in (same, cross):
                    nm = "kc_%s_to_motor_%s" % (k, m)
                    sy = getattr(_b2, nm); sy.pull_connectivity_from_device()
                    pre = np.asarray(sy.get_sparse_pre_inds(), dtype=np.int64)
                    dg = W[nm] - init_w
                    pm[m] = np.bincount(pre, weights=dg, minlength=n_k) / np.maximum(np.bincount(pre, minlength=n_k), 1)
                tot = (pm[same] + pm[cross]) / 2
                top = tot >= np.quantile(tot, 0.95)
                bot = tot <= np.quantile(tot, 0.50)
                print("=> KCPRE side=%s top5%%: 반사(%s%s) %+.4f 교차(%s%s) %+.4f 차 %+.4f | 하위50%%: 반사 %+.4f 교차 %+.4f | r(반사,교차)=%+.3f"
                      % (k, k, same, float(pm[same][top].mean()), k, cross, float(pm[cross][top].mean()),
                         float(pm[same][top].mean() - pm[cross][top].mean()),
                         float(pm[same][bot].mean()), float(pm[cross][bot].mean()),
                         float(np.corrcoef(pm[same], pm[cross])[0, 1])))
            return
        if sub:
            TE.push(_b2, sub)
            TE.verify(_b2, _b2, sub)
        acc, off, mod = evaluate(_b2, _obs2, nh, args.trials)
        gms = " ".join("%s=%.4f" % (n.replace("_to_motor_", ">"), float(sub[n].mean()) if n in sub else float("nan")) for n in kc)
        print("=> DECOMP mode=%s mod=%+.4f acc=%.1f off=%+.4f pushed=%d kc_means[%s]" % (mode, mod, acc, off, len(sub), gms))
        return

    if args.calib_kc_motor_set:
        # E109 보정 수정판: 한 뇌를 연달아 평가하면 동역학 이력이 조건을 교란한다(K22 — 첫 판에서 init≠zero 로 드러남).
        # **조건마다 같은 시드로 새로 만든 뇌를 한 번만 평가**한다(INV-B5 와 같은 원리: 이력 동일 → 가중치만 다름).
        syn = getattr(brain, "kc_motor_syn", None)
        if not syn:
            raise SystemExit("--calib-kc-motor-set 은 --kc-motor 가 필요하다")
        wm = args.kc_motor_w_max
        same, cross = {"zero": (0.0, 0.0), "rev": (0.0, wm), "ali": (wm, 0.0)}[args.calib_kc_motor_set]
        for (k, m), s_ in syn.items():
            s_.pull_connectivity_from_device()
            s_.vars["g"].pull_from_device()
            _n = np.asarray(s_.vars["g"].values).size
            s_.vars["g"].values = np.full(_n, (same if k == m else cross), dtype=np.float32)
            s_.vars["g"].push_to_device()
        acc, off, mod = evaluate(brain, obs, nh, args.trials)
        print("=> CALIBKM1 set=%s wmax=%.2f sp=%.3f mod=%+.4f acc=%.1f off=%+.4f"
              % (args.calib_kc_motor_set, wm, args.kc_motor_sparsity, mod, acc, off))
        return

    if args.calib_kc_motor:
        # E109 조작검증: KC→motor 경로의 **행동 권한**. 학습 없이 가중치를 극단으로 넣고 변조폭을 잰다.
        # 변조폭 = mean(조향|good=우) − mean(조향|good=좌). 반사면 양수, 역전이면 음수.
        syn = getattr(brain, "kc_motor_syn", None)
        if not syn:
            raise SystemExit("--calib-kc-motor 는 --kc-motor 가 필요하다")

        def set_w(same, cross):
            for (k, m), s_ in syn.items():
                s_.pull_connectivity_from_device()
                s_.vars["g"].pull_from_device()
                # SPARSE 는 view 로 쓸 수 없다(PyGeNN: values 사용) — transplant_eval 과 같은 규칙
                _n = np.asarray(s_.vars["g"].values).size
                s_.vars["g"].values = np.full(_n, (same if k == m else cross), dtype=np.float32)
                s_.vars["g"].push_to_device()

        out = {}
        _, _, out["init"] = evaluate(brain, obs, nh, args.trials)
        wm = args.kc_motor_w_max
        set_w(0.0, wm); _, _, out["rev"] = evaluate(brain, obs, nh, args.trials)      # 완전 역전 학습 상태
        set_w(wm, 0.0); _, _, out["ali"] = evaluate(brain, obs, nh, args.trials)      # 반사 정렬 학습 상태
        set_w(0.0, 0.0); _, _, out["zero"] = evaluate(brain, obs, nh, args.trials)    # 경로 없음
        print("[보정 KC→motor] w_max=%.2f 변조폭: 초기 %+.4f | 역전(교차=w_max) %+.4f | 정렬 %+.4f | 0 %+.4f"
              % (wm, out["init"], out["rev"], out["ali"], out["zero"]))
        print("=> CALIBKM wmax=%.2f init=%+.4f rev=%+.4f ali=%+.4f zero=%+.4f"
              % (wm, out["init"], out["rev"], out["ali"], out["zero"]))
        return

    if args.calib_d1_sign > 0:
        # E108 조작검증: D1 좌/우 편향이 조향(angle_delta 합) 부호를 어느 쪽으로 바꾸는가.
        # angle_delta 주석(>0=CCW=왼쪽)과 과제의 정답 판정(good 왼쪽이면 v>0이 정답=반대쪽)이 엇갈려 보여
        # 행동 흔적 구동(--act-stamp)의 방향을 코드 해석이 아니라 **실측**으로 정한다. 학습 없음(도파민 0).
        brain.reset()
        for _ in range(30):
            brain.process(obs)
        off = measure_offset(brain, obs, nh)
        neut = stim(obs, nh, "left")
        for k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
            neut[k] = np.ones(nh) * 0.45
        res = {}
        for where in ("d1", "motor"):
            for side in ("left", "right"):
                vs = [steer(brain, neut, steps=3, bias_side=side, bias_strength=args.bias,
                            bias_at_d1=(where == "d1")) - off for _ in range(args.calib_d1_sign)]
                res[(where, side)] = (float(np.mean(vs)), float(np.std(vs)))
                print("[보정] %-5s 편향=%-5s → 조향 평균 %+.4f (표준편차 %.4f, n=%d)"
                      % (where, side, res[(where, side)][0], res[(where, side)][1], args.calib_d1_sign))
        # 반사 방향 참고: good 이 왼쪽일 때의 조향 부호
        vl = float(np.mean([steer(brain, stim(obs, nh, "left"), steps=3) - off for _ in range(args.calib_d1_sign)]))
        print("[보정] 반사 참고: good=left 조향 평균 %+.4f (반사=good 쪽 접근)" % vl)
        print("=> CALIB d1_left=%+.4f d1_right=%+.4f motor_left=%+.4f motor_right=%+.4f good_left=%+.4f"
              % (res[("d1", "left")][0], res[("d1", "right")][0], res[("motor", "left")][0],
                 res[("motor", "right")][0], vl))
        return

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

    def snap_reflex():
        """E119 경로 검사(읽기 전용): 반사 경로 가중치 평균. good_food→motor 는 학습 가능(food_approach)하나
        이식 목록에 없다 — 학습 중 행동에만 영향을 준다."""
        out = {}
        for nm in ("good_food_to_motor_l", "good_food_to_motor_r", "food_explore_motor_l", "food_explore_motor_r"):
            s = getattr(brain, nm, None)
            if s is None:
                continue
            s.pull_connectivity_from_device()
            s.vars["g"].pull_from_device()
            v_ = np.asarray(s.vars["g"].values, dtype=np.float64)
            if v_.size == 0:
                raise RuntimeError("반사 가중치 %s: 빈 배열 — 측정 도구 실패" % nm)
            out[nm] = (float(v_.mean()), int(v_.size))
        return out

    reflex_before = snap_reflex()

    def snap_all_syn():
        """E121 경로 검사(읽기 전용): 모든 시냅스 집단의 g. g 가 변수가 아닌(상수) 집단은 건너뛴다."""
        out = {}
        for nm, sg in brain.model.synapse_populations.items():
            if "g" not in sg.vars:
                continue
            try:
                sg.pull_connectivity_from_device()
            except Exception:
                pass          # DENSE 등 연결 당기기가 없는 집단
            sg.vars["g"].pull_from_device()
            out[nm] = np.array(sg.vars["g"].values, dtype=np.float64).ravel()
        if not out:
            raise RuntimeError("전체 시냅스 스냅숏: 집단 0개 — 측정 도구 실패")
        return out
    syn_before = snap_all_syn() if args.snap_all_syn else None
    # E121 경로 검사(읽기 전용): 좌우 공통 KC 입력 가중치가 설정 배율대로 시냅스에 들어갔는가
    _kb = []
    for _src in ("it_food", "assoc_edible", "wernicke_food", "ppc_goal_food", "social_mem", "assoc_bind"):
        for _sd in ("l", "r"):
            _pn = "%s_to_kc_%s" % (_src, _sd)
            _sg = brain.model.synapse_populations.get(_pn)
            if _sg is None:
                continue
            _sg.pull_connectivity_from_device()
            _sg.vars["g"].pull_from_device()
            _gv = np.asarray(_sg.vars["g"].values, dtype=np.float64)
            if _gv.size == 0:
                raise RuntimeError("KC 공통 입력 %s: 빈 배열 — 측정 도구 실패" % _pn)
            _kb.append("%s n=%d w=%.3f" % (_pn, _gv.size, _gv.mean()))
    print("[KC공통입력] scale=%.2f 집단 %d개: %s" % (args.kc_bilateral_scale, len(_kb), "; ".join(_kb)))
    rew = 0
    explored = 0
    eps = args.epsilon
    TRACE_ROWS = []
    # E139 KC 계층 추적(읽기 전용): KC→motor 시냅스마다 역할 0 교차·1 같은 쪽(선택 KC, 선호 기준)·2 비선택·3 무활동.
    # 좌선택 KC 의 교차 = motor_r, 우선택 KC 의 교차 = motor_l (반사 0 정답 = good 반대쪽, K57). 분류 = kc_selectivity(E138 과 같음).
    KCT = None
    if args.trace_kc_class:
        import kc_selectivity as KS
        _ksyn = getattr(brain, "kc_motor_syn", None)
        if not _ksyn or not args.kc_rate_file:
            raise SystemExit("--trace-kc-class 는 --kc-motor 와 --kc-rate-file 이 필요하다")
        _Zc = np.load(args.kc_rate_file)
        _kcls = {kn: KS.classify(_Zc["cL_" + kn], _Zc["cR_" + kn], _Zc["c0_" + kn], int(_Zc["n_pres"]), 3, int(_Zc["base_steps"]))[4]
                 for kn in ("l", "r")}
        KCT = {"role": {}, "rows": [], "g_prev": None, "gap": 0.0}
        for (_k, _m), _s in _ksyn.items():
            _s.pull_connectivity_from_device()
            _pre = np.asarray(_s.get_sparse_pre_inds(), dtype=np.int64)
            if _pre.size == 0:
                raise RuntimeError("KC 계층 추적: %s%s 연결 0 — 측정 도구 실패" % (_k, _m))
            _c = _kcls[_k][_pre]
            _role = np.full(_pre.size, 2, dtype=np.int64)
            if _m == "r":
                _role[_c == KS.CLS_L] = 0; _role[_c == KS.CLS_R] = 1
            else:
                _role[_c == KS.CLS_R] = 0; _role[_c == KS.CLS_L] = 1
            _role[_c == KS.CLS_NONE] = 3
            KCT["role"][(_k, _m)] = _role

        def _kct_read(var):
            out = {}
            for (_k, _m), _s in _ksyn.items():
                _s.vars[var].pull_from_device()
                _a = np.asarray(_s.vars[var].values, dtype=np.float64).ravel()
                if _a.size != KCT["role"][(_k, _m)].size:
                    raise RuntimeError("KC 계층 추적: %s%s %s 크기 %d ≠ %d — 측정 도구 실패" % (_k, _m, var, _a.size, KCT["role"][(_k, _m)].size))
                out[(_k, _m)] = _a.copy()
            return out

        def _kct_sum(arrs):
            r_ = np.zeros(4)
            for key_, a_ in arrs.items():
                r_ += np.bincount(KCT["role"][key_], weights=a_, minlength=4)[:4]
            return r_
        print("[E139] KC 계층 추적: 시냅스 역할 교차 %d 같은쪽 %d 비선택 %d 무활동 %d (분류 %s)"
              % (tuple(sum(int((r_ == i).sum()) for r_ in KCT["role"].values()) for i in range(4)) + (args.kc_rate_file,)))
    # E119 판정 경로 계수(읽기 전용): 행동 창이 있을 때 v 판정과 실행 행동 판정을 시행마다 비교한다.
    J = {"n": 0, "small": 0, "v_ok_ex_no": 0, "v_no_ex_ok": 0}
    for ep in range(args.episodes):
        off = measure_offset(brain, obs, nh, n=5)
        for t in range(args.steps):
            if KCT is not None:
                # E139: 시행 시작 g(난수 소비 없음). 직전 시행 끝 g 와 같아야 한다(도파민 0 구간엔 g 불변) — 연속성 검사.
                _kg0 = _kct_read("g")
                if KCT["g_prev"] is not None:
                    KCT["gap"] = max(KCT["gap"], max(float(np.abs(_kg0[k_] - KCT["g_prev"][k_]).max()) for k_ in _kg0))
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
            if args.act_window > 0:
                # K52: v<0 = motor_left 우세. v==0 이면 무작위. (E119: 판정 비교를 위해 행동 창 앞으로 옮김 —
                # 사이에 np.random 소비가 없어 난수열은 이전과 같다)
                _ex = "left" if v < 0 else ("right" if v > 0 else ("left" if np.random.random() < 0.5 else "right"))
                correct_ex = (side == "left" and _ex == "right") or (side == "right" and _ex == "left")
                J["n"] += 1
                J["small"] += int(abs(v) <= 0.02)
                J["v_ok_ex_no"] += int(correct and not correct_ex)
                J["v_no_ex_ok"] += int(correct_ex and not correct)
                if args.judge == "exec":
                    correct = correct_ex
            if args.trace_kc_motor and getattr(brain, "kc_motor_syn", None):
                # E111: 도파민 직전 KC→motor 자격흔적·가중치 (4그룹 합/평균). 읽기 전용.
                _row = [ep, t, side, int(do_explore), round(float(v), 4), int(correct)]
                for (_k, _m) in (("l", "l"), ("l", "r"), ("r", "l"), ("r", "r")):
                    _s = brain.kc_motor_syn[(_k, _m)]
                    # SPARSE 는 연결을 먼저 당겨야 values 가 채워진다(E111 1차: 빠뜨려서 전부 빈 배열 → 합 0·평균 nan).
                    _s.pull_connectivity_from_device()
                    _s.vars["e"].pull_from_device(); _s.vars["g"].pull_from_device()
                    _e = np.asarray(_s.vars["e"].values, dtype=np.float64); _g = np.asarray(_s.vars["g"].values, dtype=np.float64)
                    if _e.size == 0 or _g.size == 0:
                        raise RuntimeError("KC→motor 추적: 빈 배열(%s%s) — 측정 도구 실패" % (_k, _m))
                    _row += [round(float(_e.sum()), 4), round(float(_g.mean()), 5)]
                TRACE_ROWS.append(_row)
            if args.act_window > 0:
                # E114: 행동 창(최소 회로 act_drive + WTA 이식). 전체 모델은 반사 경로 때문에 두 motor 가 함께 발화해
                # 자격흔적이 양쪽에 생기고 보상이 비선택적 공통 강화로만 작동했다(E112·E113). 행동이 정해진 뒤
                # 자극을 유지한 채 **실행한 motor 만 구동하고 반대쪽은 억제**해, 흔적이 실행 행동을 담게 한다.
                # _ex 는 위(판정 직후)에서 정했다.
                if args.act_current > 0:
                    # E115: 지속 전류(Ioffset) — 창 전체 동안 실행 motor +I, 반대 −I. 최소 회로 act_drive 와 같은 방식.
                    act_window_current(brain, stim(obs, nh, side), _ex, args.act_current, args.act_window)
                else:
                    steer(brain, stim(obs, nh, side), steps=args.act_window, bias_side=_ex, bias_strength=args.act_drive)
            if KCT is not None:
                _ke = _kct_sum(_kct_read("e"))      # E139: 도파민 직전(행동 창 뒤) 자격흔적 역할별 합
                _kgda = _kct_read("g")               # E139 수정: 도파민 직전 g — 도파민 전 변화 분리(V4)
                _kee = np.zeros(4); _mrw = [0.0, 0.0]   # 보상 창 끝 흔적·보상 창 중 motor 발화율 합(보상 창이 있으면 아래에서 채움)
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
            if args.reward_window > 0 or args.trial_gap > 0:
                # E109: 보상 타이밍 수리. 이 과제는 decay_dopamine()을 부르지 않아 도파민이 **다음 시행**의
                # 처리 스텝 동안 가중치에 반영됐다(시행 t 보상 → 시행 t+1 활동에 배정). 최소 회로처럼
                # (1) 같은 자극을 유지한 채 보상 창 K스텝 → (2) 도파민 0 → (3) 무자극 간격 G스텝.
                _o = stim(obs, nh, side)
                if args.reward_stim == "none":
                    # E110: 보상 구간에 **자극을 끈다**(최소 회로 apply_stim None 과 같음). 자극을 유지하면
                    # 뇌 자신의 반사 반응이 도파민과 겹쳐 반사 연합이 강화된다(E109 첫 런 +0.25).
                    for _k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
                        _o[_k] = np.zeros(nh)
                if args.rw_motor_silence > 0:
                    # E140: 보상 창(도파민 켜짐) 동안 양쪽 motor 침묵(지속 음 전류) — 새 post 스파이크가 없으면 보상 창의 비선택 LTP 흔적이 생기지 않는다.
                    # 행동 창·평가에는 걸지 않는다(보상 창 루프 직후 0 으로 되돌림).
                    brain.motor_left.set_dynamic_param_value("Ioffset", -args.rw_motor_silence)
                    brain.motor_right.set_dynamic_param_value("Ioffset", -args.rw_motor_silence)
                if args.rw_apm_scale >= 0:
                    # E141: 보상 창 동안 KC→motor 흔적 생성 배율(0 = 동결 — 새 pre·post 스파이크가 흔적을 만들지 않고,
                    # 결정·행동 창 흔적이 감쇠만 하며 가중치로 굳는다). E140 침묵은 생성 부호만 LTD 로 바꿨다(경로 검사).
                    for _s in brain.kc_motor_syn.values():
                        _s.set_dynamic_param_value("A_plus", args.rw_apm_scale * brain.kc_motor_apm[0])
                        _s.set_dynamic_param_value("A_minus", args.rw_apm_scale * brain.kc_motor_apm[1])
                for _ in range(args.reward_window):
                    _a_rw, _inf_rw = brain.process(_o)
                    if KCT is not None and isinstance(_inf_rw, dict):
                        # E139 수정(읽기 전용): 보상 창 중 좌/우 motor 발화율 — 보상 창에 양쪽 motor 가 발화하면 비선택 흔적이 생긴다
                        _mrw[0] += float(_inf_rw.get("motor_left_rate", 0.0)); _mrw[1] += float(_inf_rw.get("motor_right_rate", 0.0))
                if args.rw_motor_silence > 0:
                    brain.motor_left.set_dynamic_param_value("Ioffset", 0.0)
                    brain.motor_right.set_dynamic_param_value("Ioffset", 0.0)
                if args.rw_apm_scale >= 0:
                    for _s in brain.kc_motor_syn.values():
                        _s.set_dynamic_param_value("A_plus", brain.kc_motor_apm[0])
                        _s.set_dynamic_param_value("A_minus", brain.kc_motor_apm[1])
                if KCT is not None:
                    _kee = _kct_sum(_kct_read("e"))      # E139 수정: 보상 창 끝(도파민 0 직전) 자격흔적
                brain.dopamine_level = 0.0
                brain._push_dopamine_to_rstdp()
                _neu = stim(obs, nh, "left")
                for _k in ("good_food_rays_left", "good_food_rays_right", "food_rays_left", "food_rays_right"):
                    _neu[_k] = np.zeros(nh)
                for _ in range(args.trial_gap):
                    brain.process(_neu)
            if KCT is not None:
                # E139: 시행 끝(보상 창 → 도파민 0 → 간격 뒤) g. Δg = 이 시행의 도파민이 만든 변화.
                _kg1 = _kct_read("g")
                _dg = {k_: _kg1[k_] - _kg0[k_] for k_ in _kg1}
                _rs = _kct_sum(_dg)
                _tot = float(sum(float(d_.sum()) for d_ in _dg.values()))
                _rpre = _kct_sum({k_: _kgda[k_] - _kg0[k_] for k_ in _kgda})
                KCT["rows"].append([ep, t, int(side == "right"), int(do_explore), (int(probe_side == "right") if do_explore else -1),
                                    float(v), (int(_ex == "right") if args.act_window > 0 else -1), int(correct)]
                                   + [float(x) for x in _rs] + [_tot] + [float(x) for x in _ke]
                                   + [float(x) for x in _rpre] + [float(x) for x in _kee] + [float(_mrw[0]), float(_mrw[1])])
                KCT["g_prev"] = _kg1
    print("[학습] %dep 완료, 보상 %d회 (탐색 주입 %d회, ε=%.2f)" % (args.episodes, rew, explored, eps))
    if J["n"]:
        print("[판정경로] judge=%s 시행=%d abs_v_le_0.02=%d (%.1f%%) v정답_실행오답=%d v오답_실행정답=%d 불일치율=%.1f%%"
              % (args.judge, J["n"], J["small"], 100.0 * J["small"] / J["n"], J["v_ok_ex_no"], J["v_no_ex_ok"],
                 100.0 * (J["v_ok_ex_no"] + J["v_no_ex_ok"]) / J["n"]))
    if KCT is not None and KCT["rows"]:
        # E139 요약(정의: logs/E139/criteria_fixed.txt — 수정 기준). 열: 0 ep 1 t 2 side_r 3 explore 4 probe_r 5 v 6 ex_r 7 correct
        # 8 dg_교차 9 dg_같은쪽 10 dg_비선택 11 dg_무활동 12 dg_전체 13~16 e(도파민 직전, 같은 순서)
        # 17~20 dg_도파민전(g 도파민 직전 − g 시작) 21~24 e(보상 창 끝) 25 보상창 motor_left 발화율 합 26 motor_right
        _R = np.array(KCT["rows"], dtype=np.float64)
        np.savez_compressed(args.trace_kc_class, rows=_R)
        _rw = _R[:, 7] == 1

        def _abcp(msk):
            return (_R[msk & _rw, 8].sum(), _R[msk & _rw, 9].sum(), _R[msk & ~_rw, 8].sum(), _R[msk & ~_rw, 9].sum())
        _all = np.ones(len(_R), dtype=bool)
        _A, _B, _C, _P = _abcp(_all)
        _Bp = max(_B, 0.0); _Cm = -min(_C, 0.0)
        _shB = _Bp / (_Bp + _Cm) if (_Bp + _Cm) > 0 else float("nan")
        _cons = float(np.max(np.abs(_R[:, 8:12].sum(1) - _R[:, 12]) / np.maximum(np.abs(_R[:, 12]), 1.0)))
        _blk = [float((_R[i * 100:(i + 1) * 100, 8] - _R[i * 100:(i + 1) * 100, 9]).sum()) for i in range(int(np.ceil(len(_R) / 100)))]
        _er, _es = _R[_rw, 13].sum(), _R[_rw, 14].sum()
        print("=> KCTRACE 시행 %d 보상 %d | A %+.1f B %+.1f C %+.1f P %+.1f | ΔD %+.1f | share_B %.3f B+/A %.3f C-/A %.3f | 합일관성 최대 %.2e | g연속 최대 %.3g | 블록 ΔD %s | 보상시행 e_same/e_cross %.3f"
              % (len(_R), int(_rw.sum()), _A, _B, _C, _P, (_A + _C) - (_B + _P), _shB, (_Bp / _A) if _A else float("nan"),
                 (_Cm / _A) if _A else float("nan"), _cons, KCT["gap"], " ".join("%+.0f" % x for x in _blk), (_es / _er) if _er else float("nan")))
        _ex_ = _R[:, 3] == 1
        print("=> KCTRACE2 탐색 A %+.1f B %+.1f C %+.1f P %+.1f | 탐욕 A %+.1f B %+.1f C %+.1f P %+.1f | 비선택 Δg 보상 %+.1f 처벌 %+.1f → %s"
              % (*_abcp(_ex_), *_abcp(~_ex_), _R[_rw, 10].sum(), _R[~_rw, 10].sum(), args.trace_kc_class))
        # E139 수정 기준 요약: 단계 분리(V4), 두 도파민 부호의 공동 움직임, 흔적 부호 양상(도파민 직전 vs 보상 창 끝), 보상 창 motor 발화
        _pre_tot = float(np.abs(_R[:, 17:21].sum())); _all_tot = float(np.abs(_R[:, 12].sum()))
        _eda = _R[_rw, 14].sum() / _R[_rw, 13].sum() if _R[_rw, 13].sum() else float("nan")
        _eend = _R[_rw, 22].sum() / _R[_rw, 21].sum() if _R[_rw, 21].sum() else float("nan")
        print("=> KCTRACE3 도파민전 |Σ|/|Σ시행| %.2e | B/A %.3f C/P %.3f | 보상시행 e_da 같은쪽/교차 %+.3f e_end 같은쪽/교차 %+.3f | 보상창 motor 발화율 평균 보상(좌 %.4f 우 %.4f) 처벌(좌 %.4f 우 %.4f)"
              % ((_pre_tot / _all_tot) if _all_tot else float("nan"), (_B / _A) if _A else float("nan"), (_C / _P) if _P else float("nan"),
                 _eda, _eend, _R[_rw, 25].mean(), _R[_rw, 26].mean(), _R[~_rw, 25].mean(), _R[~_rw, 26].mean()))
    reflex_after = snap_reflex()
    for nm in sorted(reflex_before):
        print("[반사가중치] %-22s n=%d w_mean %.4f→%.4f (학습 뇌; 이식 대상 아님)"
              % (nm, reflex_before[nm][1], reflex_before[nm][0], reflex_after[nm][0]))
    if syn_before is not None:
        import transplant_eval as _TEs
        _tp = {getattr(brain, n).name for n in _TEs.learned_names(brain)}
        syn_after = snap_all_syn()
        _chg, _out = [], []
        for nm in sorted(syn_before):
            b_, a_ = syn_before[nm], syn_after[nm]
            if b_.shape != a_.shape:
                raise RuntimeError("전체 시냅스 스냅숏 %s: 크기 불일치 %s→%s" % (nm, b_.shape, a_.shape))
            d = np.abs(a_ - b_)
            if b_.size and (d > 0).any():
                _chg.append(nm)
                if nm not in _tp:
                    _out.append(nm)
                print("[전체시냅스] %-28s n=%d mean_abs_dg=%.5f changed_frac=%.1f%% w_mean %.4f→%.4f 이식=%s"
                      % (nm, b_.size, d.mean(), (d > 0).mean() * 100, b_.mean(), a_.mean(), "Y" if nm in _tp else "N"))
        print("[전체시냅스] 집단 %d개(g 변수) 중 변함 %d개, 이식 목록 %d개, **이식 밖 변함 %d개**: %s"
              % (len(syn_before), len(_chg), len(_tp), len(_out), ", ".join(_out) if _out else "-"))
    if args.trace_kc_motor and TRACE_ROWS:
        import csv
        with open(args.trace_kc_motor, "w", newline="", encoding="utf-8") as _fh:
            _w = csv.writer(_fh)
            _w.writerow(["ep", "t", "good_side", "explore", "v", "correct",
                         "e_ll", "g_ll", "e_lr", "g_lr", "e_rl", "g_rl", "e_rr", "g_rr"])
            _w.writerows(TRACE_ROWS)
        print("[추적] KC→motor %d시행 → %s" % (len(TRACE_ROWS), args.trace_kc_motor))

    d1_after = snap_d1()
    for nm in sorted(d1_before):
        if nm in d1_after and d1_before[nm].shape == d1_after[nm].shape:
            b_, a_ = d1_before[nm], d1_after[nm]
            d = np.abs(a_ - b_)
            print("[D1가중치] %-14s |Δ|평균=%.5f 변화율=%.1f%% | 평균 %.3f→%.3f | std %.4f→%.4f"
                  % (nm, d.mean(), (d > 1e-9).mean() * 100, b_.mean(), a_.mean(), b_.std(), a_.std()))

    if args.transplant_eval:
        # INV-B5. 사후 측정만 이식 경로로 간다. 사전은 훈련 전 뇌라 이력이 이미 동일하다.
        # **훈련에 쓴 cfg 그대로** 새 뇌를 만든다 — 필드를 골라 옮기면 구조가 달라진다(2026-09-19 사고).
        import transplant_eval as TE
        _w = TE.pull(brain)
        if args.save_weights:
            # E112: 학습된 가중치 저장 — 부분 이식 분해(--decomp-weights)에 쓴다.
            np.savez_compressed(args.save_weights, **_w)
            print("[저장] 학습 가중치 %d개 경로 → %s" % (len(_w), args.save_weights))
        _b2, _env2, _obs2 = TE.build_from_cfg(cfg, _bseed, env_seed=_eseed, env_cfg=_ecfg)
        TE.push(_b2, _w)          # 크기 불일치면 예외 — 조용히 넘어가지 않는다
        _names = TE.verify(brain, _b2, _w)   # 이식이 실제로 반영됐는지 확인
        print('[이식] %d개 경로: %s' % (len(_names), ', '.join(_names)))
        post, off1, mod1 = evaluate(_b2, _obs2, nh, args.trials)
    else:
        post, off1, mod1 = evaluate(brain, obs, nh, args.trials)
    print("[사후] 오프셋 %+.3f | 정답률 %.1f%% | **변조폭 %+.4f**" % (off1, post, mod1))
    dmod = mod1 - mod0
    print("=> 정답률 %+.1f%%p | **변조폭 변화 %+.4f** | 판정: %s"
          % (post - pre, dmod,
             "학습이 조향을 역전 방향으로 이동" if dmod < -0.02
             else ("학습이 반사 방향으로 강화" if dmod > 0.02 else "변화 없음")))

    if args.dump_kc_weights:
        # 2026-09-19: 예전엔 kc_to_d1만 찍었다. 그런데 조건에 따라 **학습하는 경로가 다르다**
        # (KC 학습을 끄면 food_to_d1 계열이 학습한다). 조작검증이 "가소성이 멈췄는가"를
        # 확인하려면 **이식 대상 전체**를 봐야 한다. 목록은 뇌에서 유도한다.
        import numpy as _np
        import transplant_eval as _TE
        try:
            _names = _TE.learned_names(brain)
        except Exception as _e:
            _names = ()
            print("  [경고] 학습 경로 목록 확인 실패: %s" % _e)
        for _nm in _names:
            try:
                _w = _TE._read_g(getattr(brain, _nm), _nm)
                print("  가중치 %-22s n=%d 평균 %.4f std %.4f 중앙 %.4f 최소 %.4f 최대 %.4f"
                      % (_nm, _w.size, _w.mean(), _w.std(), _np.median(_w), _w.min(), _w.max()))
            except Exception as _e:
                print("  가중치 %-22s 측정실패 %s: %s" % (_nm, type(_e).__name__, _e))

if __name__ == "__main__":
    main()
