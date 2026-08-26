#!/usr/bin/env python3
"""진짜 R-STDP 가중치 갱신 모델 (C36) — 시냅스별 신용 할당.

C35에서 확인된 것:
- `food_to_d1`은 `_create_static_synapse` = **정적**인데 로그만 "R-STDP"라 출력했다.
- `update_cortical_rstdp`의 갱신식은 `w[:] += eta * trace` = **전 시냅스 동일 스칼라**.
  시냅스별 자격흔적이 없어 std가 0에서 변하지 않았고, **변별 학습이 원리적으로 불가능**했다.

여기서 구현하는 것 = 표준 3요소 학습(three-factor learning):
  1. **시냅스별 자격흔적 e**: pre-post 스파이크 상관으로 쌓인다(STDP)
     - post 스파이크 시: e += A_plus * preTrace   (LTP 방향)
     - pre 스파이크 시:  e -= A_minus * postTrace (LTD 방향)
  2. **자격흔적 감쇠**: e *= exp(-dt/tau_e) — 보상이 늦게 와도 신용이 남도록
  3. **도파민 게이팅**: g += eta * dopamine * e — 보상 신호가 있을 때만 흔적이 가중치로 굳는다

핵심 차이: e가 **시냅스마다 다르게** 쌓이므로, 같은 도파민을 줘도 **활동한 시냅스만** 강화된다
= 신용 할당. 전역 스칼라 방식으로는 불가능했던 변별이 여기서 가능해진다.

dopamine은 extra_global_param(스칼라)으로, 파이썬에서 매 스텝 설정한다.
"""
from pygenn import create_weight_update_model, init_var


def make_rstdp_model():
    """시냅스별 자격흔적 + 도파민 게이팅 R-STDP."""
    return create_weight_update_model(
        "RSTDPEligibility",
        params=[
            ("tau_pre", "scalar"),     # pre 흔적 시정수
            ("tau_post", "scalar"),    # post 흔적 시정수
            ("tau_e", "scalar"),       # 자격흔적 시정수(보상 지연 허용폭)
            ("A_plus", "scalar"),      # LTP 이득
            ("A_minus", "scalar"),     # LTD 이득
            ("eta", "scalar"),         # 학습률
            ("w_min", "scalar"),
            ("w_max", "scalar"),
            # PyGeNN 5.4는 스칼라 extra_global_param을 폐기했다.
            # dopamine은 일반 param으로 선언하고 `syn.set_wu_param_dynamic("dopamine")`로
            # 동적 지정한 뒤 `syn.set_dynamic_param_value("dopamine", x)`로 매 스텝 갱신한다.
            ("dopamine", "scalar"),
        ],
        vars=[
            ("g", "scalar"),           # 가중치 (시냅스별)
            ("e", "scalar"),           # 자격흔적 (시냅스별) ← 신용 할당의 핵심
        ],
        pre_vars=[("preTrace", "scalar")],
        post_vars=[("postTrace", "scalar")],

        # pre 스파이크: 시냅스 전달 + LTD 방향 흔적
        pre_spike_syn_code="""
        addToPost(g);
        e -= A_minus * postTrace;
        """,
        # post 스파이크: LTP 방향 흔적 (이 시냅스가 post 발화에 기여했으면 preTrace가 크다)
        post_spike_syn_code="""
        e += A_plus * preTrace;
        """,
        # 매 스텝: 흔적 감쇠 + 도파민이 있을 때만 가중치로 굳힘
        synapse_dynamics_code="""
        e -= e * (dt / tau_e);
        if (dopamine != 0.0) {
            g += eta * dopamine * e;
            g = fmin(w_max, fmax(w_min, g));
        }
        """,
        pre_spike_code="preTrace += 1.0;",
        post_spike_code="postTrace += 1.0;",
        pre_dynamics_code="preTrace -= preTrace * (dt / tau_pre);",
        post_dynamics_code="postTrace -= postTrace * (dt / tau_post);",
    )


DEFAULT_PARAMS = {
    "tau_pre": 20.0,
    "tau_post": 20.0,
    "tau_e": 200.0,    # 보상이 200ms 뒤에 와도 신용이 남는다
    "A_plus": 1.0,
    "A_minus": 0.8,
    "eta": 0.02,
    "w_min": 0.0,
    "w_max": 20.0,
    "dopamine": 0.0,   # 런타임에 set_dynamic_param_value("dopamine", x)로 갱신
}


def rstdp_init(init_w: float, params: dict = None):
    """(model, params, vars, pre_vars, post_vars) 튜플 반환 — add_synapse_population용."""
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return (
        make_rstdp_model(),
        p,
        {"g": init_var("Constant", {"constant": init_w}), "e": 0.0},
        {"preTrace": 0.0},
        {"postTrace": 0.0},
    )
