#!/usr/bin/env python3
"""judge_e205 합성 시험 — 가짜 학습·평가 로그·가중치(L4 팔 GW·UW + 앞 능력 R·V·P, 뇌 16~20)로 판정이 사전 규칙(criteria_fixed.txt)과 같은지.
시나리오(통과·유지만 간섭·체계 확증·미확증·H126-null·부분 반전·기준/전제 미달)·경계(4/5·m 차 500·유지 몫 등호·반전 일반화 등호)·
조작검증 실패(게이팅 비·맥락 켬 수·알림 맥락·세계 시드·원형·포화·게이팅 없는 팔의 게이팅 줄·반전 줄 빠짐·둘째 쌍 줄이 R 에·V=R 가중치·결측).
L4 가짜 자료 함수는 test_judge_e199 재사용. 실행: python3 scripts/test_judge_e205.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e205 as J  # noqa: E402
import test_judge_e199 as T9  # noqa: E402

NK = 3000
POPS = T9.POPS
CELLS = J.CELLS
B5 = J.BRAINS
NONE_T = {(c, s): v for c, vs in {"A": (500, 400), "B": (-300, -200), "C": (200, 150), "D": (-100, -50)}.items() for s, v in zip(("train", "new"), vs)}


def write_l4_train(path, arm, b, o):
    T9.write_train(path, n_comp=o.get("n_comp", 3000), d1=4.5e-13, eta=2.4, cchk=3)
    with open(path, "a", encoding="utf-8") as f:
        for t in range(3):
            f.write("[E200 도파민 기준선] 시행 %d r=+1.00 r̄=0.2500 → DA=+0.7500\n" % (t + 1))
        f.write("[E200 도파민 통계] 시행 3000 · 평균 DA +0.0012 · 평균 |DA| 0.7400 · 기준선 1 · r̄ 끝 0.2700 · τ 100 · r̄₀ 0.25\n")
        f.write("[범주 과제] catrule — 원시 특징 노출 사례 범주당 6(세계 시드 %d), 범주 A 정답 교차·B 정답 같은 쪽, 배정 지역 난수 %d, 자극 = 먹이 광선\n"
                % (o.get("world", J.WORLD), J.TRIAL0 + b))
        f.write("[맥락 범주 과제] catctx — 맥락 X(zz_ctx2)·Y(zz_ctx) 반반(지역 난수 %d), 정답: X·A 교차\n" % (J.CTX0 + b))
        if J.GATED[arm] or o.get("gate_in_uw"):
            gate_lines(f, 3000, o)


def gate_lines(f, n, o):
    for _ in range(2):
        f.write("  [E204 맥락 게이팅 KC] 무리 X 좌 1493·우 1463 / Y 좌 1507·우 1537, 억제 w=-20.00 p=0.100, 연결 zz_ctx2→Y 좌 30185·우 30746 · zz_ctx→X 좌 29857·우 29482\n")
    for t in range(3):
        f.write("[E204 맥락 게이팅 점검] 시행 %d 맥락 X — 행동 창 끝 KC 발화 활성 무리 342 · 비활성 무리 0 (좌 X 237·Y 0 / 우 X 105·Y 0)\n" % t)
    r = o.get("ratio", 0.0001)
    f.write("[E204 맥락 게이팅 KC] 시행 %d · 행동 창 끝 KC 발화 활성 무리 합 1000000 · 비활성 무리 합 %d · 비(비활성/활성) %.4f\n" % (o.get("gn", n), int(r * 1e6), r))


def write_task_train(path, t, b, o):
    n = J.TASKS[t]
    nb = o.get("nb", {"R": 750, "V": 1500, "P": 1500}[t])
    with open(path, "w", encoding="utf-8") as f:
        f.write("    KC→motor [E199 가산형(경성 상한) 갱신 — g += eta·DA·e, clip]\n")
        f.write("    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=2.4, tau_e=12.0, sparsity=0.25\n")
        f.write("[E153 종류 입력 적재] x.npz 검증 일치 — a=1\n")
        if not o.get("no_ann"):
            f.write("[범주 규칙 맥락] catrule — 맥락 %s(%s) 고정: 결정·행동 창 동안 켬, 보상 창·간격 끔(규칙·배정 지역 난수 소비 불변)\n"
                    % (o.get("ann_ctx", "X"), "zz_ctx2" if o.get("ann_ctx", "X") == "X" else "zz_ctx"))
        f.write("[범주 과제] catrule — 원시 특징 노출 사례 범주당 6(세계 시드 %d), 범주 A 정답 교차·B 정답 같은 쪽, 배정 지역 난수 %d, 자극 = 먹이 광선\n"
                % (o.get("world", J.WORLD), J.TRIAL0 + b))
        gate_lines(f, n, o)
        for k in range(3):
            f.write("[범주 과제 점검] 시행 %d 범주 B 사례 5 쪽 left 정답 교차 — 행동 창 끝 특징 발화 120 · 시행 끝 0\n" % k)
            f.write("[맥락 범주 점검] 시행 %d 맥락 X 범주 B 정답 교차 — 행동 창 끝 켠 맥락 발화 600 · 끈 맥락 발화 %d · 시행 끝 두 맥락 발화 0\n" % (k, o.get("ctx_off", 0)))
        if t == "V" and not o.get("no_rev"):
            f.write("[범주 반전] 시행 1500 부터 정답 A → 같은 쪽 · B → 교차\n")
        if t == "P" or o.get("p2_in"):
            f.write("[둘째 범주 쌍] 시행 1500 부터 범주 C(같은 쪽)·D(교차) — C 는 A 와, D 는 B 와 원형 단위 공유\n")
        for k in range(3):
            f.write("[E200 도파민 기준선] 시행 %d r=+1.00 r̄=0.2500 → DA=+0.7500\n" % (k + 1))
        f.write("[E199 경쟁 정규화] 시행 1 적용 — 정규화 전 max|S−300| 쪽 l 0.1·r 0.1 → 뒤 1e-13·1e-13\n")
        f.write("[학습] %dep 완료, 보상 900회 (탐색 주입 1000회, ε=0.60)\n" % (n // 100))
        f.write("[E199 경쟁 정규화] 적용 %d회 · 정규화 전 max|S−목표| 평균 1.2345 최대 5.6789 · 정규화 뒤 최대 4.50e-13 · hard=1\n" % n)
        f.write("[E200 도파민 통계] 시행 %d · 평균 DA +0.0012 · 평균 |DA| 0.7400 · 기준선 1 · r̄ 끝 0.2700 · τ 100 · r̄₀ 0.25\n" % n)
        f.write("[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nb, n - nb))
        f.write("[범주 규칙 맥락] 시행 %d 중 맥락 X 켬 %d\n" % (n, o.get("ncnt", n)))
        if t == "V" and not o.get("no_rev"):
            f.write("[범주 반전] 반전 시행 1500 (시행 1500 부터, 전체 3000)\n")
        if t == "P" or o.get("p2_in"):
            f.write("[둘째 범주 쌍] 둘째 쌍 시행 1500 (시행 1500 부터, 전체 3000)\n")


def eval_tail(f, o, gated=True):
    f.write("[E204 평가 세계] 세계 시드 %d · 원형 A %s,33,40,41,44 … · 원형 B 5,10,14,19,20 …\n" % (o.get("eval_world", J.WORLD), o.get("proto", J.PROTO_A)))
    if gated and not o.get("no_eval_build"):
        f.write("  [E204 맥락 게이팅 KC] 무리 X 좌 1493·우 1463 / Y 좌 1507·우 1537, 억제 w=-20.00 p=0.100, 연결 …\n")


def write_l4_eval(path, arm, ctx, mods, pushed, o):
    T9.write_eval(path, ctx, mods, pushed, sat=o.get("sat", False))
    with open(path, "a", encoding="utf-8") as f:
        eval_tail(f, o, J.GATED[arm])


def write_task_eval(path, cats, e, pushed, o):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[E153 종류 입력 적재] 검증 일치 n=4\n")
        eval_tail(f, o)
        for c in cats:
            for s in ("train", "new"):
                v = NONE_T[(c, s)] + (e[(c, s)] if e is not None else 0)
                mo = "0.700000/0.700000" if o.get("sat") else "0.500000/0.400000"
                f.write("RCEVAL cat=%s set=%s mod=%s off=+0.0000 acc=50.0 trials=96 feat_on=96 ctx=X ctx_on=%d | side=left n=48 motor L/R %s\n"
                        % (c, s, T9.fmt(v), o.get("ctx_on", 96), mo))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=%d\n" % pushed)


def save_weights(E, name, b, seed, same_as=None):
    os.makedirs(os.path.join(E, "traces/E205"), exist_ok=True)
    os.makedirs(os.path.join(E, "traces/E176/snap"), exist_ok=True)
    pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
    if same_as:
        Z = np.load(os.path.join(E, "traces/E205/w_%s_b%d.npz" % (same_as, b)))
        np.savez(os.path.join(E, "traces/E205/w_%s_b%d.npz" % (name, b)), **{k: Z[k] for k in Z.files})
    else:
        d = np.random.RandomState(seed).uniform(-140, 140, NK)
        W = {}
        for kn in "lr":
            W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
            W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
        np.savez(os.path.join(E, "traces/E205/w_%s_b%d.npz" % (name, b)), **W)
    S = {}
    for p in POPS:
        S[p + "__pre"] = pre
        S[p + "__post"] = np.zeros(pre.size, np.uint32)
    np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **S)


def av_all(tr, nw):
    return {(x, c): (tr, nw) for x, c, _ in CELLS}


def ab(eA, eAn, eB, eBn, **cd):
    """e 값(1e-4) — A·B(train/new) + 선택 C·D."""
    e = {("A", "train"): eA, ("A", "new"): eAn, ("B", "train"): eB, ("B", "new"): eBn}
    for k, (t_, n_) in cd.items():
        e[(k, "train")], e[(k, "new")] = t_, n_
    return e


R_OK = ab(-2000, -1500, 2000, 1500, C=(300, 200), D=(-300, -200))
V_OK = ab(1500, 1000, -1500, -1000)
P_KEEP = ab(-1500, -1000, 1500, 1000, C=(2000, 1500), D=(-2000, -1500))
P_INTF = ab(-400, -300, 300, 200, C=(2000, 1500), D=(-2000, -1500))


def build(root, spec, mods):
    """spec = {GW: {b: av}, UW: {b: av}, R: {b: e}, V: {b: e}, P: {b: e}}; mods[(이름, 뇌)] = 옵션."""
    E = os.path.join(root, "research/experiments")
    L = os.path.join(E, "logs/E205")
    os.makedirs(L, exist_ok=True)
    for b in B5:
        for arm in J.L4:
            o = mods.get((arm, b), {})
            if o.get("missing"):
                continue
            write_l4_train(os.path.join(L, "train_%s_b%d.log" % (arm, b)), arm, b, o)
            save_weights(E, arm, b, 1000 * b + (1 if arm == "GW" else 2), same_as=o.get("same_as"))
            for x in "XY":
                write_l4_eval(os.path.join(L, "ev_%s_b%d_none_%s.log" % (arm, b, x)), arm, x, T9.NONE_M, 0, o)
                write_l4_eval(os.path.join(L, "ev_%s_b%d_learn_%s.log" % (arm, b, x)), arm, x, T9.learn_mods(spec[arm][b]), 8, o)
        for t in ("R", "V", "P"):
            o = mods.get((t, b), {})
            write_task_train(os.path.join(L, "train_%s_b%d.log" % (t, b)), t, b, o)
        save_weights(E, "R", b, 1000 * b + 3)
        save_weights(E, "V", b, 1000 * b + 4, same_as=mods.get(("V", b), {}).get("same_as"))
        save_weights(E, "P", b, 1000 * b + 5)
        for nm, t, cats, pw in J.EVALS:
            o = mods.get((nm, b), {})
            write_task_eval(os.path.join(L, "ev_b%d_%s.log" % (b, nm)), cats, spec[t][b] if t else None, pw, o)
    return E


def run(spec, mods=None):
    root = tempfile.mkdtemp(prefix="te205_")
    try:
        E = build(root, spec, mods or {})
        J.EXP = E
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        out = buf.getvalue()
        v = [ln for ln in out.splitlines() if ln.startswith("판정: ")]
        h = [ln for ln in out.splitlines() if ln.startswith("가설: ")]
        return (v[-1][len("판정: "):] if v else "?"), (h[-1] if h else ""), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


GOOD = {b: av_all(2000, 1500) for b in B5}
WEAK = {b: {**av_all(2500, 1800), ("X", "A"): (300, 200)} for b in B5}
PART = {b: {**av_all(2000, 1500), ("Y", "B"): (500, 300)} for b in B5}


def spec(gw=GOOD, uw=WEAK, r=None, v=None, p=None):
    return {"GW": gw, "UW": uw, "R": r or {b: R_OK for b in B5}, "V": v or {b: V_OK for b in B5}, "P": p or {b: P_KEEP for b in B5}}


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = (got == want) if not callable(want) else want(got)
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    v, h, out = run(spec())
    chk("통과(L4 확증·셋 보존)", v, "통합 관문 통과")
    chk("가설 줄(통과)", h, "가설: H125 확증(새 뇌·처음 쓰는 세계) · H126 지지")
    v, h, _ = run(spec(p={b: P_INTF for b in B5}))
    chk("유지만 간섭", v, "통합 관문 미통과 — L4 확증 · 범주 규칙 보존 · 반전 보존 · 유지 잃음")
    chk("가설 줄(유지만 간섭)", h, "가설: H125 확증(새 뇌·처음 쓰는 세계) · 유지만 간섭")
    v, h, _ = run(spec(uw=GOOD))
    chk("체계 확증(UW 도 같은 크기)", v, "통합 관문 미통과 — L4 체계 확증(게이팅 효과 미확증) · 범주 규칙 보존 · 반전 보존 · 유지 보존")
    v, h, _ = run(spec(gw=PART))
    chk("L4 미확증(GW 부분)", v, "통합 관문 미통과 — L4 미확증 · 범주 규칙 보존 · 반전 보존 · 유지 보존")
    rf = {b: ab(-300, -200, 300, 200, C=(0, 0), D=(0, 0)) for b in B5}
    v, h, _ = run(spec(r=rf))
    chk("범주 규칙 잃음 → H126-null(유지는 기준 미달 보류)", (v, h), ("통합 관문 미통과 — L4 확증 · 범주 규칙 잃음 · 반전 보존 · 유지 보류(기준 미달)", "가설: H125 확증(새 뇌·처음 쓰는 세계) · H126-null"))
    vs = {b: ab(-1800, -1300, 1900, 1400) for b in B5}   # Δ A +200·B −100 → 고착
    v, h, _ = run(spec(v=vs))
    chk("반전 고착 → 반전 잃음·H126-null", (v, h[-9:]), ("통합 관문 미통과 — L4 확증 · 범주 규칙 보존 · 반전 잃음 · 유지 보존", "H126-null"))
    vp = {b: ab(-500, -300, 500, 300) for b in B5}       # Δ A +1500·B −1500, 획득 아님 → 부분 반전
    v, _, out = run(spec(v=vp))
    chk("부분 반전 5/5 → 반전 부분", v, "통합 관문 미통과 — L4 확증 · 범주 규칙 보존 · 반전 부분 · 유지 보존")
    chk("부분 반전 개수 줄", "부분 반전 5/5" in out, True)
    r_one = {b: ab(-2000, -1500, 800, 500, C=(0, 0), D=(0, 0)) for b in B5}
    v, _, _ = run(spec(r=r_one))
    chk("R 한 범주만 → 범주 규칙 부분·유지 기준 미달", v, lambda s: "범주 규칙 부분" in s and "유지 보류(기준 미달)" in s)
    p_nopre = {b: ab(-1500, -1000, 1500, 1000, C=(800, 500), D=(-2000, -1500)) for b in B5}
    v, _, _ = run(spec(p=p_nopre))
    chk("둘째 쌍 미획득 → 유지 보류(전제 미달)", v, lambda s: s.endswith("유지 보류(전제 미달)"))
    # 경계: 유지 몫 등호(2·e_P = e_R) → 유지
    p_eq = {b: ab(-1000, -700, 1000, 700, C=(2000, 1500), D=(-2000, -1500)) for b in B5}
    v, _, _ = run(spec(p=p_eq))
    chk("경계: 2·e_P,A = e_R,A·2·e_P,B = e_R,B → 유지", v, "통합 관문 통과")
    p_lt = {b: ab(-999, -700, 1000, 700, C=(2000, 1500), D=(-2000, -1500)) for b in B5}
    v, _, _ = run(spec(p=p_lt))
    chk("경계: e_P,A −999 (2·−999 > −2000) → 간섭", v, lambda s: s.endswith("유지 잃음"))
    # 경계: 반전 일반화 등호(2·e'_new = e'_train) → 성공
    v_eq = {b: ab(1500, 750, -1500, -750) for b in B5}
    v, _, _ = run(spec(v=v_eq))
    chk("경계: 반전 새 사례 = 노출/2 → 반전 성공", v, "통합 관문 통과")
    v_lt = {b: ab(1500, 749, -1500, -750) for b in B5}
    v, _, _ = run(spec(v=v_lt))
    chk("경계: 반전 새 사례 749 < 750 → 사례 한정 → 반전 부분", v, lambda s: "반전 부분" in s)
    # 경계: 게이팅 효과 4/5 · 차 정확히 500 / 499
    uw4 = {**WEAK, 20: GOOD[20]}
    v, _, _ = run(spec(uw=uw4))
    chk("경계: 게이팅 효과 4/5 → 확증", v, "통합 관문 통과")
    uw3 = {**WEAK, 19: GOOD[19], 20: GOOD[20]}
    v, _, _ = run(spec(uw=uw3))
    chk("경계: 게이팅 효과 3/5 → 체계 확증", v, lambda s: "L4 체계 확증" in s)
    uw500 = {b: av_all(1500, 1200) for b in B5}            # m 차 = 2000 − 1500 = 500
    v, _, _ = run(spec(uw=uw500))
    chk("경계: m 차 정확히 500 → 확증", v, "통합 관문 통과")
    uw499 = {b: av_all(1501, 1200) for b in B5}            # 499
    v, _, _ = run(spec(uw=uw499))
    chk("경계: m 차 499 → 체계 확증", v, lambda s: "L4 체계 확증" in s)
    gw4 = {**GOOD, 20: PART[20]}
    v, _, _ = run(spec(gw=gw4))
    chk("경계: GW 범주 4/5 → 확증", v, "통합 관문 통과")
    gw3 = {**GOOD, 19: PART[19], 20: PART[20]}
    v, _, _ = run(spec(gw=gw3))
    chk("경계: GW 범주 3/5 → 미확증", v, lambda s: "L4 미확증" in s)
    rc4 = {**{b: R_OK for b in B5}, 20: ab(-2000, -900, 2000, 1500, C=(300, 200), D=(-300, -200))}   # 1 뇌 사례 한정
    v, _, _ = run(spec(r=rc4))
    chk("경계: 개념 수준 4/5 → 범주 규칙 보존", v, "통합 관문 통과")
    # 조작검증 실패
    for nm, mod, want in [
        ("R 게이팅 비 0.011", {("R", 16): {"ratio": 0.011}}, "보류(조작검증 실패)"),
        ("V 맥락 켬 2999", {("V", 17): {"ncnt": 2999}}, "보류(조작검증 실패)"),
        ("P 알림 맥락 Y", {("P", 18): {"ann_ctx": "Y"}}, "보류(조작검증 실패)"),
        ("R 알림 없음", {("R", 19): {"no_ann": True}}, "보류(조작검증 실패)"),
        ("V 세계 시드 28300", {("V", 20): {"world": 28300}}, "보류(조작검증 실패)"),
        ("GW 학습 세계 18300", {("GW", 16): {"world": 18300}}, "보류(조작검증 실패)"),
        ("평가 원형 A 다름", {("P_learn", 16): {"proto": "1,2,3,4,5"}}, "보류(조작검증 실패)"),
        ("평가 세계 28300", {("none", 17): {"eval_world": 28300}}, "보류(조작검증 실패)"),
        ("평가 포화", {("R_learn", 18): {"sat": True}}, "보류(조작검증 실패)"),
        ("평가 ctx_on 95", {("V_learn", 19): {"ctx_on": 95}}, "보류(조작검증 실패)"),
        ("UW 에 게이팅 줄", {("UW", 20): {"gate_in_uw": True}}, "보류(조작검증 실패)"),
        ("GW 게이팅 비 0.02", {("GW", 18): {"ratio": 0.02}}, "보류(조작검증 실패)"),
        ("V 반전 줄 빠짐", {("V", 16): {"no_rev": True}}, "보류(조작검증 실패)"),
        ("R 에 둘째 쌍 줄", {("R", 17): {"p2_in": True}}, "보류(조작검증 실패)"),
        ("R 범주 배정 950(범위 밖)", {("R", 18): {"nb": 950}}, "보류(조작검증 실패)"),
        ("맥락 점검 끈 맥락 발화 3", {("P", 19): {"ctx_off": 3}}, "보류(조작검증 실패)"),
        ("V 가중치 = R", {("V", 20): {"same_as": "R"}}, "보류(조작검증 실패)"),
        ("UW 가중치 = GW", {("UW", 16): {"same_as": "GW"}}, "보류(조작검증 실패)"),
        ("평가 게이팅 구성 줄 없음", {("none", 18): {"no_eval_build": True}}, "보류(조작검증 실패)"),
    ]:
        v, _, _ = run(spec(), mod)
        chk("조작검증 실패: " + nm, v, want)
    v, _, _ = run(spec(), {("GW", 17): {"missing": True}})
    chk("결측(GW b17 로그 없음)", v, lambda s: s.startswith("보류(결측"))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
