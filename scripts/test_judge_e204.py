#!/usr/bin/env python3
"""judge_e204 합성 시험 — 가짜 학습·평가 로그·가중치(세 팔 G18·GN·UN + E200 비교)로 판정이 사전 규칙과 같은지.
결과 시나리오·경계(4/5·m 차 500·일반화 등호)·조작검증 실패(게이팅 비·구성 줄·시드·평가 세계·게이팅 없는 팔의 게이팅 줄·정규화 잔차·기준선·E200 같은 가중치·결측).
가짜 자료 함수는 test_judge_e199 를 재사용. 실행: python3 scripts/test_judge_e204.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e204 as J  # noqa: E402
import test_judge_e199 as T9  # noqa: E402

NK = 3000
POPS = T9.POPS
CELLS = J.CELLS
B5 = J.BRAINS


def write_train(path, arm, b, o):
    T9.write_train(path, n_comp=o.get("n_comp", 3000), d1=o.get("d1", 4.5e-13), eta=2.4, cchk=3)
    W, T0, C0 = o.get("world", J.WORLD[arm]), o.get("trial0", J.TRIAL0[arm]), J.CTX0[arm]
    with open(path, "a", encoding="utf-8") as f:
        for t in range(3):
            f.write("[E200 도파민 기준선] 시행 %d r=+1.00 r̄=0.2500 → DA=+0.7500\n" % (t + 1))
        f.write("[E200 도파민 통계] 시행 3000 · 평균 DA %+.4f · 평균 |DA| 0.7400 · 기준선 1 · r̄ 끝 0.2700 · τ 100 · r̄₀ 0.25\n" % o.get("da_mean", 0.0012))
        f.write("[범주 과제] catrule — 원시 특징 노출 사례 범주당 6(세계 시드 %d), 범주 A 정답 교차·B 정답 같은 쪽, 배정 지역 난수 %d, 자극 = 먹이 광선\n" % (W, T0 + b))
        f.write("[맥락 범주 과제] catctx — 맥락 X(zz_ctx2)·Y(zz_ctx) 반반(지역 난수 %d), 정답: X·A 교차\n" % (C0 + b))
        if J.GATED[arm] or o.get("gate_lines_in_un"):
            for _ in range(o.get("build", 2)):
                f.write("  [E204 맥락 게이팅 KC] 무리 X 좌 1493·우 1463 / Y 좌 1507·우 1537, 억제 w=-20.00 p=0.100, 연결 zz_ctx2→Y 좌 30185·우 30746 · zz_ctx→X 좌 29857·우 29482\n")
            for t in range(o.get("gchk", 3)):
                f.write("[E204 맥락 게이팅 점검] 시행 %d 맥락 X — 행동 창 끝 KC 발화 활성 무리 %d · 비활성 무리 0 (좌 X 237·Y 0 / 우 X 105·Y 0)\n" % (t, o.get("gact", 342)))
            f.write("[E204 맥락 게이팅 KC] 시행 %d · 행동 창 끝 KC 발화 활성 무리 합 1000000 · 비활성 무리 합 %d · 비(비활성/활성) %.4f\n"
                    % (o.get("gn", 3000), int(o.get("ratio", 0.0001) * 1e6), o.get("ratio", 0.0001)))


def write_eval(path, arm, ctx, mods, pushed, o):
    T9.write_eval(path, ctx, mods, pushed)
    with open(path, "a", encoding="utf-8") as f:
        f.write("[E204 평가 세계] 세계 시드 %d · 원형 A 1,2,3,4,5 … · 원형 B 6,7,8,9,10 …\n" % o.get("eval_world", J.WORLD[arm]))
        if J.GATED[arm] and not o.get("no_eval_build"):
            f.write("  [E204 맥락 게이팅 KC] 무리 X 좌 1493·우 1463 / Y 좌 1507·우 1537, 억제 w=-20.00 p=0.100, 연결 …\n")


def make_weights(E, arm, b, rng, o):
    pre = np.repeat(np.arange(NK), 2).astype(np.uint32)
    d = rng.uniform(-140, 140, NK)
    W, S, W2 = {}, {}, {}
    for kn in "lr":
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    for p in POPS:
        S[p + "__pre"] = pre
        S[p + "__post"] = np.zeros(pre.size, np.uint32)
        W2[p] = W[p].copy() if o.get("same_as_e200") else np.full(pre.size, 150.0, np.float32)
    for d_ in ("traces/E204", "traces/E200", "traces/E176/snap"):
        os.makedirs(os.path.join(E, d_), exist_ok=True)
    np.savez(os.path.join(E, "traces/E204/w_%s_b%d.npz" % (arm, b)), **W)
    if not os.path.exists(os.path.join(E, "traces/E200/w_b%d.npz" % b)) or o.get("same_as_e200"):
        np.savez(os.path.join(E, "traces/E200/w_b%d.npz" % b), **W2)
    np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **S)


def av_all(tr, nw):
    return {(x, c): (tr, nw) for x, c, _ in CELLS}


def build(root, spec, mods):
    """spec[팔][뇌] = av(칸 → (a_train, a_new)) — 팔 G18·GN·UN·U18. mods[(팔, 뇌)] = 옵션."""
    E = os.path.join(root, "research/experiments")
    os.makedirs(os.path.join(E, "logs/E204"), exist_ok=True)
    os.makedirs(os.path.join(E, "logs/E200"), exist_ok=True)
    rng = np.random.RandomState(0)
    for b in B5:
        for x in "XY":
            T9.write_eval(os.path.join(E, "logs/E200/ev_b%d_none_%s.log" % (b, x)), x, T9.NONE_M, 0)
            T9.write_eval(os.path.join(E, "logs/E200/ev_b%d_learn_%s.log" % (b, x)), x, T9.learn_mods(spec["U18"][b]), 8)
    for arm in J.ARMS:
        for b in B5:
            o = mods.get((arm, b), {})
            if o.get("missing_train"):
                continue
            write_train(os.path.join(E, "logs/E204/train_%s_b%d.log" % (arm, b)), arm, b, o)
            make_weights(E, arm, b, rng, o)
            for x in "XY":
                write_eval(os.path.join(E, "logs/E204/ev_%s_b%d_none_%s.log" % (arm, b, x)), arm, x, T9.NONE_M, 0, o)
                write_eval(os.path.join(E, "logs/E204/ev_%s_b%d_learn_%s.log" % (arm, b, x)), arm, x, T9.learn_mods(spec[arm][b]), 8, o)
    return E


def run(spec, mods=None):
    root = tempfile.mkdtemp(prefix="te204_")
    try:
        E = build(root, spec, mods or {})
        J.EXP = E
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        out = buf.getvalue()
        v = [ln for ln in out.splitlines() if ln.startswith("판정: ")]
        return (v[-1][len("판정: "):] if v else "?"), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def spec_of(g18, gn, un, u18):
    return {"G18": g18, "GN": gn, "UN": un, "U18": u18}


GOOD = {b: av_all(2000, 1500) for b in B5}
WEAK = {b: {**av_all(2500, 1800), ("X", "A"): (300, 200)} for b in B5}      # 가장 약한 칸 300(부분 3칸) — 게이팅 없음 꼴
FAIL = {b: av_all(200, 100) for b in B5}
PART = {b: {**av_all(2000, 1500), ("Y", "B"): (500, 300)} for b in B5}


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = (got == want) if not callable(want) else want(got)
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    chk("H125(두 세계 게이팅 효과·GN 범주)", run(spec_of(GOOD, GOOD, WEAK, WEAK))[0], "H125")
    chk("세계 한정(GN 부분)", run(spec_of(GOOD, PART, WEAK, WEAK))[0], "H125-세계 한정")
    chk("새 세계는 게이팅 없이도 섬(UN 도 범주)", run(spec_of(GOOD, GOOD, GOOD, WEAK))[0], "보류(새 세계는 게이팅 없이도 섬)")
    chk("게이팅 무관(E200 도 같은 m)", run(spec_of(GOOD, GOOD, WEAK, GOOD))[0], "게이팅 무관")
    chk("H125-null(G18 실패 5/5)", run(spec_of(FAIL, FAIL, WEAK, WEAK))[0], "H125-null")
    chk("보류(G18 부분 5/5)", run(spec_of(PART, GOOD, WEAK, WEAK))[0], "보류")
    g4 = {**GOOD, 14: PART[14]}
    chk("경계: G18 범주 4/5 → H125", run(spec_of(g4, GOOD, WEAK, WEAK))[0], "H125")
    g3 = {**GOOD, 13: PART[13], 14: PART[14]}
    chk("경계: G18 범주 3/5 → 보류", run(spec_of(g3, GOOD, WEAK, WEAK))[0], "보류")
    f4 = {**FAIL, 14: PART[14]}
    chk("경계: G18 실패 4/5 → H125-null", run(spec_of(f4, FAIL, WEAK, WEAK))[0], "H125-null")
    u_eq = {b: {**av_all(2500, 1800), ("X", "A"): (1500, 900)} for b in B5}     # m 1500 — G18 m 2000 과 정확히 500 차
    chk("경계: m 차 정확히 500 → 게이팅 효과 있음", run(spec_of(GOOD, GOOD, WEAK, u_eq))[0], "H125")
    u_lt = {b: {**av_all(2500, 1800), ("X", "A"): (1501, 900)} for b in B5}     # 499 차
    chk("경계: m 차 499 → 게이팅 무관", run(spec_of(GOOD, GOOD, WEAK, u_lt))[0], "게이팅 무관")
    eqg = {b: av_all(2000, 1000) for b in B5}                                  # a_new 1000·2·a_new = a_train 등호 → 범주
    chk("경계: 일반화 등호(2·a_new = a_train) → 범주", run(spec_of(eqg, GOOD, WEAK, WEAK))[0], "H125")
    ltg = {b: av_all(2000, 999) for b in B5}                                   # 2·999 < 2000 → 사례 한정 → 범주 0 → 보류
    chk("경계: 일반화 미달(2·999 < 2000) → 보류", run(spec_of(ltg, GOOD, WEAK, WEAK))[0], "보류")
    base = spec_of(GOOD, GOOD, WEAK, WEAK)
    chk("조작: 게이팅 비 0.011 > 0.01 → 보류", run(base, {("G18", 12): {"ratio": 0.011}})[0], "보류(조작검증 실패)")
    chk("조작: 게이팅 비 0.0100 = 문턱 → 통과", run(base, {("G18", 12): {"ratio": 0.0100}})[0], "H125")
    chk("조작: 게이팅 구성 줄 1(내부 평가 뇌 없음) → 보류", run(base, {("GN", 10): {"build": 1}})[0], "보류(조작검증 실패)")
    chk("조작: 점검 줄 2 → 보류", run(base, {("G18", 11): {"gchk": 2}})[0], "보류(조작검증 실패)")
    chk("조작: 활성 무리 발화 0 → 보류", run(base, {("G18", 11): {"gact": 0}})[0], "보류(조작검증 실패)")
    chk("조작: 요약 시행 2999 → 보류", run(base, {("GN", 13): {"gn": 2999}})[0], "보류(조작검증 실패)")
    chk("조작: GN 세계 시드 18300 → 보류", run(base, {("GN", 10): {"world": 18300}})[0], "보류(조작검증 실패)")
    chk("조작: UN 배정 시드 틀림 → 보류", run(base, {("UN", 14): {"trial0": 18500}})[0], "보류(조작검증 실패)")
    chk("조작: GN 평가 세계 18300 → 보류", run(base, {("GN", 12): {"eval_world": 18300}})[0], "보류(조작검증 실패)")
    chk("조작: 게이팅 팔 평가에 구성 줄 없음 → 보류", run(base, {("G18", 10): {"no_eval_build": True}})[0], "보류(조작검증 실패)")
    chk("조작: 게이팅 없는 팔(UN)에 게이팅 줄 → 보류", run(base, {("UN", 11): {"gate_lines_in_un": True}})[0], "보류(조작검증 실패)")
    chk("조작: 정규화 잔차 2e-3 → 보류", run(base, {("G18", 13): {"d1": 2e-3}})[0], "보류(조작검증 실패)")
    chk("조작: 평균 DA 0.06 → 보류", run(base, {("GN", 14): {"da_mean": 0.06}})[0], "보류(조작검증 실패)")
    chk("조작: G18 가중치 = E200 → 보류", run(base, {("G18", 10): {"same_as_e200": True}})[0], "보류(조작검증 실패)")
    chk("결측: UN b12 학습 로그 없음 → 보류(결측)", run(base, {("UN", 12): {"missing_train": True}})[0], lambda g: g.startswith("보류(결측"))
    v, out = run(base)
    chk("보고 줄: 팔·뇌 20줄·팔별 범주 수 4줄·게이팅 효과 줄", (sum(1 for ln in out.splitlines() if " 규칙 방향 효과(1e-4" in ln),
                                                  sum(1 for ln in out.splitlines() if ": 맥락 의존 범주 " in ln),
                                                  sum(1 for ln in out.splitlines() if ln.startswith("게이팅 효과"))), (20, 4, 1))
    # decide 직접(경계 조합)
    cat = {a: {b: "conc" for b in B5} for a in ("G18", "GN", "UN", "U18")}
    m = {"G18": {b: 2000 for b in B5}, "GN": {b: 2000 for b in B5}, "UN": {b: 1500 for b in B5}, "U18": {b: 1500 for b in B5}}
    chk("decide: 두 세계 효과 정확히 500 → H125", J.decide(cat, m)[0], "H125")
    m2 = {**m, "UN": {**m["UN"], 10: 1501, 11: 1501}}
    chk("decide: 새 세계 효과 3/5 → 보류(새 세계…)", J.decide(cat, m2)[0], "보류(새 세계는 게이팅 없이도 섬)")
    print("합성 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
