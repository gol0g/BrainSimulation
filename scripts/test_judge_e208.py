#!/usr/bin/env python3
"""judge_e208 합성 시험 — 짝 E205(test_judge_e205)·E206(test_judge_e206 build6) 자료 위에 가짜 E208 학습·평가 로그·가중치(F-G0 + R·V·P, 뇌 16~20)를 만들어
판정이 사전 규칙과 같은지. 시나리오: 통과·H129-null·H129-맞교환(L4)·H129 부분(범주 규칙)·맞교환(반전)·보류(유지 기준 미달)·T_F0 −1500;
경계: L4 4/5·3/5; 조작검증 실패: 맥락 무게 20(한 로그)·zz_ctx2 줄 없음·연결 수 다름·p 다름·F 적재·동작점·가중치 = E206·같은 값(G0 = E206)·평가 세계; 결측(calib·로그·E206 맥락 줄).
실행: python3 test_judge_e208.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e208 as J  # noqa: E402
import verify_e208_independent as V  # noqa: E402
import test_judge_e205 as T5  # noqa: E402
import test_judge_e206 as T6  # noqa: E402
import test_judge_e199 as T9  # noqa: E402

B5 = J.BRAINS
NK = 3000
CNT = {"zz_ctx": (1234, 1222), "zz_ctx2": (1201, 1190)}


def ctx_lines(f, o, w):
    for pop, tag in (("zz_ctx", "[맥락 입력]"), ("zz_ctx2", "[맥락 입력 2]")):
        if o.get("no_" + pop):
            continue
        nl, nr = o.get("cnt_" + pop, CNT[pop])
        f.write("  %s 맥락 전용 집단 %s 200 → KC 좌·우, w=%.2f, p=%.3f, 연결 좌 %d·우 %d\n" % (tag, pop, o.get("w_" + pop, w), o.get("p", 0.02), nl, nr))


def save_w8(E, name, b, seed, same_as=None):
    d8 = os.path.join(E, "traces/E208")
    os.makedirs(d8, exist_ok=True)
    if same_as:
        Z = np.load(os.path.join(E, "traces", same_as))
        np.savez(os.path.join(d8, "w_%s_b%d.npz" % (name, b)), **{k: Z[k] for k in Z.files})
        return
    d = np.random.RandomState(seed).uniform(-140, 140, NK)
    W = {}
    for kn in "lr":
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    np.savez(os.path.join(d8, "w_%s_b%d.npz" % (name, b)), **W)


def build8(E, spec, mods, tf):
    L = os.path.join(E, "logs/E208")
    os.makedirs(L, exist_ok=True)
    if not mods.get("no_calib"):
        with open(os.path.join(L, "calib_tonic.txt"), "w", encoding="utf-8") as f:
            f.write("T_F=%d m_ref=0.6000 m_G0=0.5950 diff=0.0050\n" % tf)
    for b in B5:
        o = mods.get(("GW", b), {})
        if not o.get("missing"):
            tp = os.path.join(L, "train_GW_b%d.log" % b)
            T5.write_l4_train(tp, "GW", b, o)
            with open(tp, "a", encoding="utf-8") as f:
                T6.f_lines(f, b, o, tf); ctx_lines(f, o, 0.0)
            save_w8(E, "GW", b, 9000 * b + 1, same_as=("E206/w_GW_b%d.npz" % b) if o.get("same_as_gw") else None)
            for x in "XY":
                for w, mm, pw in (("none", T9.NONE_M, 0), ("learn", T9.learn_mods(spec["GW"][b]), 8)):
                    ep = os.path.join(L, "ev_GW_b%d_%s_%s.log" % (b, w, x))
                    oe = mods.get(("GWev", b, w, x), {})
                    T5.write_l4_eval(ep, "GW", x, mm, pw, oe)
                    with open(ep, "a", encoding="utf-8") as f:
                        T6.f_lines(f, b, oe, tf); ctx_lines(f, oe, 0.0)
        for t in ("R", "V", "P"):
            o = mods.get((t, b), {})
            tp = os.path.join(L, "train_%s_b%d.log" % (t, b))
            T5.write_task_train(tp, t, b, o)
            with open(tp, "a", encoding="utf-8") as f:
                T6.f_lines(f, b, o, tf); ctx_lines(f, o, 0.0)
        save_w8(E, "R", b, 9000 * b + 3)
        save_w8(E, "V", b, 9000 * b + 4)
        save_w8(E, "P", b, 9000 * b + 5)
        for nm, t, cats, pw in J.EVALS:
            o = mods.get((nm, b), {})
            ep = os.path.join(L, "ev_b%d_%s.log" % (b, nm))
            T5.write_task_eval(ep, cats, spec[t][b] if t else None, pw, o)
            with open(ep, "a", encoding="utf-8") as f:
                T6.f_lines(f, b, o, tf); ctx_lines(f, o, 0.0)


def run(spec8, spec6=None, mods=None, tf=-1500, mods6=None):
    root = tempfile.mkdtemp(prefix="te208_")
    try:
        E = T5.build(root, T5.spec(gw=T6.U_GW), {})                  # E205 U(E206 짝 경로가 쓰는 자료)
        T6.build6(E, spec6 or T6.spec(gw=GW6), mods6 or {}, -1300)   # E206 F-GW 짝
        for b in B5:                                                  # E206 학습 로그에 맥락 무게 20 줄(연결 수 = E208 기대)
            if not (mods or {}).get(("e206_noctx", b)):
                with open(os.path.join(E, "logs/E206/train_GW_b%d.log" % b), "a", encoding="utf-8") as f:
                    ctx_lines(f, {}, 20.0)
        build8(E, spec8, mods or {}, tf)
        J.EXP = E; V.EXP = E
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        out = buf.getvalue()
        vb = io.StringIO()
        with contextlib.redirect_stdout(vb):
            V.main()
        vout = vb.getvalue()
        pick = lambda s, p: ([ln[len(p):] for ln in s.splitlines() if ln.startswith(p)] or ["?"])[-1]  # noqa: E731
        RUNV.append((pick(vout, "독립 판정: "), pick(vout, "독립 가설: "), vout))
        return pick(out, "판정: "), pick(out, "가설: "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


RUNV = []   # run() 마다 독립 대조 (판정, 가설, 출력)
GW6 = {b: T5.av_all(1900, 1400) for b in B5}       # E206 F-GW 짝 — G0 와 값이 다름
GOOD = T6.GOOD
PART = T6.PART
FAIL = T6.FAIL


def spec(gw=GOOD, r=None, v=None, p=None):
    return {"GW": gw, "R": r or {b: T5.R_OK for b in B5}, "V": v or {b: T5.V_OK for b in B5}, "P": p or {b: T5.P_KEEP for b in B5}}


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = (got == want) if not callable(want) else want(got)
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    ab = T5.ab
    pv = "통합 관문 통과(개발 비교)"
    scen = [
        ("통과(넷 다 보존)", spec(), {}, -1500, pv, "H129 지지"),
        ("유지 잃음 → H129-null", spec(p={b: T5.P_INTF for b in B5}), {}, -1500,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 보존 · 유지 잃음", "H129-null(유지 안 섬)"),
        ("L4 부분·유지 보존 → H129-맞교환", spec(gw=PART), {}, -1500,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H129-맞교환(유지 ↔ L4)"),
        ("L4 잃음·유지 보존 → H129-맞교환", spec(gw=FAIL), {}, -1500,
         "통합 관문 미통과 — L4 잃음 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H129-맞교환(유지 ↔ L4)"),
        ("범주 규칙 사례 한정 → H129 부분", spec(r={b: ab(-2000, -900, 2000, 1500, C=(300, 200), D=(-300, -200)) for b in B5}), {}, -1500,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 부분 · 반전 보존 · 유지 보존", "H129 부분(유지는 섬 — 범주 규칙 부분)"),
        ("반전 부분 → 맞교환(유지 ↔ 반전)", spec(v={b: ab(-500, -300, 500, 300) for b in B5}), {}, -1500,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 부분 · 유지 보존", "맞교환(유지 ↔ 반전)"),
        ("범주 규칙 실패 → 유지 보류(기준 미달)", spec(r={b: ab(-300, -200, 300, 200, C=(0, 0), D=(0, 0)) for b in B5}), {}, -1500,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 잃음 · 반전 보존 · 유지 보류(기준 미달)", "H129 부분·보류"),
        ("T_F0 −2000(파일·로그 같음) → 통과", spec(), {}, -2000, pv, "H129 지지"),
        ("T_F0 +800(양수 — 파일·로그 같음) → 통과", spec(), {}, 800, pv, "H129 지지"),
        ("경계: L4 4/5 → 보존", spec(gw={**GOOD, 20: PART[20]}), {}, -1500, pv, "H129 지지"),
        ("경계: L4 3/5 → 부분", spec(gw={**GOOD, 19: PART[19], 20: PART[20]}), {}, -1500,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H129-맞교환(유지 ↔ L4)"),
    ]
    for nm, sp, md, tf, wv, wh in scen:
        v, h, _ = run(sp, mods=md, tf=tf)
        chk(nm, (v, h), (wv, wh))
        chk(nm + " — 독립 대조 같음", RUNV[-1][:2], (v, h))
    fails = [
        ("맥락 무게 20(GW 학습)", {("GW", 16): {"w_zz_ctx": 20.0}}, "F-G0 b16 맥락 흥분 무게 줄"),
        ("zz_ctx2 줄 없음(R 학습)", {("R", 17): {"no_zz_ctx2": True}}, "G0-R b17 맥락 흥분 무게 줄"),
        ("연결 수 다름(V_learn 평가)", {("V_learn", 18): {"cnt_zz_ctx": (1234, 1223)}}, "G0 b18 V_learn 맥락 흥분 무게 줄"),
        ("p 0.10(GW none Y 평가)", {("GWev", 19, "none", "Y"): {"p": 0.10}}, "F-G0 b19 none Y 맥락 흥분 무게 줄"),
        ("F 적재 줄 없음(P 학습)", {("P", 20): {"no_feat": True}}, "G0-P b20 F 적재 줄 []"),
        ("동작점 −1300(none 평가)", {("none", 16): {"tonic": -1300}}, "G0 b16 none 동작점 줄 [-1300.0]"),
        ("F-G0 가중치 = E206 F-GW", {("GW", 17): {"same_as_gw": True}}, "F-G0 b17 가중치("),
        ("평가 세계 28300", {("P_learn", 18): {"eval_world": 28300}}, "G0 b18 P_learn 평가 세계·원형"),
    ]
    VCATCH = {"맥락 무게 20(GW 학습)", "zz_ctx2 줄 없음(R 학습)", "연결 수 다름(V_learn 평가)", "p 0.10(GW none Y 평가)", "F 적재 줄 없음(P 학습)", "동작점 −1300(none 평가)", "평가 세계 28300"}
    for nm, md, kw in fails:
        v, _, out = run(spec(), mods=md)
        jline = ([ln for ln in out.splitlines() if ln.startswith("조작검증 실패: ")] or [""])[-1]
        chk("조작검증 실패: " + nm, (v, kw in jline), ("보류(조작검증 실패)", True))
        if nm in VCATCH:
            chk("조작검증 실패(독립): " + nm, RUNV[-1][0], "보류(조작검증 실패)")
    v, _, out = run(spec(), spec6=T6.spec(gw=GOOD))
    chk("같은 값(F-G0 learn = E206 F-GW learn) → 보류", (v, "같은 값(F-G0 learn 8값 = E206 F-GW) 뇌 [16, 17, 18, 19, 20]" in out), ("보류(조작검증 실패)", True))
    chk("같은 값 — 독립 대조도 보류", (RUNV[-1][0], "F-G0 b16 learn = E206 F-GW learn(같은 값)" in RUNV[-1][2]), ("보류(조작검증 실패)", True))
    v, _, out = run(spec())
    chk("통과 자료에 조작검증 실패 줄 없음(판정기·독립)", ("조작검증 실패" in out, "조작검증 실패" in RUNV[-1][2]), (False, False))
    chk("보고 줄 — E206 F-GW 짝과 G0 − GW", "(E206 F-GW +1900 · G0 − GW +100)" in out and "G0 − GW: R A" in out, True)
    v, _, _ = run(spec(), mods={"no_calib": True})
    chk("결측(calib_tonic.txt 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    v, _, _ = run(spec(), mods={("GW", 18): {"missing": True}})
    chk("결측(F-G0 b18 학습 로그 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    v, _, _ = run(spec(), mods={("e206_noctx", 19): True})
    chk("결측(E206 b19 맥락 연결 줄 없음 → 기대 연결 수 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
