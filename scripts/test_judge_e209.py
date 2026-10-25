#!/usr/bin/env python3
"""judge_e209 합성 시험 — 짝 E205(test_judge_e205)·E206(test_judge_e206 build6)·E208(test_judge_e208 build8) 자료 위에 가짜 E209 형성·학습·평가 로그·가중치
(F_R 형성 + F-G0R + R·V·P, 뇌 16~20)를 만들어 판정기와 독립 대조가 사전 규칙과 같은지.
시나리오: 통과·H130-null·H130-맞교환(L4 부분·실패)·H130 부분(범주 규칙)·맞교환(반전)·보류(유지 기준 미달)·T_C +2900·−400; 경계: L4 4/5·3/5·게이팅 요약 비 0.10(통과)·0.11(실패);
조작검증 실패: 재배선 구성 줄 없음·장치 줄 없음·옮김 수 다름·장치 연결 수 다름·연결 절반 판 가중치(10/−30)·재배선 꼬리 없음·학습 로그 구성 줄 1개·형성 J_post 0.1100·형성 로그 재배선 줄 없음·
  F 경로가 옛 F(E206)·맥락 무게 20·동작점 다름·가중치 = E208·평가 세계 28300; 같은 값(G0R = E208 G0); 결측(calib·학습 로그·형성 로그·E206 맥락 줄).
스냅숏: 가짜 스냅숏에 KC ↔ 억제 4집단(작은 무작위 연결)을 더하고, 기대 '옮김 k/n' 은 모델 코드(ctx_gate.build_all + 19900 순열)로 계산 — 독립 대조기의 다시 만든 무리와 맞는지도 함께 시험.
실행: python3 scripts/test_judge_e209.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend", "genesis"))
import ctx_gate as CG  # noqa: E402 — 모델 코드(기대값 계산용 — 독립 대조기와 다른 구현)
import judge_e209 as J  # noqa: E402
import verify_e209_independent as V  # noqa: E402
import test_judge_e205 as T5  # noqa: E402
import test_judge_e206 as T6  # noqa: E402
import test_judge_e199 as T9  # noqa: E402
import test_judge_e208 as T8  # noqa: E402

B5 = J.BRAINS
NK = 3000
GROUPS = ("kc_l_to_inh_l", "kc_inh_l_to_kc_l", "kc_r_to_inh_r", "kc_inh_r_to_kc_r")
WROOT9 = "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/research/experiments/traces/E209"
PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")


def add_inh_snap(E, b):
    """가짜 스냅숏(T5 가 만든 KC→motor 집단)에 KC ↔ 억제 4집단을 더함 → 기대 {집단: (옮김, 전체)}(모델 코드로 무리 계산)."""
    path = os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b)
    Z = dict(np.load(path))
    rs = np.random.RandomState(500 + b)
    gs = 12345 + b
    gm = CG.build_all(19300 + gs, NK, 200, 200, 0.5, 0.10)
    ri = np.random.RandomState(19900 + gs)
    ih = {}
    for sd in "lr":
        m = np.zeros(400, bool); m[ri.permutation(400)[:200]] = True; ih[sd] = m
    exp = {}
    for sd in "lr":
        isx = gm[sd][0]
        for nm, npre, npost, pm, qm in (("kc_%s_to_inh_%s" % (sd, sd), NK, 400, isx, ih[sd]), ("kc_inh_%s_to_kc_%s" % (sd, sd), 400, NK, ih[sd], isx)):
            pre, post = np.nonzero(rs.random_sample((npre, npost)) < 0.004)
            Z[nm + "__pre"] = pre.astype(np.uint32); Z[nm + "__post"] = post.astype(np.uint32)
            exp[nm] = (int(np.sum(pm[pre] != qm[post])), int(pre.size))
    np.savez(path, **Z)
    return exp


def comp_lines(f, exp, o, nb):
    """재배선 구성·장치 확인 줄 nb 쌍(옵션: 없앰·값 바꿈)."""
    if o.get("no_compart"):
        return
    mv = o.get("mv", exp)
    if mv == "SHIFT":   # 옮김 수 하나 다름
        mv = {g: (exp[g][0] + (1 if g == "kc_inh_r_to_kc_r" else 0), exp[g][1]) for g in GROUPS}
    wki, wik = o.get("cw", (5.0, -15.0))
    for _ in range(o.get("nb", nb)):
        f.write("  [E209 구획 되먹임 억제 — 재배선] KC 억제 뉴런 무리 X 좌 %d·우 200 / 400, 무리 밖 연결의 앞 뉴런을 같은 무리 안으로(뒤 뉴런마다 받는 연결 수 그대로) — %s, 가중치 그대로 KC→억제 %.2f·억제→KC %.2f\n"
                % (o.get("nihx", 200), " · ".join("%s 옮김 %d/%d" % (g, mv[g][0], mv[g][1]) for g in GROUPS), wki, wik))
        if not o.get("no_dev"):
            dv = o.get("dev", {g: mv[g][1] for g in GROUPS})
            if dv == "SHIFT":   # 장치 연결 수 하나 다름
                dv = {g: mv[g][1] + (1 if g == "kc_l_to_inh_l" else 0) for g in GROUPS}
            f.write("  [E209 구획 되먹임 억제 확인] 장치 — " + " · ".join("%s %d(같음·무리 밖 0·g %.2f)" % (g, dv[g], wki if "_to_inh_" in g else wik) for g in GROUPS)
                    + ("" if o.get("no_tail") else " · 재배선 — 뒤 뉴런마다 받는 연결 수 = 스냅숏·중복 없음") + "\n")


def f9_lines(f, b, o, tc):
    """F_R 적재 줄(E209 경로 — 옵션으로 옛 F 경로) + 동작점 줄."""
    if not o.get("no_feat"):
        root = T6.WROOT if o.get("old_f") else WROOT9
        f.write("[E184 특징 가중치 적재] %s/feat_formed_b%d.npz 검증 일치 — zz_feat_l_to_kc_l=1234.00 zz_feat_r_to_kc_r=1200.00\n" % (root, b))
    if not o.get("no_tonic"):
        f.write("  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n" % o.get("tonic", tc))


def write_form9(path, b, o, exp):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[원시 특징 세계] n 200 k 40 공유 10 왜곡 12 노출 사례 6 새 사례 6 seed %d | 원형 A 5,14,15,19,29,33,40,41,44 | 원형 B 1,2,3\n" % o.get("form_world", 38300))
        for tg in ("pre", "pre2", "post"):
            f.write("RCCHK tag=%s npres=288 feat_active_frac=0.2000 feat_inactive_spikes=0 feat_active_spikes=999 feat_other_side=0 goodbad=0 g_same=1 base_l=0 base_r=0\n" % tg)
        f.write("[원시 특징 노출] 제시 816(주기 17 × 사례 6 × 범주 4 × 쪽 2), 노출 eta 0.02, Oja 켬, 도파민 호출 없음 — 노출 범주 A,B,C,D\n")
        for sd in "lr":
            f.write("RCW side=%s nsyn=24000 g_mean0=24.0000 g_mean1=23.0000 dg_abs=1.2345 frac_changed=0.9000 wsel0=0.1 wsel1=0.2 kc_in_mean0=1.0 kc_in_mean1=1.1 g0_init=1\n" % sd)
        for sd in "lr":
            for p in PAIRS:
                jp = o.get("jpost", {}).get((sd, p), 400)
                f.write("RCJ side=%s pair=%s J_pre=0.4500 J_post=%s n_pre=180/170 n_post=70/66\n" % (sd, p, "%.4f" % (jp / 10000.0)))
        comp_lines(f, exp, o, 2)
        f.write("=> RAWCAT exposure done\n")


def save_w9(E, name, b, seed, same_as=None):
    d9 = os.path.join(E, "traces/E209")
    os.makedirs(d9, exist_ok=True)
    if same_as:
        Z = np.load(os.path.join(E, "traces", same_as))
        np.savez(os.path.join(d9, "w_%s_b%d.npz" % (name, b)), **{k: Z[k] for k in Z.files})
        return
    d = np.random.RandomState(seed).uniform(-140, 140, NK)
    W = {}
    for kn in "lr":
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    np.savez(os.path.join(d9, "w_%s_b%d.npz" % (name, b)), **W)


def build9(E, spec, mods, tc, EXP_):
    L = os.path.join(E, "logs/E209")
    os.makedirs(L, exist_ok=True)
    if not mods.get("no_calib"):
        with open(os.path.join(L, "calib_tonic.txt"), "w", encoding="utf-8") as f:
            f.write("T_C=%d m_ref=0.6181 m_C=0.6150 diff=0.0031\n" % tc)
    for b in B5:
        exp = EXP_[b]
        o = mods.get(("form", b), {})
        if not o.get("missing"):
            write_form9(os.path.join(L, "form_b%d.log" % b), b, o, exp)
        os.makedirs(os.path.join(E, "traces/E209"), exist_ok=True)
        np.savez(os.path.join(E, "traces/E209/feat_formed_b%d.npz" % b), zz_feat_l_to_kc_l=np.ones(4, np.float32))
        o = mods.get(("GC", b), {})
        if not o.get("missing"):
            tp = os.path.join(L, "train_GC_b%d.log" % b)
            T5.write_l4_train(tp, "GW", b, o)
            with open(tp, "a", encoding="utf-8") as f:
                f9_lines(f, b, o, tc); T8.ctx_lines(f, o, 0.0); comp_lines(f, exp, o, 2)
            save_w9(E, "GC", b, 11000 * b + 1, same_as=("E208/w_GW_b%d.npz" % b) if o.get("same_as_g0") else None)
            for x in "XY":
                for w, mm, pw in (("none", T9.NONE_M, 0), ("learn", T9.learn_mods(spec["GC"][b]), 8)):
                    ep = os.path.join(L, "ev_GC_b%d_%s_%s.log" % (b, w, x))
                    oe = mods.get(("GCev", b, w, x), {})
                    T5.write_l4_eval(ep, "GW", x, mm, pw, oe)
                    with open(ep, "a", encoding="utf-8") as f:
                        f9_lines(f, b, oe, tc); T8.ctx_lines(f, oe, 0.0); comp_lines(f, exp, oe, 1)
        for t in ("R", "V", "P"):
            o = mods.get((t, b), {})
            tp = os.path.join(L, "train_%s_b%d.log" % (t, b))
            T5.write_task_train(tp, t, b, o)
            with open(tp, "a", encoding="utf-8") as f:
                f9_lines(f, b, o, tc); T8.ctx_lines(f, o, 0.0); comp_lines(f, exp, o, 2)
        save_w9(E, "R", b, 11000 * b + 3)
        save_w9(E, "V", b, 11000 * b + 4)
        save_w9(E, "P", b, 11000 * b + 5)
        for nm, t, cats, pw in J.EVALS:
            o = mods.get((nm, b), {})
            ep = os.path.join(L, "ev_b%d_%s.log" % (b, nm))
            T5.write_task_eval(ep, cats, spec[t][b] if t else None, pw, o)
            with open(ep, "a", encoding="utf-8") as f:
                f9_lines(f, b, o, tc); T8.ctx_lines(f, o, 0.0); comp_lines(f, exp, o, 1)


RUNV = []   # run() 마다 독립 대조 (판정, 가설, 출력)
G08 = {b: T5.av_all(1950, 1450) for b in B5}       # E208 F-G0 짝 — G0R 와 값이 다름
GOOD = T6.GOOD
PART = T6.PART
FAIL = T6.FAIL


def spec(gc=GOOD, r=None, v=None, p=None):
    return {"GC": gc, "R": r or {b: T5.R_OK for b in B5}, "V": v or {b: T5.V_OK for b in B5}, "P": p or {b: T5.P_KEEP for b in B5}}


def run(spec9, spec8=None, mods=None, tc=1400):
    root = tempfile.mkdtemp(prefix="te209_")
    mods = mods or {}
    try:
        E = T5.build(root, T5.spec(gw=T6.U_GW), {})                  # E205 U(E206 짝 경로가 쓰는 자료)
        T6.build6(E, T6.spec(gw=T8.GW6), {}, -1300)                  # E206 F-GW(E208 짝 경로가 쓰는 자료)
        for b in B5:                                                  # E206 학습 로그에 맥락 무게 20 줄(연결 수 = 기대)
            if not mods.get(("e206_noctx", b)):
                with open(os.path.join(E, "logs/E206/train_GW_b%d.log" % b), "a", encoding="utf-8") as f:
                    T8.ctx_lines(f, {}, 20.0)
        T8.build8(E, spec8 or T8.spec(gw=G08), {}, 1100)              # E208 F-G0 짝
        EXP_ = {b: add_inh_snap(E, b) for b in B5}                   # 스냅숏에 억제 4집단(모든 생성 뒤)
        build9(E, spec9, mods, tc, EXP_)
        J.EXP = E; V.EXP = E
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
        out = buf.getvalue()
        vb = io.StringIO()
        with contextlib.redirect_stdout(vb):
            V.main()
        vout = vb.getvalue()
        vexp = {b: V.expected_compart(b) for b in B5}
        pick = lambda s, p: ([ln[len(p):] for ln in s.splitlines() if ln.startswith(p)] or ["?"])[-1]  # noqa: E731
        RUNV.append((pick(vout, "독립 판정: "), pick(vout, "독립 가설: "), vout, vexp == EXP_))
        return pick(out, "판정: "), pick(out, "가설: "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
        ("통과(넷 다 보존)", spec(), {}, 1400, pv, "H130 지지"),
        ("유지 잃음 → H130-null", spec(p={b: T5.P_INTF for b in B5}), {}, 1400,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 보존 · 유지 잃음", "H130-null(유지 안 섬)"),
        ("L4 부분·유지 보존 → H130-맞교환", spec(gc=PART), {}, 1400,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H130-맞교환(유지 ↔ L4)"),
        ("L4 잃음·유지 보존 → H130-맞교환", spec(gc=FAIL), {}, 1400,
         "통합 관문 미통과 — L4 잃음 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H130-맞교환(유지 ↔ L4)"),
        ("범주 규칙 사례 한정 → H130 부분", spec(r={b: ab(-2000, -900, 2000, 1500, C=(300, 200), D=(-300, -200)) for b in B5}), {}, 1400,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 부분 · 반전 보존 · 유지 보존", "H130 부분(유지는 섬 — 범주 규칙 부분)"),
        ("반전 부분 → 맞교환(유지 ↔ 반전)", spec(v={b: ab(-500, -300, 500, 300) for b in B5}), {}, 1400,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 부분 · 유지 보존", "맞교환(유지 ↔ 반전)"),
        ("범주 규칙 실패 → 유지 보류(기준 미달)", spec(r={b: ab(-300, -200, 300, 200, C=(0, 0), D=(0, 0)) for b in B5}), {}, 1400,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 잃음 · 반전 보존 · 유지 보류(기준 미달)", "H130 부분·보류"),
        ("T_C +2900(파일·로그 같음) → 통과", spec(), {}, 2900, pv, "H130 지지"),
        ("T_C −400(음수 — 파일·로그 같음) → 통과", spec(), {}, -400, pv, "H130 지지"),
        ("경계: L4 4/5 → 보존", spec(gc={**GOOD, 20: PART[20]}), {}, 1400, pv, "H130 지지"),
        ("경계: L4 3/5 → 부분", spec(gc={**GOOD, 19: PART[19], 20: PART[20]}), {}, 1400,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "H130-맞교환(유지 ↔ L4)"),
        ("경계: 게이팅 요약 비 0.1000(R 학습) → 통과", spec(), {("R", 17): {"ratio": 0.1000}}, 1400, pv, "H130 지지"),
    ]
    for nm, sp, md, tc, wv, wh in scen:
        v, h, _ = run(sp, mods=md, tc=tc)
        chk(nm, (v, h), (wv, wh))
        chk(nm + " — 독립 대조 같음", RUNV[-1][:2], (v, h))
    chk("독립 대조기의 다시 만든 무리 = 모델 코드(ctx_gate) 기대값(옮김 k/n, 5뇌)", RUNV[-1][3], True)
    E0 = None
    fails = [
        ("재배선 구성 줄 없음(GC 학습)", {("GC", 16): {"no_compart": True}}, "F-G0R b16 구획 억제 줄"),
        ("장치 줄 없음(R 학습)", {("R", 17): {"no_dev": True}}, "G0R-R b17 구획 억제 줄"),
        ("옮김 수 다름(V_learn 평가)", {("V_learn", 18): {"mv": "SHIFT"}}, "G0R b18 V_learn 구획 억제 줄"),
        ("장치 연결 수 다름(GC none Y 평가)", {("GCev", 19, "none", "Y"): {"dev": "SHIFT"}}, "F-G0R b19 none Y 구획 억제 줄"),
        ("연결 절반 판 가중치 10/−30(P 학습)", {("P", 20): {"cw": (10.0, -30.0)}}, "G0R-P b20 구획 억제 줄"),
        ("재배선 꼬리 없음(none 평가)", {("none", 16): {"no_tail": True}}, "G0R b16 none 구획 억제 줄"),
        ("학습 로그 구성 줄 1개(GC 학습)", {("GC", 17): {"nb": 1}}, "F-G0R b17 구획 억제 줄"),
        ("억제 뉴런 무리 X 199(R 학습)", {("R", 18): {"nihx": 199}}, "G0R-R b18 구획 억제 줄"),
        ("형성 J_post 0.1100(형성 b18)", {("form", 18): {"jpost": {("l", "AC"): 1100}}}, "형성 b18 겹침 J_post"),
        ("형성 로그 재배선 줄 없음(형성 b19)", {("form", 19): {"no_compart": True}}, "b19 구획 기준(형성 로그) 읽기 실패"),
        ("F 경로가 옛 F(E206 — GC 학습)", {("GC", 20): {"old_f": True}}, "F-G0R b20 F_R 적재 줄"),
        ("맥락 무게 20(V 학습)", {("V", 16): {"w_zz_ctx": 20.0}}, "G0R-V b16 맥락 흥분 무게 줄"),
        ("동작점 +1100(none 평가 — T_C 1400)", {("none", 17): {"tonic": 1100}}, "G0R b17 none 동작점 줄 [1100.0]"),
        ("F-G0R 가중치 = E208 F-G0", {("GC", 18): {"same_as_g0": True}}, "F-G0R b18 가중치("),
        ("평가 세계 28300(P_learn)", {("P_learn", 19): {"eval_world": 28300}}, "G0R b19 P_learn 평가 세계·원형"),
        ("게이팅 요약 비 0.1100(P 학습)", {("P", 20): {"ratio": 0.1100}}, "G0R-P b20 게이팅"),
    ]
    VCATCH = {"재배선 구성 줄 없음(GC 학습)", "장치 줄 없음(R 학습)", "옮김 수 다름(V_learn 평가)", "장치 연결 수 다름(GC none Y 평가)", "연결 절반 판 가중치 10/−30(P 학습)",
              "재배선 꼬리 없음(none 평가)", "학습 로그 구성 줄 1개(GC 학습)", "억제 뉴런 무리 X 199(R 학습)", "형성 J_post 0.1100(형성 b18)", "형성 로그 재배선 줄 없음(형성 b19)",
              "F 경로가 옛 F(E206 — GC 학습)", "맥락 무게 20(V 학습)", "동작점 +1100(none 평가 — T_C 1400)", "평가 세계 28300(P_learn)", "게이팅 요약 비 0.1100(P 학습)"}
    for nm, md, kw in fails:
        v, _, out = run(spec(), mods=md)
        jline = ([ln for ln in out.splitlines() if ln.startswith("조작검증 실패: ")] or [""])[-1]
        chk("조작검증 실패: " + nm, (v, kw in jline), ("보류(조작검증 실패)", True))
        if nm in VCATCH:
            chk("조작검증 실패(독립): " + nm, RUNV[-1][0], "보류(조작검증 실패)")
    v, _, out = run(spec(gc=G08), spec8=T8.spec(gw=G08))
    chk("같은 값(F-G0R learn = E208 F-G0 learn) → 보류", (v, "같은 값(F-G0R learn 8값 = E208 F-G0) 뇌 [16, 17, 18, 19, 20]" in out), ("보류(조작검증 실패)", True))
    chk("같은 값 — 독립 대조도 보류", (RUNV[-1][0], "F-G0R b16 learn = E208 F-G0 learn(같은 값)" in RUNV[-1][2]), ("보류(조작검증 실패)", True))
    v, _, out = run(spec())
    chk("통과 자료에 조작검증 실패 줄 없음(판정기·독립)", ("조작검증 실패" in out, "조작검증 실패" in RUNV[-1][2]), (False, False))
    chk("보고 줄 — E208 F-G0 짝과 G0R − G0·형성 줄", "(E208 F-G0 +1950 · G0R − G0 +50)" in out and "G0R − G0: R A" in out and "형성 b16 F_R 겹침" in out, True)
    v, _, _ = run(spec(), mods={"no_calib": True})
    chk("결측(calib_tonic.txt 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    v, _, _ = run(spec(), mods={("GC", 18): {"missing": True}})
    chk("결측(F-G0R b18 학습 로그 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    v, _, _ = run(spec(), mods={("form", 17): {"missing": True}})
    chk("결측(형성 b17 로그 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    v, _, _ = run(spec(), mods={("e206_noctx", 19): True})
    chk("결측(E206 b19 맥락 연결 줄 없음 → 기대 연결 수 없음)", (v.startswith("보류(결측"), RUNV[-1][0].startswith("보류(결측")), (True, True))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
