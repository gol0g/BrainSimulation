#!/usr/bin/env python3
"""judge_e206·verify_e206_independent 합성 시험 — 가짜 형성·학습·평가 로그·가중치(F-GW + 앞 능력 R·V·P, 뇌 16~20)와 짝 E205 자료(test_judge_e205 생성기)로
판정·독립 대조가 사전 규칙(criteria_fixed.txt)과 같은지, 그리고 서로 같은지.
시나리오: 통과·H127-null·맞교환(부분·고착)·F 가 L4 를 깸(부분·잃음)·H127 부분(범주 규칙 부분)·유지 보류(기준 미달)·T_F −3100;
경계: J_post 1000/1001·L4 범주 4/5·3/5; 조작검증 실패: 형성(세계·제시·겹침·저장 파일)·F 적재(없음·다른 뇌·검증 일치 없음)·동작점(값·없음)·
같은 값(F = E205)·F-GW 가중치 = E205 GW·게이팅 비·맥락 켬·평가 세계; 결측(calib·로그).
실행: python3 scripts/test_judge_e206.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e206 as J  # noqa: E402
import verify_e206_independent as V  # noqa: E402
import test_judge_e205 as T5  # noqa: E402
import test_judge_e199 as T9  # noqa: E402

NK = 3000
B5 = J.BRAINS
PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")
WROOT = "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/research/experiments/traces/E206"


def f_lines(f, b, o, tf):
    """F 적재 줄 + 동작점 줄(옵션으로 빼거나 바꿈)."""
    if not o.get("no_feat"):
        fb = o.get("feat_brain", b)
        tail = "" if o.get("feat_noverify") else " 검증 일치"
        f.write("[E184 특징 가중치 적재] %s/feat_formed_b%d.npz%s — zz_feat_l_to_kc_l=1234.00 zz_feat_r_to_kc_r=1200.00\n" % (WROOT, fb, tail))
    if not o.get("no_tonic"):
        f.write("  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n" % o.get("tonic", tf))


def write_form(path, b, o):
    with open(path, "w", encoding="utf-8") as f:
        f.write("[원시 특징 세계] n 200 k 40 공유 10 왜곡 12 노출 사례 6 새 사례 6 seed %d | 원형 A %s,33,40,41,44 | 원형 B 1,2,3\n"
                % (o.get("form_world", 38300), o.get("form_proto", "5,14,15,19,29")))
        for tg in ("pre", "pre2", "post"):
            f.write("RCCHK tag=%s npres=288 feat_active_frac=0.2000 feat_inactive_spikes=0 feat_active_spikes=999 feat_other_side=0 goodbad=0 g_same=1 base_l=0 base_r=0\n" % tg)
        f.write("[원시 특징 노출] 제시 %d(주기 17 × 사례 6 × 범주 4 × 쪽 2), 노출 eta 0.02, Oja 켬, 도파민 호출 없음 — 노출 범주 A,B,C,D\n" % o.get("expo", 816))
        for sd in "lr":
            f.write("RCW side=%s nsyn=24000 g_mean0=24.0000 g_mean1=23.0000 dg_abs=1.2345 frac_changed=0.9000 wsel0=0.1 wsel1=0.2 kc_in_mean0=1.0 kc_in_mean1=1.1 g0_init=1\n" % sd)
        for sd in "lr":
            for p in PAIRS:
                jp = o.get("jpost", {}).get((sd, p), 300)
                f.write("RCJ side=%s pair=%s J_pre=0.3300 J_post=%s n_pre=132/130 n_post=86/80\n" % (sd, p, "%.4f" % (jp / 10000.0)))
        f.write("=> RAWCAT exposure done\n")


def save_w(E, name, b, seed, same_as=None):
    d6 = os.path.join(E, "traces/E206")
    os.makedirs(d6, exist_ok=True)
    if same_as:   # 경로(E205/…)의 가중치 그대로
        Z = np.load(os.path.join(E, "traces", same_as))
        np.savez(os.path.join(d6, "w_%s_b%d.npz" % (name, b)), **{k: Z[k] for k in Z.files})
        return
    d = np.random.RandomState(seed).uniform(-140, 140, NK)
    W = {}
    for kn in "lr":
        W["kc_%s_to_motor_l" % kn] = (150 + np.repeat(d, 2)).astype(np.float32)
        W["kc_%s_to_motor_r" % kn] = (150 - np.repeat(d, 2)).astype(np.float32)
    np.savez(os.path.join(d6, "w_%s_b%d.npz" % (name, b)), **W)


def build6(E, spec, mods, tf):
    L = os.path.join(E, "logs/E206")
    os.makedirs(L, exist_ok=True)
    if not mods.get("no_calib"):
        with open(os.path.join(L, "calib_tonic.txt"), "w", encoding="utf-8") as f:
            f.write("T_F=%d m_U=0.5974 m_F=0.5900 diff=0.0074 (규칙: |m_F − m_U| ≤ 0.0170 → −2800)\n" % tf)
    for b in B5:
        o = mods.get(("form", b), {})
        write_form(os.path.join(L, "form_b%d.log" % b), b, o)
        os.makedirs(os.path.join(E, "traces/E206"), exist_ok=True)
        if not o.get("no_ff"):
            np.savez(os.path.join(E, "traces/E206/feat_formed_b%d.npz" % b), zz_feat_l_to_kc_l=np.ones(4, np.float32))
        o = mods.get(("GW", b), {})
        if not o.get("missing"):
            tp = os.path.join(L, "train_GW_b%d.log" % b)
            T5.write_l4_train(tp, "GW", b, o)
            with open(tp, "a", encoding="utf-8") as f:
                f_lines(f, b, o, tf)
            save_w(E, "GW", b, 7000 * b + 1, same_as=("E205/w_GW_b%d.npz" % b) if o.get("same_as_u") else None)
            for x in "XY":
                for w, mm, pw in (("none", T9.NONE_M, 0), ("learn", T9.learn_mods(spec["GW"][b]), 8)):
                    ep = os.path.join(L, "ev_GW_b%d_%s_%s.log" % (b, w, x))
                    oe = mods.get(("GWev", b, w, x), {})
                    T5.write_l4_eval(ep, "GW", x, mm, pw, oe)
                    with open(ep, "a", encoding="utf-8") as f:
                        f_lines(f, b, oe, tf)
        for t in ("R", "V", "P"):
            o = mods.get((t, b), {})
            tp = os.path.join(L, "train_%s_b%d.log" % (t, b))
            T5.write_task_train(tp, t, b, o)
            with open(tp, "a", encoding="utf-8") as f:
                f_lines(f, b, o, tf)
        save_w(E, "R", b, 7000 * b + 3)
        save_w(E, "V", b, 7000 * b + 4)
        save_w(E, "P", b, 7000 * b + 5)
        for nm, t, cats, pw in J.EVALS:
            o = mods.get((nm, b), {})
            ep = os.path.join(L, "ev_b%d_%s.log" % (b, nm))
            T5.write_task_eval(ep, cats, spec[t][b] if t else None, pw, o)
            with open(ep, "a", encoding="utf-8") as f:
                f_lines(f, b, o, tf)


def run(spec6, spec5=None, mods=None, tf=-2800):
    root = tempfile.mkdtemp(prefix="te206_")
    try:
        E = T5.build(root, spec5 or T5.spec(gw=U_GW), {})       # 짝 E205(U) — logs/E205·traces/E205·스냅숏
        build6(E, spec6, mods or {}, tf)
        J.EXP = E; V.EXP = E
        out = {}
        for nm, fn in (("J", J.main), ("V", V.main)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn()
            out[nm] = buf.getvalue()
        pick = lambda s, p: ([ln[len(p):] for ln in s.splitlines() if ln.startswith(p)] or ["?"])[-1]  # noqa: E731
        return pick(out["J"], "판정: "), pick(out["J"], "가설: "), pick(out["V"], "독립 판정: "), pick(out["V"], "독립 가설: "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


U_GW = {b: T5.av_all(2100, 1600) for b in B5}       # E205 U 짝 — F 와 값이 다름
GOOD = {b: T5.av_all(2000, 1500) for b in B5}
PART = T5.PART
FAIL = {b: T5.av_all(500, 300) for b in B5}


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
    pass_v = "통합 관문 통과(개발 비교)"
    scen = [
        ("통과(넷 다 보존)", spec(), {}, -2800, pass_v, "H127 지지"),
        ("유지 잃음 → H127-null", spec(p={b: T5.P_INTF for b in B5}), {}, -2800,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 보존 · 유지 잃음", "H127-null(유지 안 섬)"),
        ("반전 부분 → 맞교환", spec(v={b: ab(-500, -300, 500, 300) for b in B5}), {}, -2800,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 부분 · 유지 보존", "맞교환(유지 ↔ 반전)"),
        ("반전 고착 → 반전 잃음·맞교환", spec(v={b: ab(-1800, -1300, 1900, 1400) for b in B5}), {}, -2800,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 보존 · 반전 잃음 · 유지 보존", "맞교환(유지 ↔ 반전)"),
        ("F-GW 부분 → F 가 L4 를 깸", spec(gw=PART), {}, -2800,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "F 가 L4 를 깸"),
        ("F-GW 실패 → L4 잃음", spec(gw=FAIL), {}, -2800,
         "통합 관문 미통과 — L4 잃음 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "F 가 L4 를 깸"),
        ("범주 규칙 사례 한정 5/5 → H127 부분", spec(r={b: ab(-2000, -900, 2000, 1500, C=(300, 200), D=(-300, -200)) for b in B5}), {}, -2800,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 부분 · 반전 보존 · 유지 보존", "H127 부분(유지는 섬 — 범주 규칙 부분)"),
        ("범주 규칙 실패 → 유지 보류(기준 미달)", spec(r={b: ab(-300, -200, 300, 200, C=(0, 0), D=(0, 0)) for b in B5}), {}, -2800,
         "통합 관문 미통과 — L4 보존 · 범주 규칙 잃음 · 반전 보존 · 유지 보류(기준 미달)", "H127 부분·보류"),
        ("T_F −3100(파일·로그 같음) → 통과", spec(), {}, -3100, pass_v, "H127 지지"),
        ("경계: J_post 정확히 0.1000 → 통과", spec(), {("form", 18): {"jpost": {("r", "BD"): 1000}}}, -2800, pass_v, "H127 지지"),
        ("경계: F-GW 범주 4/5 → L4 보존", spec(gw={**GOOD, 20: PART[20]}), {}, -2800, pass_v, "H127 지지"),
        ("경계: F-GW 범주 3/5 → L4 부분", spec(gw={**GOOD, 19: PART[19], 20: PART[20]}), {}, -2800,
         "통합 관문 미통과 — L4 부분 · 범주 규칙 보존 · 반전 보존 · 유지 보존", "F 가 L4 를 깸"),
    ]
    for nm, sp, md, tf, wv, wh in scen:
        v, h, iv, ih, _ = run(sp, mods=md, tf=tf)
        chk(nm, (v, h), (wv, wh))
        chk(nm + " — 독립 대조 같음", (iv, ih), (v, h))
    # 조작검증 실패 — (이름, mods, 독립 대조도 잡는가)
    # (이름, mods, 독립 대조도 잡는가, 판정기 실패 사유에 있어야 할 말, 독립 대조 사유에 있어야 할 말)
    fails = [
        ("형성 세계 18300", {("form", 16): {"form_world": 18300}}, True, "형성 b16 세계 줄", "형성 b16"),
        ("형성 원형 A 다름", {("form", 17): {"form_proto": "1,2,3,4,5"}}, True, "형성 b17 세계 줄", "형성 b17"),
        ("형성 제시 815", {("form", 18): {"expo": 815}}, True, "형성 b18 E191 형성 검사", "형성 b18"),
        ("겹침 J_post 0.1001(AC 좌)", {("form", 19): {"jpost": {("l", "AC"): 1001}}}, True, "형성 b19 겹침 J_post", "형성 b19"),
        ("형성 저장 파일 없음", {("form", 20): {"no_ff": True}}, False, "형성 b20 저장 파일 없음", None),
        ("F 적재 줄 없음(GW 학습)", {("GW", 16): {"no_feat": True}}, True, "F-GW b16 F 적재 줄 []", "F-GW b16 학습"),
        ("F 적재 줄이 다른 뇌 파일(R 학습)", {("R", 17): {"feat_brain": 18}}, True, "F-R b17 F 적재 줄", "F-R b17 학습"),
        ("F 적재 '검증 일치' 없음(평가)", {("V_learn", 18): {"feat_noverify": True}}, True, "F b18 V_learn F 적재 줄 [None]", "F b18 V_learn 평가"),
        ("F 적재 줄 없음(GW none 평가)", {("GWev", 19, "none", "Y"): {"no_feat": True}}, True, "F-GW b19 none Y F 적재 줄 []", "F-GW b19 none Y 평가"),
        ("동작점 −4900(P 학습)", {("P", 20): {"tonic": -4900}}, True, "F-P b20 동작점 줄 [-4900.0]", "F-P b20 학습"),
        ("동작점 줄 없음(none 평가)", {("none", 16): {"no_tonic": True}}, True, "F b16 none 동작점 줄 []", "F b16 none 평가"),
        ("F-GW 가중치 = E205 GW", {("GW", 17): {"same_as_u": True}}, False, "F-GW b17 가중치(", None),
        ("R 게이팅 비 0.011", {("R", 18): {"ratio": 0.011}}, True, "F-R b18 게이팅(", "F-R b18 학습"),
        ("V 맥락 켬 2999", {("V", 19): {"ncnt": 2999}}, True, "F-V b19 맥락 고정(", "F-V b19 학습"),
        ("평가 세계 28300", {("P_learn", 20): {"eval_world": 28300}}, True, "F b20 P_learn 평가 세계·원형", "F b20 P_learn 평가"),
    ]
    for nm, md, vcatch, kw, vkw in fails:
        v, _, iv, _, out = run(spec(), mods=md)
        jline = ([ln for ln in out["J"].splitlines() if ln.startswith("조작검증 실패: ")] or [""])[-1]
        vline = ([ln for ln in out["V"].splitlines() if ln.startswith("독립 조작검증 실패: ")] or [""])[-1]
        chk("조작검증 실패: " + nm, (v, kw in jline), ("보류(조작검증 실패)", True))
        if vcatch:
            chk("조작검증 실패(독립): " + nm, (iv, vkw in vline), ("보류(조작검증 실패)", True))
    v, _, iv, _, out = run(spec(), spec5=T5.spec(gw=GOOD))
    chk("같은 값(F-GW learn = E205 GW learn) → 보류", (v, "같은 값(F-GW learn 8값 = E205 GW) 뇌 [16, 17, 18, 19, 20]" in out["J"]), ("보류(조작검증 실패)", True))
    chk("같은 값 — 독립 대조도 보류", (iv, "F-GW b16 learn = E205 GW learn(같은 값)" in out["V"]), ("보류(조작검증 실패)", True))
    v, _, iv, _, out = run(spec(), mods={})
    chk("통과 자료에 조작검증 실패 줄 없음(판정기·독립)", ("조작검증 실패" in out["J"], "조작검증 실패" in out["V"]), (False, False))
    v, _, iv, _, _ = run(spec(), mods={"no_calib": True})
    chk("결측(calib_tonic.txt 없음)", (v.startswith("보류(결측"), iv.startswith("보류(결측")), (True, True))
    v, _, iv, _, _ = run(spec(), mods={("GW", 18): {"missing": True}})
    chk("결측(F-GW b18 학습 로그 없음)", (v.startswith("보류(결측"), iv.startswith("보류(결측")), (True, True))
    _, _, _, _, out = run(spec())
    chk("보고 줄 — E205 U 짝과 F − U", "(E205 U +2100 · F − U -100)" in out["J"] and "F − U: R A" in out["J"], True)
    chk("보고 줄 — 형성 J·반응 KC", "형성 J_pre/J_post AC 좌 3300/300" in out["J"] and "n_pre 132/130 → n_post 86/80" in out["J"], True)
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
