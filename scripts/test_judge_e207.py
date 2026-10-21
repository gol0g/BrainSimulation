#!/usr/bin/env python3
"""judge_e207·verify_e207_independent 합성 시험 — 가짜 측정 로그(U·F × 맥락 없음·X, 뇌 16~20)로 판정·독립 대조가 사전 규칙과 같은지·서로 같은지.
시나리오: H128 지지·H128-null·보류, 경계(S_X 6000·차 4000 정확히·5996·차 3996·S_X 3200 정확히), 4/5·3/5;
정정 1: 첫 측정(J_pre)·둘째 측정(J_post) 분류가 다르면 가운데(한 뇌·두 뇌·둘째만 null·둘째만 경계), J_pre ≠ J_post 자체는 실패 아님(기본 합성 로그는 S 밖 쌍에서 늘 다름);
조작검증 실패: 맥락 줄 없음·끈 맥락 발화·게이팅 비활성 비율·맥락 없음 측정에 맥락 줄·노출 816·F 적재 없음·U 에 F 적재·동작점·세계·g_same 0·같은 값(J_pre·J_post 각각); 결측.
실행: python3 scripts/test_judge_e207.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e207 as J  # noqa: E402
import verify_e207_independent as V  # noqa: E402

B5 = J.BRAINS
PAIRS = J.PAIRS


def write_log(path, b, rep, cond, jvals, o, jpost=None):
    """jvals[(side, pair)] = 첫 측정 J_pre(정수 1e-4), jpost = 둘째 측정 J_post(없으면 jvals). o: 실패 옵션."""
    jpost = jpost or jvals
    with open(path, "w", encoding="utf-8") as f:
        f.write("[원시 특징 세계] n 200 k 40 공유 10 왜곡 12 노출 사례 6 새 사례 6 seed %d | 원형 A %s,33,40 | 원형 B 1,2,3\n" % (o.get("world", 38300), J.PROTO_A))
        if rep == "F" and not o.get("no_feat"):
            f.write("[E184 특징 가중치 적재] /mnt/c/x/research/experiments/traces/E206/feat_formed_b%d.npz 검증 일치 — a=1\n" % b)
        if rep == "U" and o.get("feat_in_u"):
            f.write("[E184 특징 가중치 적재] /mnt/c/x/research/experiments/traces/E206/feat_formed_b%d.npz 검증 일치 — a=1\n" % b)
        f.write("  [motor 동작점] motor 좌·우 Ioffset 기저값 %.1f(긴장성 억제)\n" % o.get("tonic", J.TONIC[rep]))
        for tg in ("pre", "post"):
            f.write("RCCHK tag=%s npres=288 feat_active_frac=0.2000 feat_inactive_spikes=0 feat_active_spikes=999 feat_other_side=0 goodbad=0 g_same=%d base_l=0 base_r=0\n" % (tg, o.get("g_same", 1)))
            if cond == "X" and not o.get("no_ctx"):
                f.write("[E207 측정 맥락] tag=%s 맥락 X 켬 — 제시 288 · 켠 맥락 발화 5000 · 끈 맥락 발화 %d · KC 발화 활성 무리 9000 · 비활성 무리 %d · 게이팅 있음\n"
                        % (tg, o.get("ctx_off", 0), o.get("inactive", 10)))
            if cond == "none" and o.get("ctx_in_none"):
                f.write("[E207 측정 맥락] tag=%s 맥락 X 켬 — 제시 288 · 켠 맥락 발화 5000 · 끈 맥락 발화 0 · KC 발화 활성 무리 9000 · 비활성 무리 10 · 게이팅 있음\n" % tg)
        f.write("[원시 특징 노출] 제시 %d(주기 0 × 사례 6 × 범주 4 × 쪽 2), 노출 eta 0, Oja 끔(정적), 도파민 호출 없음 — 노출 범주 A,B,C,D\n" % o.get("expo", 0))
        for sd in "lr":
            for p in PAIRS:
                v0, v1 = jvals[(sd, p)], jpost[(sd, p)]
                f.write("RCJ side=%s pair=%s J_pre=%d.%04d J_post=%d.%04d n_pre=100/100 n_post=80/80\n" % (sd, p, v0 // 10000, v0 % 10000, v1 // 10000, v1 % 10000))
        f.write("=> RAWCAT done\n")


def jv(ac_bd, ab=500, other=300):
    d = {}
    for sd in "lr":
        for p in PAIRS:
            d[(sd, p)] = ac_bd if p in ("AC", "BD") else (ab if p == "AB" else other)
    return d


def build(root, fx, fn, ux=4000, un=3000, mods=None, fxp=None, fnp=None):
    """fx[b]·fn[b] = F 의 X·none 첫 측정 J(AC=BD) 값, fxp·fnp = 둘째 측정(없으면 첫 측정과 같은 S), U 는 고정. mods[(rep, cond, b)] = 옵션.
    둘째 측정은 S 밖 쌍(AD·BC·CD)에서 늘 첫 측정과 다르게 쓴다(정정 1 — J_pre ≠ J_post 는 실패가 아님을 시험)."""
    mods = mods or {}
    fxp = fxp or fx; fnp = fnp or fn
    E = os.path.join(root, "research/experiments")
    L = os.path.join(E, "logs/E207")
    os.makedirs(L, exist_ok=True)
    for b in B5:
        for rep in J.REPS:
            for cond in J.CONDS:
                o = mods.get((rep, cond, b), {})
                if o.get("missing"):
                    continue
                v = (fx[b] if cond == "X" else fn[b]) if rep == "F" else (ux if cond == "X" else un)
                v1 = (fxp[b] if cond == "X" else fnp[b]) if rep == "F" else (ux if cond == "X" else un)
                vals = jv(v)
                vals1 = jv(v1, other=300 + (37 if cond == "X" else 11))
                if o.get("same_as_none"):
                    vals = jv(fn[b] if rep == "F" else un)
                if o.get("same_as_none_post"):
                    vals1 = jv(fnp[b] if rep == "F" else un, other=311)
                write_log(os.path.join(L, "meas_%s_%s_b%d.log" % (rep, cond, b)), b, rep, cond, vals, o, vals1)
    return E


def run(fx, fn, mods=None, fxp=None, fnp=None):
    root = tempfile.mkdtemp(prefix="te207_")
    try:
        E = build(root, fx, fn, mods=mods, fxp=fxp, fnp=fnp)
        J.EXP = E; V.EXP = E
        out = {}
        for nm, fn_ in (("J", J.main), ("V", V.main)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn_()
            out[nm] = buf.getvalue()
        pick = lambda s, p: ([ln[len(p):] for ln in s.splitlines() if ln.startswith(p)] or ["?"])[-1]  # noqa: E731
        return pick(out["J"], "판정: "), pick(out["V"], "독립 판정: "), out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def const(v):
    return {b: v for b in B5}


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = (got == want) if not callable(want) else want(got)
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    H = "H128 지지 — 맥락 X 가 형성 표현의 범주 KC 겹침을 되살림"
    N = "H128-null — 맥락 X 에서도 형성 표현의 범주 KC 겹침 낮음"
    scen = [
        ("H128 지지(F X 0.20·none 0.03)", const(2000), const(300), H),
        ("H128-null(F X 0.05)", const(500), const(300), N),
        ("보류(F X 0.12 — 가운데)", const(1200), const(300), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("경계: S_X 6000·차 4000 정확히 → 되살림", const(1500), const(500), H),
        ("경계: S_X 5996 → 가운데", const(1499), const(300), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("경계: 차 3996 → 가운데", const(1500), const(501), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("경계: S_X 3200 정확히 → 되살리지 않음", const(800), const(300), N),
        ("경계: 되살림 4/5 → 지지", {**const(2000), 20: 1200}, const(300), H),
        ("경계: 되살림 3/5 → 보류", {**const(2000), 19: 1200, 20: 1200}, const(300), "보류 — 되살림 3/5·되살리지 않음 0/5"),
    ]
    for nm, fx, fn, want in scen:
        v, iv, _ = run(fx, fn)
        chk(nm, v, want)
        chk(nm + " — 독립 대조 같음", iv, v)
    # 정정 1: 두 측정 분류가 다르면 가운데
    scen2 = [
        ("정정 1: 둘째 측정 X 0.12(가운데) → 전부 가운데", const(2000), const(300), const(1200), const(300), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("정정 1: 한 뇌만 둘째 측정 다름 → 되살림 4/5 지지", const(2000), const(300), {**const(2000), 18: 1200}, const(300), H),
        ("정정 1: 두 뇌 둘째 측정 다름 → 3/5 보류", const(2000), const(300), {**const(2000), 18: 1200, 19: 1200}, const(300), "보류 — 되살림 3/5·되살리지 않음 0/5"),
        ("정정 1: 첫 null·둘째 되살림 → 가운데", const(500), const(300), const(2000), const(300), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("정정 1: 둘째만 차 3996 → 가운데", const(1500), const(500), const(1500), const(501), "보류 — 되살림 0/5·되살리지 않음 0/5"),
        ("정정 1: 둘째 측정 경계 S_X 3200 → null 일치", const(700), const(300), const(800), const(250), N),
        ("정정 1: 두 측정 값 달라도 분류 같으면 그 분류", const(2000), const(300), const(1800), const(450), H),
    ]
    for nm, fx, fn, fxp, fnp, want in scen2:
        v, iv, _ = run(fx, fn, fxp=fxp, fnp=fnp)
        chk(nm, v, want)
        chk(nm + " — 독립 대조 같음", iv, v)
    fails = [
        ("X 측정 맥락 줄 없음", {("F", "X", 16): {"no_ctx": True}}, True, "측정 맥락"),
        ("끈 맥락 발화 3", {("U", "X", 17): {"ctx_off": 3}}, True, "측정 맥락"),
        ("게이팅 비활성 91 > 활성 9000/100", {("F", "X", 18): {"inactive": 91}}, True, "측정 맥락"),
        ("맥락 없음 측정에 맥락 줄", {("F", "none", 19): {"ctx_in_none": True}}, True, "맥락 없음 측정에 맥락 줄"),
        ("노출 816", {("F", "none", 20): {"expo": 816}}, True, "노출 816"),
        ("F 적재 없음", {("F", "X", 17): {"no_feat": True}}, False, "F 적재 []"),
        ("U 에 F 적재", {("U", "none", 18): {"feat_in_u": True}}, False, "U 에 F 적재"),
        ("동작점 U −1300", {("U", "X", 19): {"tonic": -1300.0}}, False, "동작점 [-1300.0]"),
        ("세계 18300", {("F", "X", 20): {"world": 18300}}, False, "세계"),
        ("g_same 0", {("F", "none", 16): {"g_same": 0}}, False, "RCCHK"),
        ("같은 값(X = none, J_pre)", {("F", "X", 17): {"same_as_none": True}}, True, "같은 값 F b17(X = none RCJ J_pre"),
        ("같은 값(X = none, J_post)", {("U", "X", 18): {"same_as_none_post": True}}, True, "같은 값 U b18(X = none RCJ J_post"),
    ]
    for nm, md, vcatch, kw in fails:
        v, iv, out = run(const(2000), const(300), md)
        jl = ([ln for ln in out["J"].splitlines() if ln.startswith("조작검증 실패: ")] or [""])[-1]
        chk("조작검증 실패: " + nm, (v, kw in jl), ("보류(조작검증 실패)", True))
        if vcatch:
            chk("조작검증 실패(독립): " + nm, iv, "보류(조작검증 실패)")
    v, iv, _ = run(const(2000), const(300), {("F", "X", 18): {"missing": True}})
    chk("결측(F X b18 없음)", (v.startswith("보류(결측"), iv.startswith("보류(결측")), (True, True))
    v, iv, out = run(const(2000), const(300))
    chk("보고 줄 — S·차·분류(두 측정)", "b16 F: 첫 측정 S_none 1200 · S_X 8000 · 차 +6800 → 되살림 · 둘째 측정 S_none 1200 · S_X 8000 · 차 +6800 → 되살림 ⇒ 되살림" in out["J"], True)
    chk("보고 줄 — 측정 이력 차(S 밖 쌍 37·11)", "측정 이력 차 최대 F none 11 · X 37 · U none 11 · X 37" in out["J"], True)
    chk("통과 자료에 조작검증 실패 줄 없음", ("조작검증 실패" in out["J"], "조작검증 실패" in out["V"]), (False, False))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
