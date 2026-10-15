#!/usr/bin/env python3
"""judge_e196 합성 시험 — 임시 저장소 구조에 가짜 평가 로그·지수 파일을 만들어 판정이 사전 규칙과 같은지 본다.
실행: python3 scripts/test_judge_e196.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e196 as J  # noqa: E402

BR = J.BRAINS
NONE_M = {("X", "A", "train"): 500, ("X", "A", "new"): 400, ("X", "B", "train"): -300, ("X", "B", "new"): -200,
          ("Y", "A", "train"): 100, ("Y", "A", "new"): 50, ("Y", "B", "train"): -600, ("Y", "B", "new"): -500}


def fmt(v):
    return "%+.4f" % (v / 10000.0)


def write_eval(path, ctx, mods, pushed, orc=None, ver=True, sat=False, trials=96, kt=True):
    with open(path, "w", encoding="utf-8") as f:
        if kt:
            f.write("[E153 종류 입력 적재] 검증 일치 n=4\n")
        if orc is not None:
            f.write("[E196 오라클] 설계 " + " ".join("%s=%s" % kv for kv in orc.items()) + "\n")
            if ver:
                f.write("[E196 오라클 장치 검증] 일치 n=4\n")
        for c in "AB":
            for s in ("train", "new"):
                mo = "0.700000/0.700000" if sat else "0.500000/0.400000"
                f.write("RCEVAL cat=%s set=%s mod=%s off=+0.0000 acc=50.0 trials=%d feat_on=%d ctx=%s ctx_on=%d | side=left n=48 motor L/R %s KC L/R 0.010000/0.010000\n"
                        % (c, s, fmt(mods[(ctx, c, s)]), trials, trials, ctx, trials, mo))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[]\n" % pushed)


def make_npz(path, seed):
    rs = np.random.RandomState(seed)
    Z = {}
    for sd in "lr":
        Z["sxi_" + sd] = np.round(rs.uniform(-1, 1, 300), 3)
        Z["csi_" + sd] = np.round(rs.uniform(-1, 1, 300), 3)
        Z["resp_" + sd] = rs.rand(300) < 0.3
        Z["tot_" + sd] = rs.rand(300)
        Z["npc_" + sd] = np.int64(18)
    np.savez(path, **Z)


def orc_line(npz, arm, theta="0.50", bump=None):
    ex = J.expect_counts(npz, arm)
    d = {"arm": arm, "theta": theta, "file": os.path.basename(npz)}
    d.update({k: str(v) for k, v in ex.items()})
    if bump:
        d[bump] = str(int(d[bump]) + 1)
    return d


def build(root, a_sel, a_eq, k_cat, brains_override=None, mutate=None):
    """a_sel·a_eq: {(x,c): a_train} (a_new = a_train//2 + 1 → 일반화), k_cat: {(x,c): 범주 방향 노출 효과}. mutate(b, files) 로 결함 주입."""
    L = os.path.join(root, "research/experiments/logs/E196")
    L3 = os.path.join(root, "research/experiments/logs/E193")
    T = os.path.join(root, "research/experiments/traces/E196")
    for d in (L, L3, T):
        os.makedirs(d, exist_ok=True)
    for b in BR:
        npz = os.path.join(T, "sxi_b%d.npz" % b)
        make_npz(npz, b)
        sel = (brains_override or {}).get(b, {}).get("sel", a_sel)
        eq = (brains_override or {}).get(b, {}).get("eq", a_eq)
        cat = (brains_override or {}).get(b, {}).get("cat", k_cat)
        mods = {"none": dict(NONE_M)}
        for w, av in (("sel", sel), ("eq", eq)):
            mm = {}
            for x, c, sg in J.CELLS:
                for s in ("train", "new"):
                    a = av[(x, c)] if s == "train" else av[(x, c)] // 2 + 1
                    mm[(x, c, s)] = NONE_M[(x, c, s)] + sg * a
            mods[w] = mm
        mm = {}
        for x in "XY":
            for c in "AB":
                for s in ("train", "new"):
                    kk = cat[(x, c)] if s == "train" else cat[(x, c)] // 2
                    mm[(x, c, s)] = NONE_M[(x, c, s)] + J.CATDIR[c] * kk
        mods["cat"] = mm
        spec = {}
        for w in ("none",) + J.ARMS:
            for x in "XY":
                spec[(w, x)] = {"mods": mods[w], "pushed": 0 if w == "none" else 4,
                                "orc": None if w == "none" else orc_line(npz, w), "ver": True, "sat": False, "trials": 96, "kt": True}
        ref = {x: dict(NONE_M) for x in "XY"}
        if mutate:
            mutate(b, spec, ref, npz)
        for (w, x), sp in spec.items():
            if sp is None:
                continue
            write_eval(os.path.join(L, "ev_b%d_%s_%s.log" % (b, w, x)), x, sp["mods"], sp["pushed"], sp["orc"], sp["ver"], sp["sat"], sp["trials"], sp["kt"])
        for x in "XY":
            write_eval(os.path.join(L3, "ev_b%d_none_%s.log" % (b, x)), x, ref[x], 0)


def run(root):
    cwd = os.getcwd()
    os.chdir(root)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            J.main()
    finally:
        os.chdir(cwd)
    out = buf.getvalue()
    return [ln for ln in out.splitlines() if ln.startswith("판정")][-1], out


def full(v):
    return {(x, c): v for x in "XY" for c in "AB"}


CASES = []


def case(name, want, a_sel, a_eq, k_cat, **kw):
    CASES.append((name, want, a_sel, a_eq, k_cat, kw))


case("T01 sel·eq 획득 → 동역학", "표현 충분·평형 충분", full(1500), full(1200), full(3000))
case("T02 sel 획득·eq 실패 → 규칙", "표현 충분·상관 평형 부족", full(1500), full(100), full(3000))
case("T03 sel 획득·eq 부분 → 평형 보류", "표현 충분·평형 보류", full(1500), {("X", "A"): 1500, ("X", "B"): 1500, ("Y", "A"): 0, ("Y", "B"): 0}, full(3000))
case("T04 sel 실패 → 표현·출력 병목", "표현·출력 병목", full(200), full(100), full(3000))
case("T05 sel 부분(2칸) → 부분", "부분", {("X", "A"): 1500, ("X", "B"): 1500, ("Y", "A"): 0, ("Y", "B"): 0}, full(0), full(3000))
case("T06 경계 1000 → 획득", "표현 충분·평형 충분", full(1000), full(1000), full(1000))
case("T07 경계 999(한 칸) → 부분", "부분", {("X", "A"): 999, ("X", "B"): 1000, ("Y", "A"): 1000, ("Y", "B"): 1000}, full(0), full(3000))
case("T08 같은 값(팔 = none) → 양성 대조 실패", "보류(양성 대조 실패", full(0), full(0), full(0))
case("T09 양성 대조 미달(한 값 999) → 보류", "보류(양성 대조 실패", full(1500), full(1500), {("X", "A"): 3000, ("X", "B"): 3000, ("Y", "A"): 3000, ("Y", "B"): 999})
case("T10 양성 대조 3/5 → 보류", "보류(양성 대조 실패", full(1500), full(1500), full(3000),
     brains_override={10: {"cat": full(0)}, 11: {"cat": full(0)}})
case("T11 양성 대조 4/5 → 통과", "표현 충분·평형 충분", full(1500), full(1500), full(3000), brains_override={10: {"cat": full(0)}})
case("T12 sel 획득 3/5 → 보류", "판정: 보류", full(1500), full(1500), full(3000),
     brains_override={10: {"sel": full(0)}, 11: {"sel": {("X", "A"): 1500, ("X", "B"): 1500, ("Y", "A"): 0, ("Y", "B"): 0}}})


def m_missing(b, spec, ref, npz):
    if b == 12:
        spec[("eq", "Y")] = None


def m_pushed(b, spec, ref, npz):
    if b == 13:
        spec[("sel", "X")]["pushed"] = 8


def m_count(b, spec, ref, npz):
    if b == 14:
        spec[("sel", "Y")]["orc"] = orc_line(npz, "sel", bump="cross_l")


def m_reg3(b, spec, ref, npz):
    if b == 10:
        ref["X"][("X", "A", "train")] += 3


def m_reg2(b, spec, ref, npz):
    if b == 10:
        ref["X"][("X", "A", "train")] += 2


def m_sat(b, spec, ref, npz):
    if b == 11:
        spec[("eq", "X")]["sat"] = True


def m_ver(b, spec, ref, npz):
    if b == 12:
        spec[("cat", "Y")]["ver"] = False


def m_theta(b, spec, ref, npz):
    if b == 13:
        spec[("eq", "Y")]["orc"] = orc_line(npz, "eq", theta="0.30")


def m_armname(b, spec, ref, npz):
    if b == 14:
        spec[("cat", "X")]["orc"] = orc_line(npz, "sel")


def m_trials(b, spec, ref, npz):
    if b == 10:
        spec[("none", "Y")]["trials"] = 95


case("T13 로그 결측 → 보류(결측)", "보류(결측", full(1500), full(1500), full(3000), mutate=m_missing)
case("T14 pushed 8 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_pushed)
case("T15 오라클 수 불일치 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_count)
case("T16 none 회귀 차 3 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_reg3)
case("T17 none 회귀 차 2 → 허용", "표현 충분·평형 충분", full(1500), full(1500), full(3000), mutate=m_reg2)
case("T18 포화 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_sat)
case("T19 장치 검증 줄 없음 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_ver)
case("T20 θ 0.30 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_theta)
case("T21 팔 이름 다름 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_armname)
case("T22 trials 95 → 조작검증 실패", "조작검증 실패", full(1500), full(1500), full(3000), mutate=m_trials)


def main():
    ok = 0
    for name, want, a_sel, a_eq, k_cat, kw in CASES:
        root = tempfile.mkdtemp(prefix="tj196_")
        try:
            build(root, a_sel, a_eq, k_cat, brains_override=kw.get("brains_override"), mutate=kw.get("mutate"))
            got, out = run(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        good = want in got
        ok += good
        print("[%s] %s → %s" % ("통과" if good else "실패", name, got))
        if not good:
            print(out)
    print("합성 %d/%d" % (ok, len(CASES)))
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
