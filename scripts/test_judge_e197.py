#!/usr/bin/env python3
"""judge_e197 합성 시험 — 임시 저장소 구조에 가짜 평가 로그·지수 파일을 만들어 판정이 사전 규칙과 같은지 본다.
실행: python3 scripts/test_judge_e197.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e197 as J  # noqa: E402

BR = J.BRAINS
NONE_M = {("X", "A", "train"): 500, ("X", "A", "new"): 400, ("X", "B", "train"): -300, ("X", "B", "new"): -200,
          ("Y", "A", "train"): 100, ("Y", "A", "new"): 50, ("Y", "B", "train"): -600, ("Y", "B", "new"): -500}
SEL_A = {("X", "A"): 800, ("X", "B"): 900, ("Y", "A"): 700, ("Y", "B"): 1100}     # E196 sel 비슷한 값(회귀 기준과 같게 씀)


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
        Z["sxi_" + sd][:5] = 0.0                       # 부호 0 (sgn 팔에서 init)
        Z["csi_" + sd] = np.round(rs.uniform(-1, 1, 300), 3)
        Z["resp_" + sd] = rs.rand(300) < 0.3
        tot = rs.rand(300)
        tot[rs.rand(300) < 0.2] = 0.0                  # 발화 없는 KC (sgnA 대상 아님)
        Z["tot_" + sd] = tot
        Z["npc_" + sd] = np.int64(18)
    np.savez(path, **Z)


def orc_line(npz, arm, theta="0.50", bump=None, as_arm=None):
    ex = J.expect_counts(npz, as_arm or arm)
    d = {"arm": arm, "theta": theta, "file": os.path.basename(npz)}
    d.update({k: str(v) for k, v in ex.items()})
    if bump:
        d[bump] = str(int(d[bump]) + 1)
    return d


def arm_mods(av):
    mm = {}
    for x, c, sg in J.CELLS:
        for s in ("train", "new"):
            a = av[(x, c)] if s == "train" else av[(x, c)] // 2 + 1
            mm[(x, c, s)] = NONE_M[(x, c, s)] + sg * a
    return mm


def build(root, a_sgnR, a_sgnA, brains_override=None, mutate=None):
    L = os.path.join(root, "research/experiments/logs/E197")
    L6 = os.path.join(root, "research/experiments/logs/E196")
    T = os.path.join(root, "research/experiments/traces/E196")
    for d in (L, L6, T):
        os.makedirs(d, exist_ok=True)
    for b in BR:
        npz = os.path.join(T, "sxi_b%d.npz" % b)
        make_npz(npz, b)
        ov = (brains_override or {}).get(b, {})
        mods = {"none": dict(NONE_M), "sel": arm_mods(SEL_A), "sgnR": arm_mods(ov.get("sgnR", a_sgnR)), "sgnA": arm_mods(ov.get("sgnA", a_sgnA))}
        spec = {}
        for w in ("none",) + J.ARMS:
            for x in "XY":
                spec[(w, x)] = {"mods": mods[w], "pushed": 0 if w == "none" else 4,
                                "orc": None if w == "none" else orc_line(npz, w), "ver": True, "sat": False, "trials": 96, "kt": True}
        ref = {("none", x): dict(NONE_M) for x in "XY"}
        ref.update({("sel", x): arm_mods(SEL_A) for x in "XY"})
        if mutate:
            mutate(b, spec, ref, npz)
        for (w, x), sp in spec.items():
            if sp is None:
                continue
            write_eval(os.path.join(L, "ev_b%d_%s_%s.log" % (b, w, x)), x, sp["mods"], sp["pushed"], sp["orc"], sp["ver"], sp["sat"], sp["trials"], sp["kt"])
        for (w, x), mm in ref.items():
            write_eval(os.path.join(L6, "ev_b%d_%s_%s.log" % (b, w, x)), x, mm, 0 if w == "none" else 4,
                       None if w == "none" else orc_line(npz, w))


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


TWO = {("X", "A"): 1500, ("X", "B"): 1500, ("Y", "A"): 300, ("Y", "B"): 300}
CASES = []


def case(name, want, a_sgnR, a_sgnA, **kw):
    CASES.append((name, want, a_sgnR, a_sgnA, kw))


case("T01 sgnR 획득 5/5 → 표현 충분", "표현 충분(부호 판독", full(2000), full(2500))
case("T02 sgnR 실패 5/5 → 표현·출력 병목", "표현·출력 병목", full(300), full(400))
case("T03 sgnR 부분(2칸) 5/5 → 부분", "판정: 부분", TWO, full(0))
case("T04 경계 1000 → 획득", "표현 충분(부호 판독", full(1000), full(0))
case("T05 경계 999(한 칸) → 부분", "판정: 부분", {("X", "A"): 999, ("X", "B"): 1000, ("Y", "A"): 1000, ("Y", "B"): 1000}, full(0))
case("T06 sgnR 획득 4/5(한 뇌 부분) → 표현 충분", "표현 충분(부호 판독", full(2000), full(0), brains_override={12: {"sgnR": TWO}})
case("T07 sgnR 획득 3/5·부분 2 → 보류", "판정: 보류", full(2000), full(0), brains_override={12: {"sgnR": TWO}, 13: {"sgnR": TWO}})
case("T08 sgnR 실패 4/5(한 뇌 획득) → 병목", "표현·출력 병목", full(100), full(0), brains_override={10: {"sgnR": full(2000)}})
case("T09 sgnA 만 획득(sgnR 실패) → 병목(sgnA 는 보고)", "표현·출력 병목", full(100), full(3000))
case("T10 같은 값(sgnR = none) → 조작 무효 의심", "조작검증 실패", full(0), full(2000))
case("T11 sgnR 효과 2(경계 안) → 조작 무효 의심", "조작검증 실패", full(2), full(2000))


def m_missing(b, spec, ref, npz):
    if b == 12:
        spec[("sgnA", "Y")] = None


def m_pushed(b, spec, ref, npz):
    if b == 13:
        spec[("sgnR", "X")]["pushed"] = 8


def m_count(b, spec, ref, npz):
    if b == 14:
        spec[("sgnR", "Y")]["orc"] = orc_line(npz, "sgnR", bump="same_r")


def m_sgnA_resp(b, spec, ref, npz):
    if b == 10:      # sgnA 수를 반응 KC 로 센 줄(대상 정의 오류) → 불일치
        spec[("sgnA", "X")]["orc"] = orc_line(npz, "sgnA", as_arm="sgnR")


def m_reg3(b, spec, ref, npz):
    if b == 10:
        ref[("none", "X")][("X", "A", "train")] += 3


def m_reg2(b, spec, ref, npz):
    if b == 10:
        ref[("none", "X")][("X", "A", "train")] += 2


def m_selreg(b, spec, ref, npz):
    if b == 11:
        ref[("sel", "Y")][("Y", "B", "new")] -= 3


def m_sat(b, spec, ref, npz):
    if b == 11:
        spec[("sgnA", "X")]["sat"] = True


def m_ver(b, spec, ref, npz):
    if b == 12:
        spec[("sgnR", "Y")]["ver"] = False


def m_theta(b, spec, ref, npz):
    if b == 13:
        spec[("sgnR", "Y")]["orc"] = orc_line(npz, "sgnR", theta="0.30")


def m_armname(b, spec, ref, npz):
    if b == 14:
        spec[("sgnA", "X")]["orc"] = orc_line(npz, "sgnR")


def m_trials(b, spec, ref, npz):
    if b == 10:
        spec[("none", "Y")]["trials"] = 95


case("T12 로그 결측 → 보류(결측)", "보류(결측", full(2000), full(2000), mutate=m_missing)
case("T13 pushed 8 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_pushed)
case("T14 sgnR 오라클 수 불일치 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_count)
case("T15 sgnA 대상을 반응 KC 로 셈 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_sgnA_resp)
case("T16 none 회귀 차 3 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_reg3)
case("T17 none 회귀 차 2 → 허용", "표현 충분(부호 판독", full(2000), full(2000), mutate=m_reg2)
case("T18 sel 회귀 차 3 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_selreg)
case("T19 포화 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_sat)
case("T20 장치 검증 줄 없음 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_ver)
case("T21 θ 0.30 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_theta)
case("T22 팔 이름 다름 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_armname)
case("T23 trials 95 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_trials)


def main():
    ok = 0
    for name, want, a1, a2, kw in CASES:
        root = tempfile.mkdtemp()
        try:
            build(root, a1, a2, kw.get("brains_override"), kw.get("mutate"))
            got, out = run(root)
        finally:
            shutil.rmtree(root)
        good = want in got
        ok += good
        print("[%s] %s → %s" % ("통과" if good else "실패", name, got))
        if not good:
            print("      " + out.replace("\n", "\n      ")[:1200])
    print("합성 %d/%d" % (ok, len(CASES)))
    return ok == len(CASES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
