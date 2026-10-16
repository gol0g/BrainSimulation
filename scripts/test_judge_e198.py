#!/usr/bin/env python3
"""judge_e198 합성 시험 — 임시 저장소에 가짜 평가 로그·지수·학습 가중치·스냅숏을 만들어 판정이 사전 규칙과 같은지 본다.
실행: python3 scripts/test_judge_e198.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e198 as J  # noqa: E402

BR = J.BRAINS
NK = 3000
NONE_M = {("X", "A", "train"): 500, ("X", "A", "new"): 400, ("X", "B", "train"): -300, ("X", "B", "new"): -200,
          ("Y", "A", "train"): 100, ("Y", "A", "new"): 50, ("Y", "B", "train"): -600, ("Y", "B", "new"): -500}
SGNR_A = {("X", "A"): 2000, ("X", "B"): 2500, ("Y", "A"): 1800, ("Y", "B"): 2200}    # E197 sgnR 비슷한 값(회귀 기준과 같게 씀)


def fmt(v):
    return "%+.4f" % (v / 10000.0)


def write_eval(path, ctx, mods, pushed, orc=None, ver=True, sat=False, trials=96, kt=True, extra=()):
    with open(path, "w", encoding="utf-8") as f:
        if kt:
            f.write("[E153 종류 입력 적재] 검증 일치 n=4\n")
        if orc is not None:
            f.write("[E196 오라클] 설계 " + " ".join("%s=%s" % kv for kv in orc.items()) + "\n")
            for ln in extra:
                f.write(ln + "\n")
            if ver:
                f.write("[E196 오라클 장치 검증] 일치 n=4\n")
        for c in "AB":
            for s in ("train", "new"):
                mo = "0.700000/0.700000" if sat else "0.500000/0.400000"
                f.write("RCEVAL cat=%s set=%s mod=%s off=+0.0000 acc=50.0 trials=%d feat_on=%d ctx=%s ctx_on=%d | side=left n=48 motor L/R %s KC L/R 0.010000/0.010000\n"
                        % (c, s, fmt(mods[(ctx, c, s)]), trials, trials, ctx, trials, mo))
        f.write("=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[]\n" % pushed)


def make_inputs(root, b):
    rs = np.random.RandomState(b)
    T6 = os.path.join(root, "research/experiments/traces/E196"); T3 = os.path.join(root, "research/experiments/traces/E193")
    TS = os.path.join(root, "research/experiments/traces/E176/snap")
    for d in (T6, T3, TS):
        os.makedirs(d, exist_ok=True)
    Z = {}
    for sd in "lr":
        Z["sxi_" + sd] = np.round(rs.uniform(-1, 1, NK), 3)
        Z["sxi_" + sd][:7] = 0.0
        Z["csi_" + sd] = np.round(rs.uniform(-1, 1, NK), 3)
        Z["resp_" + sd] = rs.rand(NK) < 0.3
        Z["tot_" + sd] = rs.rand(NK)
        Z["npc_" + sd] = np.int64(18)
    np.savez(os.path.join(T6, "sxi_b%d.npz" % b), **Z)
    W, S = {}, {}
    for kn in "lr":
        for m in "lr":
            nm = "kc_%s_to_motor_%s" % (kn, m)
            pre = np.sort(rs.randint(0, NK, 6000)).astype(np.uint32)
            pre = pre[pre >= 5]                                   # KC 0~4 는 시냅스 없음(Δ = 0)
            w = 150 + rs.normal(0, 5, pre.size)
            w[pre < 40] = 150.0                                   # 학습 안 된 KC(Δ = 0 경계)
            S[nm + "__pre"] = pre
            S[nm + "__post"] = rs.randint(0, 500, pre.size).astype(np.uint32)
            W[nm] = w.astype(np.float32)
    np.savez(os.path.join(T3, "w_b%d.npz" % b), **W)
    np.savez(os.path.join(TS, "conn_b%d.npz" % b), **S)
    return os.path.join(T6, "sxi_b%d.npz" % b), os.path.join(T3, "w_b%d.npz" % b), os.path.join(TS, "conn_b%d.npz" % b)


def arm_mods(av):
    mm = {}
    for x, c, sg in J.CELLS:
        for s in ("train", "new"):
            a = av[(x, c)] if s == "train" else av[(x, c)] // 2 + 1
            mm[(x, c, s)] = NONE_M[(x, c, s)] + sg * a
    return mm


def build(root, a_l, a_la, brains_override=None, mutate=None):
    L = os.path.join(root, "research/experiments/logs/E198")
    L7 = os.path.join(root, "research/experiments/logs/E197")
    for d in (L, L7):
        os.makedirs(d, exist_ok=True)
    for b in BR:
        npz, wf, sf = make_inputs(root, b)
        D, conn = J.deltas(wf, sf)
        ov = (brains_override or {}).get(b, {})
        mods = {"none": dict(NONE_M), "sgnR": arm_mods(SGNR_A), "lsgnR": arm_mods(ov.get("lsgnR", a_l)), "lsgnA": arm_mods(ov.get("lsgnA", a_la))}
        spec = {}
        for w in ("none",) + J.ARMS:
            for x in "XY":
                orc, extra = None, []
                if w != "none":
                    ec, ea = J.expect(npz, w, D)
                    orc = {"arm": w, "theta": "0.50", "file": npz}
                    orc.update({k: str(v) for k, v in ec.items()})
                    if w != "sgnR":
                        extra = ["[E198 학습 부호] 가중치=%s 부호 일치(대상 중 sign(Δ) = sign(SXI)) agree_l=%d agree_r=%d" % (wf, ea["agree_l"], ea["agree_r"]),
                                 "[E198 연결 확인] " + " | ".join("%s n=%d pre_sum=%d" % (nm, n, ps) for nm, (n, ps) in sorted(conn.items()))]
                spec[(w, x)] = {"mods": mods[w], "pushed": 0 if w == "none" else 4, "orc": orc, "extra": extra, "ver": True, "sat": False, "trials": 96, "kt": True}
        ref = {("none", x): dict(NONE_M) for x in "XY"}
        ref.update({("sgnR", x): arm_mods(SGNR_A) for x in "XY"})
        if mutate:
            mutate(b, spec, ref, (npz, wf, sf))
        for (w, x), sp in spec.items():
            if sp is None:
                continue
            write_eval(os.path.join(L, "ev_b%d_%s_%s.log" % (b, w, x)), x, sp["mods"], sp["pushed"], sp["orc"], sp["ver"], sp["sat"], sp["trials"], sp["kt"], sp["extra"])
        for (w, x), mm in ref.items():
            write_eval(os.path.join(L7, "ev_b%d_%s_%s.log" % (b, w, x)), x, mm, 0 if w == "none" else 4)


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


def case(name, want, a_l, a_la, **kw):
    CASES.append((name, want, a_l, a_la, kw))


case("T01 lsgnR 획득 5/5 → 학습 방향 충분", "학습 방향 충분", full(2000), full(2500))
case("T02 lsgnR 실패 5/5 → 학습 방향 부족", "학습 방향 부족", full(300), full(400))
case("T03 lsgnR 부분 5/5 → 부분", "판정: 부분", TWO, full(0))
case("T04 경계 1000 → 획득", "학습 방향 충분", full(1000), full(0))
case("T05 경계 999(한 칸) → 부분", "판정: 부분", {("X", "A"): 999, ("X", "B"): 1000, ("Y", "A"): 1000, ("Y", "B"): 1000}, full(0))
case("T06 lsgnR 획득 4/5 → 충분", "학습 방향 충분", full(2000), full(0), brains_override={12: {"lsgnR": TWO}})
case("T07 lsgnR 획득 3/5·부분 2 → 보류", "판정: 보류", full(2000), full(0), brains_override={12: {"lsgnR": TWO}, 13: {"lsgnR": TWO}})
case("T08 같은 값(lsgnR = none) → 조작 무효 의심", "조작검증 실패", full(0), full(2000))


def m_missing(b, spec, ref, f):
    if b == 12:
        spec[("lsgnA", "Y")] = None


def m_nosnap(b, spec, ref, f):
    if b == 13:
        os.remove(f[2])


def m_count(b, spec, ref, f):
    if b == 14:
        spec[("lsgnR", "Y")]["orc"]["cross_l"] = str(int(spec[("lsgnR", "Y")]["orc"]["cross_l"]) + 1)


def m_agree(b, spec, ref, f):
    if b == 10:
        e = spec[("lsgnR", "X")]["extra"]
        e[0] = e[0].replace("agree_l=", "agree_l=1")


def m_conn(b, spec, ref, f):
    if b == 11:
        e = spec[("lsgnA", "X")]["extra"]
        e[1] = e[1].replace("pre_sum=", "pre_sum=9", 1)


def m_wname(b, spec, ref, f):
    if b == 12:
        e = spec[("lsgnR", "Y")]["extra"]
        e[0] = e[0].replace("w_b12.npz", "w_b12_E194.npz")


def m_lsgnA_resp(b, spec, ref, f):
    if b == 13:
        npz, wf, sf = f
        D, _ = J.deltas(wf, sf)
        ec, _ = J.expect(npz, "lsgnR", D)            # lsgnA 대상을 반응 KC 로 센 줄
        spec[("lsgnA", "X")]["orc"].update({k: str(v) for k, v in ec.items()})


def m_reg3(b, spec, ref, f):
    if b == 10:
        ref[("none", "X")][("X", "A", "train")] += 3


def m_reg2(b, spec, ref, f):
    if b == 10:
        ref[("none", "X")][("X", "A", "train")] += 2


def m_sreg(b, spec, ref, f):
    if b == 11:
        ref[("sgnR", "Y")][("Y", "B", "new")] -= 3


def m_sat(b, spec, ref, f):
    if b == 11:
        spec[("lsgnA", "X")]["sat"] = True


def m_ver(b, spec, ref, f):
    if b == 12:
        spec[("lsgnR", "Y")]["ver"] = False


def m_pushed(b, spec, ref, f):
    if b == 13:
        spec[("lsgnR", "X")]["pushed"] = 8


def m_trials(b, spec, ref, f):
    if b == 14:
        spec[("none", "Y")]["trials"] = 95


def m_wsize(b, spec, ref, f):
    if b == 14:
        W = dict(np.load(f[1]))
        W["kc_l_to_motor_r"] = W["kc_l_to_motor_r"][:-1]
        np.savez(f[1], **W)


case("T09 로그 결측 → 보류(결측)", "보류(결측", full(2000), full(2000), mutate=m_missing)
case("T10 스냅숏 결측 → 보류(결측)", "보류(결측", full(2000), full(2000), mutate=m_nosnap)
case("T11 lsgnR 오라클 수 불일치 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_count)
case("T12 부호 일치 수 불일치 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_agree)
case("T13 연결 확인 불일치 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_conn)
case("T14 학습 가중치 파일 다름 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_wname)
case("T15 lsgnA 대상을 반응 KC 로 셈 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_lsgnA_resp)
case("T16 none 회귀 차 3 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_reg3)
case("T17 none 회귀 차 2 → 허용", "학습 방향 충분", full(2000), full(2000), mutate=m_reg2)
case("T18 sgnR 회귀 차 3 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_sreg)
case("T19 포화 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_sat)
case("T20 장치 검증 줄 없음 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_ver)
case("T21 pushed 8 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_pushed)
case("T22 trials 95 → 조작검증 실패", "조작검증 실패", full(2000), full(2000), mutate=m_trials)
case("T23 가중치·스냅숏 크기 불일치 → 보류(결측)", "보류(결측", full(2000), full(2000), mutate=m_wsize)


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
            print("      " + out.replace("\n", "\n      ")[:1500])
    print("합성 %d/%d" % (ok, len(CASES)))
    return ok == len(CASES)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
