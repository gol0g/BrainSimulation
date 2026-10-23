#!/usr/bin/env python3
"""posthoc_e207_core main() 끝까지 합성 시험 — 정답을 아는 가짜 저장소 트리(스냅숏·w_R·w_P·측정 로그 RCKC)를 만들어 출력 줄을 손계산 기대값과 맞춘다.
사용: python3 test_posthoc_e207_main.py <모듈 디렉터리(posthoc_e207_core·posthoc_kc_rewrite 가 있는 곳)>
(2026-10-22 절전 중 로컬 사본으로 먼저, 깨어나면 저장소 scripts/ 로 다시)"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

import numpy as np

MOD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MOD)
import posthoc_e207_core as C  # noqa: E402

BR = (16, 17, 18, 19, 20)


def snap():
    """KC 0~99 가 쪽마다 motor_l·motor_r 에 시냅스 2개씩."""
    pre = np.repeat(np.arange(100), 2)
    return {"kc_%s_to_motor_%s__pre" % (sd, m): pre for sd in "lr" for m in "lr"}


def weights(d):
    """d[KC] = 원하는 선호(mean w_l − mean w_r) → 시냅스별 무게(w_l = 150 + d/2, w_r = 150 − d/2)."""
    out = {}
    for sd in "lr":
        wl = np.repeat(150.0 + d / 2.0, 2); wr = np.repeat(150.0 - d / 2.0, 2)
        out["kc_%s_to_motor_l" % sd] = wl; out["kc_%s_to_motor_r" % sd] = wr
    return out


def write_meas(path, sets_by_tag):
    with open(path, "w", encoding="utf-8") as f:
        for tag, sets in sets_by_tag.items():
            for sd in "lr":
                for c in "ABCD":
                    for i in range(2):   # 노출 사례 2개 — 같은 집합(절반 이상 = 1 이지만 둘 다 같아 집합이 정확)
                        ids = sorted(sets[c])
                        f.write("RCKC tag=%s side=%s set=train cat=%s i=%d n=%d ids=%s\n" % (tag, sd, c, i, len(ids), ",".join(str(x) for x in ids)))
                    f.write("RCKC tag=%s side=%s set=new cat=%s i=0 n=1 ids=2999\n" % (tag, sd, c))


def main():
    ok = tot = 0

    def chk(name, cond):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(("OK   " if cond else "실패 ") + name)

    core = set(range(0, 8)) | set(range(100, 110))
    SX = {"A": core | set(range(200, 210)), "B": core | set(range(210, 220)),
          "C": core | set(range(200, 205)) | set(range(220, 225)), "D": core | set(range(230, 240))}
    SN = {"A": {300, 301}, "B": {310}, "C": {301, 302}, "D": {311}}
    dR = np.zeros(100); dR[0:10] = 100.0
    dP = dR.copy(); dP[5:10] = -100.0; dP[50:55] = 80.0
    root = tempfile.mkdtemp(prefix="tphm_")
    cwd = os.getcwd()
    try:
        E = os.path.join(root, "research/experiments")
        os.makedirs(os.path.join(E, "traces/E176/snap")); os.makedirs(os.path.join(E, "logs/E207"))
        for ed in ("E205", "E206"):
            os.makedirs(os.path.join(E, "traces", ed))
        for b in BR:
            np.savez(os.path.join(E, "traces/E176/snap/conn_b%d.npz" % b), **snap())
            for ed in ("E205", "E206"):
                np.savez(os.path.join(E, "traces", ed, "w_R_b%d.npz" % b), **weights(dR))
                np.savez(os.path.join(E, "traces", ed, "w_P_b%d.npz" % b), **weights(dP))
            for rep in ("U", "F"):
                write_meas(os.path.join(E, "logs/E207/meas_%s_none_b%d.log" % (rep, b)), {"pre": SN, "post": SN})
                write_meas(os.path.join(E, "logs/E207/meas_%s_X_b%d.log" % (rep, b)), {"pre": SX, "post": SX})
        os.chdir(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            C.main(["x"])
        out = buf.getvalue()
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)
    lines = [ln for ln in out.splitlines() if ln.startswith("b")]
    chk("뇌·표현 줄 10개", len(lines) == 10)
    # 손계산: 쪽마다 핵 18·|A| 28 → 두 쪽 36·56. J(A,C) = 23/33 → 핵 빼면 5/15. J(A,B)·J(B,D) = 18/38 → 0. R 맞춤 KC 0~9(쪽마다 10, 두 쪽 20) 중 핵(0~7) 16 → 0.800.
    # P 바뀜 = 5~9(|−100 − 100|) + 50~54(|80 − 0|) → 두 쪽 20, 핵 안(5·6·7) 6 → 0.300. 맥락 없음: 핵 0, J(A,C) = 1/3, J(A,B) 0, 핵 몫 0.
    want_x = "X·첫 핵 36(|A| 56) J(A,C) 0.697→0.333 · J(B,D) 0.474→0.000 · J(A,B) 0.474→0.000 · R 맞춤 20 중 핵 0.800 · P 바뀜 20 중 핵 0.300"
    want_n = "none·첫 핵 0(|A| 4) J(A,C) 0.333→0.333 · J(B,D) 0.000→0.000 · J(A,B) 0.000→0.000 · R 맞춤 20 중 핵 0.000 · P 바뀜 20 중 핵 0.000"
    chk("맥락 X 첫 측정 조각(핵·핵 뺀 J·학습 KC 핵 몫)", all(want_x in ln for ln in lines))
    chk("맥락 X 둘째 측정 조각", all(want_x.replace("X·첫", "X·둘째") in ln for ln in lines))
    chk("맥락 없음 조각(핵 0)", all(want_n in ln for ln in lines))
    chk("U·F 둘 다(E205·E206 무게 경로)", sum(ln.startswith("b16 U:") for ln in lines) == 1 and sum(ln.startswith("b16 F:") for ln in lines) == 1)
    chk("새 사례(2999)는 집합 밖 — |A| 에 안 듦", "|A| 56" in out and "|A| 58" not in out)
    print(out.splitlines()[1][:300] if len(out.splitlines()) > 1 else out)
    print("합성 시험(main 끝까지) %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
