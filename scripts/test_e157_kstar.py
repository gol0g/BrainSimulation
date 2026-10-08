#!/usr/bin/env python3
"""e157_kstar.py 합성 시험: 최근접·동점(1 에 가까운 값)·±10% 경계·밖이면 실패·배율 줄/적재 줄 제외·결측·k=1 재현 불일치·로그 파싱.
실행: python3 scripts/test_e157_kstar.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e157_kstar as S


def rows(arm, tots, over=None):
    r = {k: (t, 2 if arm == "Fk" else 0, 2) for k, t in zip(S.GRID[arm], tots)}
    for k, v in (over or {}).items():
        r[k] = v
    return r


ok_all = True


def chk(name, arm, rws, want):
    global ok_all
    k, _, why = S.select(arm, rws)
    good = k == want
    ok_all &= good
    print("%-36s 기대 %-5s → %-5s %s" % (name, want, k, "✓" if good else "✗ %s" % why))


chk("Fk 최근접(48,000)", "Fk", rows("Fk", (40000, 44000, 48000, 52000, 56000)), "0.80")
chk("Fk 동점 → 1 에 가까운 값", "Fk", rows("Fk", (40000, 44000, 47856, 49856, 56000)), "0.85")
chk("Fk 경계 −10% 정확(43,971) 통과", "Fk", rows("Fk", (30000, 31000, 32000, 33000, 43971)), "0.90")
chk("Fk −10% 밖(43,970) 실패", "Fk", rows("Fk", (30000, 31000, 32000, 33000, 43970)), None)
chk("Dk 경계 +10% 정확 통과", "Dk", rows("Dk", (67444, 80000, 90000, 95000, 99000)), "1.10")
chk("Dk +10% 밖 실패", "Dk", rows("Dk", (67445, 80000, 90000, 95000, 99000)), None)
chk("Dk 최근접(61,000)", "Dk", rows("Dk", (55000, 58000, 61000, 64000, 67000)), "1.30")
chk("Fk 최근접 배율 1줄 → 다음", "Fk", rows("Fk", (40000, 44000, 48000, 52000, 56000), {"0.80": (48000, 2, 1)}), "0.85")
chk("Fk 최근접 적재 1줄 → 다음", "Fk", rows("Fk", (40000, 44000, 48000, 52000, 56000), {"0.80": (48000, 1, 2)}), "0.85")
chk("Dk 최근접 적재 있음 → 다음", "Dk", rows("Dk", (55000, 58000, 61000, 64000, 67000), {"1.30": (61000, 2, 2)}), "1.40")
chk("Fk 최근접 발화 없음 → 다음", "Fk", rows("Fk", (40000, 44000, 48000, 52000, 56000), {"0.80": (None, 2, 2)}), "0.85")
r = rows("Fk", (40000, 44000, 48000, 52000, 56000)); del r["0.80"]
chk("Fk 최근접 결측 → 다음", "Fk", r, "0.85")
chk("Fk 전부 결측 → 실패", "Fk", {}, None)


def log(sp_l, sp_r, nld, nsc):
    return ("[E153 종류 입력 적재] k 검증 일치 — x\n" * nld + "[E157 종류 입력 배율] k=0.8000 검증 일치 — x\n" * nsc
            + "=> KCRATE kc_l | 좌선택 1 우선택 2 비선택 3 무활동 4 | 희석 0.1 | x | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n"
              "=> KCRATE kc_r | 좌선택 1 우선택 2 비선택 3 무활동 4 | 희석 0.1 | x | 제시 스파이크 %d 기준선(제시창) 평균 0.0500 | KC별 ΔS y\n" % (sp_l, sp_r))


def run_main(repro_d=48856):
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "logs", "E157", "calib")
        os.makedirs(d)
        open(os.path.join(d, "kcrate_D_k1.00_b15.log"), "w", encoding="utf-8").write(log(repro_d - 25484, 25484, 0, 0))
        open(os.path.join(d, "kcrate_F_k1.00_b15.log"), "w", encoding="utf-8").write(log(30159, 31154, 2, 0))
        for k, t in zip(S.GRID["Fk"], (40000, 44000, 48000, 52000, 56000)):
            open(os.path.join(d, "kcrate_Fk_k%s_b15.log" % k), "w", encoding="utf-8").write(log(t // 2, t - t // 2, 2, 2))
        for k, t in zip(S.GRID["Dk"], (55000, 58000, 61000, 64000, 67000)):
            open(os.path.join(d, "kcrate_Dk_k%s_b15.log" % k), "w", encoding="utf-8").write(log(t // 2, t - t // 2, 0, 2))
        S.EXP = td
        with contextlib.redirect_stdout(io.StringIO()):
            S.main()
        return open(os.path.join(td, "logs", "E157", "kstar.txt"), encoding="utf-8").read().strip()


got = run_main()
g = got == "Fk k=0.80 총수 48000 (목표 48856, 차 -1.8%) | Dk k=1.30 총수 61000 (목표 61313, 차 -0.5%)"
ok_all &= g
print("%-36s → %s %s" % ("로그 파싱·kstar.txt", got, "✓" if g else "✗"))
got = run_main(repro_d=48857)
g = got == "보정 실패(경로 — k=1 재현 불일치)"
ok_all &= g
print("%-36s → %s %s" % ("k=1 재현 불일치 → 실패", got, "✓" if g else "✗"))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
