#!/usr/bin/env python3
"""E183 보정 선택(사전 규칙 — logs/E183/criteria_fixed.txt [보정]). 뇌 15 O 팔 rawcat 로그 logs/E183/calib/rc_w{W}_b15.log 에서
P1(pre·pre2·post RCCHK 충실도) · P2(형성 전 KC 반응 평균 크기 50~400) · P3(형성 후 ≤ 3 × 형성 전)을 만족하는 W 중 |ln(K_pre/130)| 최소(같으면 작은 W).
분리 지표(S·G)는 읽지 않는다. 없으면: 기본 격자만 있을 때 모든 K_pre < 50 → EXTEND=32,48, 모든 K_pre > 400 → EXTEND=4, 그 밖 W=none; 확장 뒤에도 없으면 W=none.
출력: 화면 + logs/E183/pick.txt 한 줄('W=<값>' | 'EXTEND=<목록>' | 'W=none').
실행: python3 scripts/e183_pick.py (저장소 루트에서)"""
import math
import os
import sys

EXP = "research/experiments"
BASE = (8, 16, 24)
EXT = (4, 32, 48)
TARGET = 130.0


def read(path):
    chk, k = {}, {"pre": [], "post": []}
    done = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCCHK "):
                d = dict(x.split("=", 1) for x in ln.split()[1:])
                chk[d["tag"]] = d
            elif ln.startswith("RCKC "):
                d = dict(x.split("=", 1) for x in ln.split()[1:])
                if d["tag"] in k:
                    k[d["tag"]].append(int(d["n"]))
            elif ln.startswith("=> RAWCAT "):
                done = True
    if not done or set(chk) < {"pre", "pre2", "post"} or len(k["pre"]) != 48 or len(k["post"]) != 48:
        raise ValueError("불완전")
    return chk, sum(k["pre"]) / 48.0, sum(k["post"]) / 48.0


def p1(chk):
    for t in ("pre", "pre2", "post"):
        c = chk[t]
        if not (float(c["feat_active_frac"]) >= 0.90 and int(c["feat_inactive_spikes"]) * 100 <= int(c["feat_active_spikes"])
                and int(c["feat_other_side"]) == 0 and int(c["goodbad"]) == 0):
            return False
    return True


def main():
    rows = {}
    for w in sorted(BASE + EXT):
        p = os.path.join(EXP, "logs", "E183", "calib", "rc_w%d_b15.log" % w)
        if not os.path.exists(p):
            continue
        try:
            chk, kp, kq = read(p)
        except (ValueError, KeyError):
            print("W=%d 로그 불완전 — 후보 제외" % w)
            continue
        ok1, ok2, ok3 = p1(chk), 50.0 <= kp <= 400.0, kq <= 3.0 * kp
        rows[w] = (kp, kq, ok1, ok2, ok3)
        print("W=%d K_pre %.1f K_post %.1f P1 %s P2 %s P3 %s" % (w, kp, kq, ok1, ok2, ok3))
    cand = [w for w, r in rows.items() if r[2] and r[3] and r[4]]
    if cand:
        best = min(cand, key=lambda w: (abs(math.log(rows[w][0] / TARGET)), w))
        res = "W=%d" % best
    elif any(w in rows for w in EXT):
        res = "W=none"
    elif all(w in rows for w in BASE):
        kps = [rows[w][0] for w in BASE]
        res = "EXTEND=32,48" if all(x < 50 for x in kps) else "EXTEND=4" if all(x > 400 for x in kps) else "W=none"
    else:
        res = "W=none"
    print("선택: %s" % res)
    open(os.path.join(EXP, "logs", "E183", "pick.txt"), "w", encoding="utf-8").write(res + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
