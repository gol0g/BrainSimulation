#!/usr/bin/env python3
"""E153 독립 대조 — judge_e153.py 를 쓰지 않고 뇌별 원 로그(logs/E153/dev_b*·ov_b*·train_b*.log)와 추적(traces/E153/tr_b*.npz)에서 다시 계산한다.
형성·겹침 줄은 key=value 로, 학습 효과는 [사전]/[사후] 변조폭 줄에서, 적재는 '[E153 종류 입력 적재] ... 검증 일치' 줄 수로, 동결·규칙 일치는 추적 행마다.
실행: python3 scripts/verify_e153_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
BASE141 = {10: "-0.2384", 11: "-0.2583", 12: "-0.2733", 13: "-0.2514", 14: "-0.2715"}


def q(x):
    if x == "nan":
        return None
    sgn = -1 if x.startswith("-") else 1
    a, b = x.lstrip("+-").split(".")
    return sgn * (int(a) * 10000 + int((b + "0000")[:4]))


def sides(line, head):
    out = {}
    for p in line[len(head):].split("|"):
        d = dict(t.split("=", 1) for t in p.split() if "=" in t)
        if d.get("side") in ("l", "r"):
            out[d["side"]] = d
    return out


def text(path):
    return open(path, encoding="utf-8", errors="replace").read()


def first(txt, head):
    for ln in txt.splitlines():
        if ln.startswith(head):
            return ln
    raise RuntimeError("줄 없음: %s" % head)


def main():
    ok_n = both = nosep = sep_noauth = 0
    for b in BRAINS:
        tdev = text(os.path.join(EXP, "logs", "E153", "dev_b%d.log" % b))
        tov = text(os.path.join(EXP, "logs", "E153", "ov_b%d.log" % b))
        ttr = text(os.path.join(EXP, "logs", "E153", "train_b%d.log" % b))
        dv = sides(first(tdev, "=> KCDEV "), "=> KCDEV ")
        ov = sides(first(tov, "=> KCOVERLAP "), "=> KCOVERLAP ")
        k1 = all(q(dv[k]["sel_med"]) is not None and q(dv[k]["sel_med"]) >= 8000 and float(dv[k]["relerr"]) <= 1e-6 for k in "lr")
        n_ov = sum(1 for ln in tov.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
        n_tr = sum(1 for ln in ttr.splitlines() if ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln)
        k2 = n_ov >= 1 and n_tr >= 2
        refl = re.findall(r"^\[반사가중치\] good_food_to_motor_[lr]\s+n=\d+ w_mean (\S+)→(\S+)", ttr, re.M)
        R = np.load(os.path.join(EXP, "traces", "E153", "tr_b%d.npz" % b))["rows"]
        agree = tot = 0
        for i in range(len(R)):
            if R[i, 6] < 0:
                continue
            tot += 1
            agree += int((R[i, 7] == 1) == (R[i, 6] != R[i, 2]))
        res = float(np.abs(R[:, 21:25] - ((11.0 / 12.0) ** 20) * R[:, 13:17]).sum() / np.abs(R[:, 13:17]).sum())
        pre = abs(float(R[:, 17:21].sum())) / max(abs(float(R[:, 12].sum())), 1e-12)
        k3 = len(R) == 500 and tot > 0 and agree == tot and res <= 1e-3 and pre <= 1e-3 and len(refl) == 2 and all(r == ("0.0000", "0.0000") for r in refl)
        ok = k1 and k2 and k3
        ok_n += ok
        pre_m = re.search(r"^\[사전\].*변조폭 ([-+]?\d+\.\d{4})", ttr, re.M).group(1)
        post_m = re.search(r"^\[사후\].*변조폭 ([-+]?\d+\.\d{4})", ttr, re.M).group(1)
        e = q(post_m) - q(pre_m)
        Jl, Jr = q(ov["l"]["jac"]), q(ov["r"]["jac"])
        sep = Jl is not None and Jr is not None and Jl <= 2500 and Jr <= 2500
        auth = 3 * e <= 2 * q(BASE141[b])
        both += sep and auth; nosep += not sep; sep_noauth += sep and not auth
        print("b%d 형성 %s/%s J %s/%s 효과 %+d(기본 %s) 적재 %d·%d 동결 %.1e 일치 %d/%d | K1 %s K2 %s K3 %s | %s %s"
              % (b, dv["l"]["sel_med"], dv["r"]["sel_med"], ov["l"]["jac"], ov["r"]["jac"], e, BASE141[b], n_ov, n_tr, res, agree, tot,
                 "✓" if k1 else "✗", "✓" if k2 else "✗", "✓" if k3 else "✗", "분리" if sep else "-", "권한" if auth else "-"))
    print("조작검증 %d/5 | 둘 다 %d/5 분리 아님 %d/5 분리·권한 아님 %d/5" % (ok_n, both, nosep, sep_noauth))
    if ok_n < 5:
        v = "보류(조작검증 실패)"
    elif both >= 4:
        v = "형성 성공(H076)"
    elif nosep >= 4:
        v = "분리 실패(H076-null)"
    elif sep_noauth >= 4:
        v = "권한 상실(H076-auth)"
    else:
        v = "보류"
    print("독립 판정: %s" % v)


if __name__ == "__main__":
    sys.exit(main())
