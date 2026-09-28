#!/usr/bin/env python3
"""E121 경로 검사: 추적 CSV(--trace-kc-motor)에서 학습 중 블록별 정답률.
단위(P19): greedy_acc_exec = 탐욕 시행 중 correct(=judge exec: 실행 motor 가 교차 쪽)의 비율,
greedy_acc_thr = 탐욕 시행 중 |v|>0.02 이고 부호가 교차 쪽인 비율(이식 평가 정답률과 같은 문턱),
explore_rew = 탐색 시행 중 correct 비율. 블록 = 연속 시행 수(기본 300)."""
import csv
import sys


def block_acc(rows, block=300):
    out = []
    for i in range(0, len(rows), block):
        B = rows[i:i + block]
        g = [r for r in B if int(r["explore"]) == 0]
        x = [r for r in B if int(r["explore"]) == 1]
        def thr(r):
            v = float(r["v"])
            return (r["good_side"] == "left" and v > 0.02) or (r["good_side"] == "right" and v < -0.02)
        out.append({"start": i, "n": len(B), "n_greedy": len(g),
                    "greedy_acc_exec": (sum(int(r["correct"]) for r in g) / len(g)) if g else float("nan"),
                    "greedy_acc_thr": (sum(thr(r) for r in g) / len(g)) if g else float("nan"),
                    "explore_rew": (sum(int(r["correct"]) for r in x) / len(x)) if x else float("nan")})
    return out


if __name__ == "__main__":
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    if not rows:
        print("[측정 확인] 추적 CSV 0행 — 측정 도구 실패"); sys.exit(1)
    print("[측정 확인] 추적 %d행, 탐욕 %d, 탐색 %d" % (len(rows), sum(r["explore"] == "0" for r in rows), sum(r["explore"] == "1" for r in rows)))
    for b in block_acc(rows, int(sys.argv[2]) if len(sys.argv) > 2 else 300):
        print("블록 시행 %4d~%4d: 탐욕 %3d, greedy_acc_exec=%.3f greedy_acc_thr=%.3f explore_rew=%.3f"
              % (b["start"], b["start"] + b["n"] - 1, b["n_greedy"], b["greedy_acc_exec"], b["greedy_acc_thr"], b["explore_rew"]))
