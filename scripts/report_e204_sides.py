#!/usr/bin/env python3
"""E204 보고(판정 밖) — 사전등록 4절 '(보고) 300 받는 반응 KC 좌우 비' 가 judge_e204·posthoc_e204 출력에 빠져 있어, 결과 뒤(2026-10-18)
E200 판정기와 같은 함수(judge_e200.side_counts — 반응 KC 를 학습 차 Δ 부호로 motor_l·motor_r 쪽에 셈, judge_e199.weight_checks 의 Δ)로 계산한다.
반응 KC 집합은 traces/E196/sxi_b{B}.npz(세계 18300 에서 잰 것) — 그래서 G18 과 같은 뇌 E200 만(세계 28300 인 GN·UN 은 반응 집합이 없어 계산하지 않음).
실행: python3 scripts/report_e204_sides.py [저장소 루트]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e199 as J9  # noqa: E402
import judge_e200 as J0  # noqa: E402

BRAINS = (10, 11, 12, 13, 14)


def main(root="."):
    E = os.path.join(root, "research", "experiments")
    print("[E204 보고 — 판정 밖, 결과 뒤 계산(사전등록 보고 항목)] 반응 KC(E196 세계 18300) 를 학습 차 Δ 부호로 motor_l·motor_r 에 셈 — G18 대 같은 뇌 E200")
    for b in BRAINS:
        sn = os.path.join(E, "traces", "E176", "snap", "conn_b%d.npz" % b)
        npz = os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)
        out = []
        for lab, wf, cmp_ in (("G18", os.path.join(E, "traces", "E204", "w_G18_b%d.npz" % b), os.path.join(E, "traces", "E200", "w_b%d.npz" % b)),
                              ("E200", os.path.join(E, "traces", "E200", "w_b%d.npz" % b), os.path.join(E, "traces", "E199", "w_b%d.npz" % b))):
            D = J9.weight_checks(wf, sn, cmp_)[4]
            to = J0.side_counts(D, npz)
            out.append("%s motor_l %d · motor_r %d · 비(큰/작은) %.2f" % (lab, to["l"], to["r"], max(to["l"], to["r"]) / max(min(to["l"], to["r"]), 1)))
        print("b%d %s" % (b, " | ".join(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
