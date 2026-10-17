#!/usr/bin/env python3
"""posthoc_e201 합성 시험 — test_posthoc_e199 의 가짜 자료(E199·E193 디렉터리)를 E201·E200 으로 옮겨 쓰고 출력 값을 대조.
옮김: logs/traces E199 → E201, E193 → E200. 그래서 E201 = 'E199 판' 값(Δ 300·300·260·−300·100 | 300·−300·50·−50·0, 칸 +0.2), E200 = 'E193 판' 값(Δ 10·−5·…, 칸 +0.02). 좌우 비 E201 6/3 = 2.00·E200 7/3 = 2.33, 교차 몫(활동 1) E201 6/9 = 0.667·E200 7/10 = 0.700.
정답: E200 부호 일치 전체 6/9·강 4/5·약 2/4·가중 0.750, motor_l·motor_r E200 3·6, E199(Δ 10·−5·3·−2·4·6·−1·2·3·1 — 반응 10 개 모두 Δ ≠ 0) 3·7, E199 약 4/5(KC5~9 의 Δ 6·−1·2·3·1 중 양수 4 — 처음에 3/5 로 잘못 셈),
|SXI| 구간: 0.2~0.3 에 약 KC(SXI 0.2), ≥ 0.5 에 강 KC(SXI 0.8) — E200 2/4·4/5, 보상 1700·1573, 평균 DA +0.0012.
실행: python3 scripts/test_posthoc_e201.py"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_e201 as P  # noqa: E402
import test_posthoc_e199 as T9  # noqa: E402


def build(root):
    T9.build(root)
    E = os.path.join(root, "research", "experiments")
    for kind in ("logs", "traces"):
        os.rename(os.path.join(E, kind, "E199"), os.path.join(E, kind, "E201"))
        os.rename(os.path.join(E, kind, "E193"), os.path.join(E, kind, "E200"))
    import numpy as np
    for b in P.BRAINS:
        zp = os.path.join(E, "traces", "E196", "sxi_b%d.npz" % b)          # 활동 가중 교차 몫용: 활동량 1(원 가짜 자료는 0)
        Z = dict(np.load(zp)); Z["tot_l"] = np.ones_like(Z["tot_l"]); Z["tot_r"] = np.ones_like(Z["tot_r"]); np.savez(zp, **Z)
        with open(os.path.join(E, "logs", "E201", "train_b%d.log" % b), "a", encoding="utf-8") as f:
            f.write("[E200 도파민 통계] 시행 3000 · 평균 DA +0.0012 · 평균 |DA| 0.7400 · 기준선 1 · r̄ 끝 0.2700 · τ 100 · r̄₀ 0.25\n")


def main():
    root = tempfile.mkdtemp()
    try:
        build(root)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(root)
        out = buf.getvalue()
    finally:
        shutil.rmtree(root)
    checks = [
        ("보상·평균 DA", "보상 E201 1700 · E200 1573 | 평균 DA E201 +0.0012"),
        ("좌우 수·비·교차 몫", "300 받는 반응 KC motor_l·motor_r E201 3·6 · E200 3·7 · 좌우 비 E201 2.00 · E200 2.33 · 교차 몫(활동) E201 0.667 · E200 0.700"),
        ("부호 일치 E201·E200 약", "부호 일치 E201 전체 6/9(0.667) · 강 4/5(0.800) · 약 2/4(0.500) · 가중 0.750 | E200 약 4/5(0.800)"),
        ("칸", "칸 노출 E201 +0.2000 +0.2000 +0.2000 +0.2000 · 새 +0.1000 +0.1000 +0.1000 +0.1000 | E200 +0.0200 +0.0200 +0.0200 +0.0200"),
        ("좌우 비 중앙값", "좌우 비 중앙값 E201 2.00 · E200 2.33"),
        ("구간 정확도 E201", "E201 부호 정확도 |SXI| 구간 0.0~0.1 0/0(0.000) · 0.1~0.2 0/0(0.000) · 0.2~0.3 10/20(0.500) · 0.3~0.5 0/0(0.000) · 0.5~1.0 20/25(0.800)"),
        ("요약 E201", "E201 노출 평균 +0.2000 범위 +0.2000~+0.2000 ≥0.10 20/20 정답 방향 20/20"),
        ("요약 E200", "E200 노출 평균 +0.0200"),
    ]
    ok = 0
    for nm, s in checks:
        good = s in out
        ok += good
        print("[%s] %s" % ("통과" if good else "실패", nm))
    if ok != len(checks):
        print(out)
    print("합성 %d/%d" % (ok, len(checks)))
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
