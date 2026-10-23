#!/usr/bin/env python3
"""report_e208_overlap 합성 시험 — 정답을 아는 가짜 측정 로그(E208 흥분 0·E207 맥락 없음·흥분 20)로 S 계산·읽기 규칙·점검이 사전 문장대로인지.
실행: python3 test_report_e208_overlap.py (scripts/ 에서)"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_e208_overlap as R  # noqa: E402

PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")


def meas(path, jac, w=0.0, ctx=True, nrcj=12, inactive=700, expo=0, off=0, active=130000):
    """jac[(side, pair)] = (J_pre, J_post) — 없으면 (0.01, 0.01). ctx=False 면 맥락 줄 없음(맥락 없음 측정)."""
    with open(path, "w", encoding="utf-8") as f:
        for pop, tag in (("zz_ctx", "[맥락 입력]"), ("zz_ctx2", "[맥락 입력 2]")):
            f.write("  %s 맥락 전용 집단 %s 200 → KC 좌·우, w=%.2f, p=0.020, 연결 좌 1234·우 1222\n" % (tag, pop, w))
        f.write("[원시 특징 노출] 제시 %d(측정만)\n" % expo)
        if ctx:
            for t in ("pre", "post"):
                f.write("[E207 측정 맥락] tag=%s 맥락 X 켬 — 제시 288 · 켠 맥락 발화 576000 · 끈 맥락 발화 %d · KC 발화 활성 무리 %d · 비활성 무리 %d · 게이팅 있음\n" % (t, off, active, inactive))
        n = 0
        for s in "lr":
            for p in PAIRS:
                if n >= nrcj:
                    break
                a, b = jac.get((s, p), (0.01, 0.01))
                f.write("RCJ side=%s pair=%s J_pre=%.4f J_post=%.4f n_pre=70/73 n_post=67/75\n" % (s, p, a, b))
                n += 1
        f.write("=> RAWCAT done\n")


def jset(v0, v1):
    """AC·BD 두 쪽 모두 (v0, v1) → S = 4·v."""
    return {(s, p): (v0, v1) for s in "lr" for p in ("AC", "BD")}


def build(E, g0, mods):
    L8, L7 = os.path.join(E, "research/experiments/logs/E208"), os.path.join(E, "research/experiments/logs/E207")
    os.makedirs(L8); os.makedirs(L7)
    for b in R.BRAINS:
        o = mods.get(b, {})
        meas(os.path.join(L7, "meas_F_none_b%d.log" % b), jset(0.05, 0.03), w=20.0, ctx=False)
        meas(os.path.join(L7, "meas_F_X_b%d.log" % b), jset(0.52, 0.53), w=20.0)
        if not o.get("missing"):
            meas(os.path.join(L8, "meas_F_X_b%d.log" % b), o.get("jac", g0), w=o.get("w", 0.0), nrcj=o.get("nrcj", 12), inactive=o.get("inactive", 700),
                 expo=o.get("expo", 0), off=o.get("off", 0), active=o.get("active", 130000))


def run(g0, mods=None):
    root = tempfile.mkdtemp(prefix="ro208_")
    try:
        build(root, g0, mods or {})
        R.EXP = os.path.join(root, "research/experiments")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.main()
        return buf.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    ok = tot = 0

    def chk(name, got, want):
        nonlocal ok, tot
        tot += 1
        good = want(got) if callable(want) else got == want
        ok += bool(good)
        print(("OK   " if good else "실패 ") + "%s → %s" % (name, got))

    summ = lambda o: [ln for ln in o.splitlines() if ln.startswith("요약: ")][-1]  # noqa: E731
    line = lambda o, b: [ln for ln in o.splitlines() if ln.startswith("b%d:" % b)][-1]  # noqa: E731
    o = run(jset(0.03, 0.02))
    chk("모두 낮음(S 1200/800) → 공유 사라짐 5/5", summ(o), "요약: 공유 사라짐 5/5 · 공유 남음 0/5 · 가운데 0/5 · 측정 무효 0/5 · 결측 0/5")
    chk("b16 줄 — S·대조 값", line(o, 16).startswith("b16: 흥분 0 맥락 X S 첫/둘째 1200/800 → 공유 사라짐 | 대조 E207 맥락 없음 2000/1200 · 흥분 20 맥락 X 20800/21200"), True)
    o = run(jset(0.03, 0.02), {17: {"jac": jset(0.50, 0.51)}})
    chk("b17 높음(20000/20400) → 공유 남음", ("→ 공유 남음" in line(o, 17), summ(o)), (True, "요약: 공유 사라짐 4/5 · 공유 남음 1/5 · 가운데 0/5 · 측정 무효 0/5 · 결측 0/5"))
    o = run(jset(0.03, 0.02), {18: {"jac": jset(0.03, 0.10)}})
    chk("b18 첫 1200·둘째 4000 → 가운데", "→ 가운데" in line(o, 18), True)
    o = run(jset(0.03, 0.02), {19: {"jac": jset(0.08, 0.08)}, 20: {"jac": jset(0.15, 0.15)}})
    chk("경계 S 3200·6000 → 사라짐·남음", ("→ 공유 사라짐" in line(o, 19), "→ 공유 남음" in line(o, 20)), (True, True))
    o = run(jset(0.03, 0.02), {16: {"w": 20.0}})
    chk("맥락 무게 20 → 측정 무효", line(o, 16).startswith("b16: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {17: {"inactive": 1400}})
    chk("비활성 1400 > E207 흥분 20 활성 130000 의 1% → 측정 무효", line(o, 17).startswith("b17: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {18: {"active": 64000, "inactive": 800}})
    chk("정정 1: 흥분 0 활성 64000·비활성 800(제 비 1.25%) — E207 활성의 1% 이내 → 유효", line(o, 18).startswith("b18: 흥분 0 맥락 X"), True)
    o = run(jset(0.03, 0.02), {19: {"active": 64000, "inactive": 1301}})
    chk("정정 1 경계: 비활성 1301 > 1300(E207 활성 130000 의 1%) → 측정 무효", line(o, 19).startswith("b19: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {18: {"nrcj": 11}})
    chk("RCJ 11줄 → 측정 무효", line(o, 18).startswith("b18: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {19: {"expo": 288}})
    chk("노출 288 → 측정 무효", line(o, 19).startswith("b19: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {20: {"off": 5}})
    chk("끈 맥락 발화 5 → 측정 무효", line(o, 20).startswith("b20: 측정 무효"), True)
    o = run(jset(0.03, 0.02), {16: {"missing": True}})
    chk("E208 로그 없음 → 결측", (line(o, 16).startswith("b16: 결측"), summ(o).endswith("결측 1/5")), (True, True))
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
