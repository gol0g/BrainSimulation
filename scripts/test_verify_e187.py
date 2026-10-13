#!/usr/bin/env python3
"""verify_e187_independent.py 합성 시험 — test_judge_e187.py 와 같은 가짜 로그 생성(복사)으로 독립 대조 확인.
실행: python3 scripts/test_verify_e187.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e187_independent as J  # noqa: E402

SIGN = {("X", "A"): -1, ("X", "B"): +1, ("Y", "A"): +1, ("Y", "B"): -1}


def train_log(nX=1500, nsame=1500, nB=1500, n=3000, fchk=((40, 0),) * 3, cchk=((30, 0, 0),) * 3, done=True, kt=True, catctx=True):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    if catctx:
        t += "[맥락 범주 과제] catctx — 가짜\n"
    for i in range(3):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, fchk[i][0], fchk[i][1])
        t += "[맥락 범주 점검] 시행 %d 맥락 X 범주 A 정답 교차 — 행동 창 끝 켠 맥락 발화 %d · 끈 맥락 발화 %d · 시행 끝 두 맥락 발화 %d\n" % (i, *cchk[i])
    if done:
        t += "[학습] 30ep 완료, 보상 1600회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
    t += "[맥락 범주 과제] 시행 %d 중 맥락 X %d · Y %d · 정답 같은 쪽 %d · 교차 %d\n" % (n, nX, n - nX, nsame, n - nsame)
    return t


def eval_log(mods, pushed, ctx, trials=96, feat_on=96, ctx_on=96, ctxlab=None, kt=True, sat=False, drop=None):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    for c in "AB":
        for s in ("train", "new"):
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d ctx=%s ctx_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, ctxlab or ctx, ctx_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[x]\n" % pushed
    return t


def cells(xa, xb, ya, yb, ratio=0.7):
    """규칙 방향 효과(노출) → 칸별 e(노출·새)"""
    a = {("X", "A"): xa, ("X", "B"): xb, ("Y", "A"): ya, ("Y", "B"): yb}
    e = {}
    for (x, c), v in a.items():
        e[(x, c, "train")] = SIGN[(x, c)] * v
        e[(x, c, "new")] = SIGN[(x, c)] * int(round(v * ratio))
    return e


def run(eff=None, per=None, mods=None, miss=False):
    eff = eff or cells(1500, 1500, 1500, 1500)
    per = per or {}
    mods = mods or {}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E187"))
        for b in J.BRAINS:
            e = per.get(b, eff)
            open(os.path.join(td, "logs", "E187", "train_b%d.log" % b), "w", encoding="utf-8").write(train_log(**mods.get((b, "train"), {})))
            for w in ("learn", "none"):
                for x in "XY":
                    if miss and b == 12 and w == "none" and x == "Y":
                        continue
                    m = {(c, s): 120 + (e[(x, c, s)] if w == "learn" else 0) for c in "AB" for s in ("train", "new")}
                    ek = dict(mods.get((b, w, x), {}))
                    pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                    open(os.path.join(td, "logs", "E187", "ev_b%d_%s_%s.log" % (b, w, x)), "w", encoding="utf-8").write(eval_log(m, pushed, x, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("맥락 의존 범주", dict(), "독립 판정: 맥락 의존 범주(H108)"),
    ("경계 정확", dict(eff=cells(1000, 1000, 1000, 1000, ratio=0.5)), "독립 판정: 맥락 의존 범주(H108)"),
    ("새 비 0.49 → 사례 한정", dict(eff=cells(1500, 1500, 1500, 1500, ratio=0.49)), "독립 판정: 맥락 의존(사례 한정)"),
    ("맥락 X 만 → 부분", dict(eff=cells(1500, 1500, 200, 200)), "독립 판정: 부분"),
    ("효과 0 → 획득 실패", dict(eff=cells(0, 0, 0, 0)), "독립 판정: 획득 실패(H108-null)"),
    ("3/5 → 보류", dict(per={13: cells(1500, 1500, 200, 200), 14: cells(0, 0, 0, 0)}), "독립 판정: 보류"),
    ("맥락 점검 끈 맥락 발화", dict(mods={(12, "train"): {"cchk": ((30, 2, 0), (30, 0, 0), (30, 0, 0))}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 ctx 표지 틀림", dict(mods={(10, "learn", "X"): {"ctxlab": "Y"}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 포화", dict(mods={(12, "learn", "Y"): {"sat": True}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 로그 결측", dict(miss=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-24s → %-36s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
