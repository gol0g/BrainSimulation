#!/usr/bin/env python3
"""verify_e194_independent.py 합성 시험 — test_judge_e194.py 와 같은 가짜 로그 생성(복사)으로 독립 대조 확인.
실행: python3 scripts/test_verify_e194.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e194_independent as J  # noqa: E402

SIGN = {("X", "A"): -1, ("X", "B"): +1, ("Y", "A"): +1, ("Y", "B"): -1}


def train_log(nX=4500, nsame=4500, nB=4500, n=9000, fchk=((40, 0),) * 3, cchk=((30, 0, 0),) * 3, done=True, kt=True, catctx=True):
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


def write_exp(td, exp, eff, per, mods, ntrain, miss_b=None):
    os.makedirs(os.path.join(td, "logs", exp), exist_ok=True)
    for b in J.BRAINS:
        e = per.get(b, eff)
        tk = dict(mods.get((b, "train"), {}))
        tk.setdefault("n", ntrain); tk.setdefault("nX", ntrain // 2); tk.setdefault("nsame", ntrain // 2); tk.setdefault("nB", ntrain // 2)
        open(os.path.join(td, "logs", exp, "train_b%d.log" % b), "w", encoding="utf-8").write(train_log(**tk))
        for w in ("learn", "none"):
            for x in "XY":
                if miss_b == b and w == "none" and x == "Y":
                    continue
                m = {(c, s): 120 + (e[(x, c, s)] if w == "learn" else 0) for c in "AB" for s in ("train", "new")}
                ek = dict(mods.get((b, w, x), {}))
                pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                open(os.path.join(td, "logs", exp, "ev_b%d_%s_%s.log" % (b, w, x)), "w", encoding="utf-8").write(eval_log(m, pushed, x, **ek))


def run(eff=None, base=None, per=None, perbase=None, mods=None, miss=False, nobase=False):
    eff = eff or cells(1500, 1500, 1500, 1500)
    base = base or cells(260, 260, 260, 260)
    with tempfile.TemporaryDirectory() as td:
        write_exp(td, "E194", eff, per or {}, mods or {}, 9000, 12 if miss else None)
        if not nobase:
            write_exp(td, "E193", base, perbase or {}, {}, 3000)
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("9,000 획득·자람", dict(), "독립 판정: 학습량 자람(H115) · L4 맥락 의존 범주"),
    ("9,000 = 3,000 → 상한", dict(eff=cells(260, 260, 260, 260)), "독립 판정: 학습량 상한(H115-null) · L4 획득 실패"),
    ("경계 Δ +300 → 자람", dict(eff=cells(560, 560, 560, 560)), "독립 판정: 학습량 자람(H115) · L4 획득 실패"),
    ("Δ +150 → 보류", dict(eff=cells(410, 410, 410, 410)), "독립 판정: 학습량 보류 · L4 획득 실패"),
    ("2 칸 획득 → 부분", dict(eff=cells(1500, 1500, 200, 200)), "독립 판정: 학습량 자람(H115) · L4 부분"),
    ("3,000 기준 결측", dict(nobase=True), "독립 판정: 보류(결측 — 3,000 기준 E193)"),
    ("학습 길이 3,000(잘못)", dict(mods={(10, "train"): {"n": 3000, "nX": 1500, "nsame": 1500, "nB": 1500}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 포화", dict(mods={(13, "learn", "X"): {"sat": True}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 로그 결측", dict(miss=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-22s → %-46s %s" % (name, last, "✓" if good else "✗\n" + out[-1200:]))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
