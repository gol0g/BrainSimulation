#!/usr/bin/env python3
"""judge_e188.py 합성 시험 — 반전 뒤·반전 전(E184 U) 가짜 로그로 판정 범주·경계·조작검증·결측 확인.
실행: python3 scripts/test_judge_e188.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e188 as J  # noqa: E402


def train_log(start=1500, nrev=(1500, 1500, 3000), n=3000, nB=1500, chk=((40, 0),) * 3, done=True, kt=True):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    t += "[범주 과제] catrule — 가짜\n"
    for i in range(3):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, chk[i][0], chk[i][1])
    if start is not None:
        t += "[범주 반전] 시행 %d 부터 정답 A → 같은 쪽 · B → 교차\n" % start
    if done:
        t += "[학습] 30ep 완료, 보상 1700회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
    if nrev is not None:
        t += "[범주 반전] 반전 시행 %d (시행 %d 부터, 전체 %d)\n" % nrev
    return t


def eval_log(mods, pushed, trials=96, feat_on=96, kt=True, sat=False, drop=None):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    for c in "AB":
        for s in ("train", "new"):
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[x]\n" % pushed
    return t


E = lambda aT, aN, bT, bN: {("A", "train"): aT, ("A", "new"): aN, ("B", "train"): bT, ("B", "new"): bN}
PRE = E(-1800, -1500, 2000, 1400)      # 반전 전(E184 U 꼴)


def run(post=E(1500, 1200, -1600, -1100), per=None, mods=None, miss=False, nopre=False):
    per = per or {}
    mods = mods or {}
    base = {("A", "train"): 150, ("A", "new"): 140, ("B", "train"): 160, ("B", "new"): 150}
    with tempfile.TemporaryDirectory() as td:
        for d in ("E188", "E184"):
            os.makedirs(os.path.join(td, "logs", d))
        for b in J.BRAINS:
            p = per.get(b, post)
            open(os.path.join(td, "logs", "E188", "train_b%d.log" % b), "w", encoding="utf-8").write(train_log(**mods.get((b, "train"), {})))
            for w in ("learn", "none"):
                if miss and b == 12 and w == "none":
                    continue
                m = {k: base[k] + (p[k] if w == "learn" else 0) for k in base}
                ek = dict(mods.get((b, w), {}))
                pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                open(os.path.join(td, "logs", "E188", "ev_b%d_%s.log" % (b, w)), "w", encoding="utf-8").write(eval_log(m, pushed, **ek))
                if not (nopre and b == 13):
                    mr = {k: base[k] + (PRE[k] if w == "learn" else 0) for k in base}
                    open(os.path.join(td, "logs", "E184", "ev_b%d_U_%s.log" % (b, w)), "w", encoding="utf-8").write(eval_log(mr, 8 if w == "learn" else 0))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("반전 성공", dict(), "판정: 반전 성공(H109)"),
    ("경계: ±0.10·새 ±0.05·비 0.5", dict(post=E(1000, 500, -1000, -500)), "판정: 반전 성공(H109)"),
    ("새 비 0.49 → 사례 한정", dict(post=E(1500, 735, -1600, -1100)), "판정: 반전(사례 한정)"),
    ("A 0.0999 → 부분 반전", dict(post=E(999, 700, -1600, -1100)), "판정: 부분 반전"),
    ("0 근처(Δ 큼) → 부분 반전", dict(post=E(0, 0, 0, 0)), "판정: 부분 반전"),
    ("고착(반전 전과 같음)", dict(post=E(-1800, -1500, 2000, 1400)), "판정: 고착(H109-null)"),
    ("고착 경계 |Δ| 499", dict(post=E(-1301, -1500, 1501, 1400)), "판정: 고착(H109-null)"),
    ("3/5 → 보류", dict(per={13: E(-1800, -1500, 2000, 1400), 14: E(-1700, -1400, 1900, 1300)}), "판정: 보류"),
    ("학습 반전 줄 없음", dict(mods={(10, "train"): {"start": None}}), "판정: 보류(조작검증 실패)"),
    ("반전 시행 수 1400", dict(mods={(11, "train"): {"nrev": (1400, 1500, 3000)}}), "판정: 보류(조작검증 실패)"),
    ("특징 점검 시행 끝 발화", dict(mods={(12, "train"): {"chk": ((40, 0), (40, 2), (40, 0))}}), "판정: 보류(조작검증 실패)"),
    ("평가 포화", dict(mods={(13, "learn"): {"sat": True}}), "판정: 보류(조작검증 실패)"),
    ("평가 pushed 0(학습)", dict(mods={(14, "learn"): {"pushed": 0}}), "판정: 보류(조작검증 실패)"),
    ("반전 전 기준 결측", dict(nopre=True), "판정: 보류(결측"),
    ("평가 로그 결측", dict(miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-28s → %-30s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
