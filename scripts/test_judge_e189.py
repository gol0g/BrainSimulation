#!/usr/bin/env python3
"""judge_e189.py 합성 시험 — 둘째 쌍 학습 뒤(E189)·첫 쌍 기준(E184 U) 가짜 로그로 판정 범주·경계·전제·조작검증·결측 확인.
실행: python3 scripts/test_judge_e189.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e189 as J  # noqa: E402


def train_log(start=1500, n2=(1500, 1500, 3000), n=3000, nB=1500, chk=((40, 0),) * 3, done=True, kt=True):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    t += "[범주 과제] catrule — 가짜\n"
    for i in range(3):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, chk[i][0], chk[i][1])
    if start is not None:
        t += "[둘째 범주 쌍] 시행 %d 부터 범주 C(같은 쪽)·D(교차) — C 는 A 와, D 는 B 와 원형 단위 공유\n" % start
    if done:
        t += "[학습] 30ep 완료, 보상 1700회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
    if n2 is not None:
        t += "[둘째 범주 쌍] 둘째 쌍 시행 %d (시행 %d 부터, 전체 %d)\n" % n2
    return t


def eval_log(mods, pushed, cats="ABCD", trials=96, feat_on=96, kt=True, sat=False, drop=None, fw=False):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    if fw:
        t += "[E184 특징 가중치 적재] 가짜\n"
    for c in cats:
        for s in ("train", "new"):
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[x]\n" % pushed
    return t


def E(aT, bT, cT, dT, aN=None, bN=None, cN=None, dN=None):
    return {("A", "train"): aT, ("A", "new"): aN if aN is not None else aT // 2, ("B", "train"): bT, ("B", "new"): bN if bN is not None else bT // 2,
            ("C", "train"): cT, ("C", "new"): cN if cN is not None else cT // 2, ("D", "train"): dT, ("D", "new"): dN if dN is not None else dT // 2}


PRE = {("A", "train"): -1800, ("A", "new"): -1500, ("B", "train"): 2000, ("B", "new"): 1400}   # 첫 쌍 기준(E184 U 꼴)
GOOD = E(-1500, 1600, 1400, -1300)     # 유지 몫 0.83·0.80, 둘째 쌍 획득


def run(post=GOOD, per=None, mods=None, miss=False, nopre=False, pre=None):
    per = per or {}
    mods = mods or {}
    pre = pre or {}
    base = {(c, s): 150 + 10 * i for i, (c, s) in enumerate((c, s) for c in "ABCD" for s in ("train", "new"))}
    with tempfile.TemporaryDirectory() as td:
        for d in ("E189", "E184"):
            os.makedirs(os.path.join(td, "logs", d))
        for b in J.BRAINS:
            p = per.get(b, post)
            open(os.path.join(td, "logs", "E189", "train_b%d.log" % b), "w", encoding="utf-8").write(train_log(**mods.get((b, "train"), {})))
            for w in ("learn", "none"):
                if miss and b == 12 and w == "none":
                    continue
                m = {k: base[k] + (p[k] if w == "learn" else 0) for k in base}
                ek = dict(mods.get((b, w), {}))
                pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                open(os.path.join(td, "logs", "E189", "ev_b%d_%s.log" % (b, w)), "w", encoding="utf-8").write(eval_log(m, pushed, **ek))
                if not (nopre and b == 13):
                    pr = pre.get(b, PRE)
                    mr = {k: base[k] + (pr[k] if w == "learn" else 0) for k in PRE}
                    open(os.path.join(td, "logs", "E184", "ev_b%d_U_%s.log" % (b, w)), "w", encoding="utf-8").write(
                        eval_log(mr, 4 if w == "learn" else 0, cats="AB"))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("유지", dict(), "판정: 유지(H110)"),
    ("경계: r 정확히 0.5·둘째 쌍 ±0.10", dict(post=E(-900, 1000, 1000, -1000)), "판정: 유지(H110)"),
    ("A r 0.4994 → 간섭", dict(post=E(-899, 1600, 1400, -1300)), "판정: 간섭(H110-null)"),
    ("B r 0.4995 → 간섭", dict(post=E(-1500, 999, 1400, -1300)), "판정: 간섭(H110-null)"),
    ("첫 쌍 뒤집힘 → 간섭", dict(post=E(1200, -1100, 1400, -1300)), "판정: 간섭(H110-null)"),
    ("유지 3/5·간섭 2/5 → 보류", dict(per={13: E(-300, 400, 1400, -1300), 14: E(-200, 300, 1400, -1300)}), "판정: 보류"),
    ("전제: C 0.0999 → 획득 아님(3/5) → 전제 미달", dict(per={13: E(-1500, 1600, 999, -1300), 14: E(-1500, 1600, 1400, -999)}), "판정: 보류(전제 미달"),
    ("전제 4/5 → 판정 진행(유지)", dict(per={14: E(-1500, 1600, 500, -1300)}), "판정: 유지(H110)"),
    ("첫 쌍 기준 미달 뇌는 셈 밖(4/5 유지)", dict(pre={14: {("A", "train"): -800, ("A", "new"): -500, ("B", "train"): 2000, ("B", "new"): 1400}}), "판정: 유지(H110)"),
    ("첫 쌍 기준 미달 2 뇌 → 유지 3 → 보류", dict(pre={13: {("A", "train"): -800, ("A", "new"): -500, ("B", "train"): 2000, ("B", "new"): 1400},
                                                   14: {("A", "train"): -1800, ("A", "new"): -500, ("B", "train"): 900, ("B", "new"): 1400}}), "판정: 보류"),
    ("둘째 쌍 시작 줄 없음", dict(mods={(10, "train"): {"start": None}}), "판정: 보류(조작검증 실패)"),
    ("둘째 쌍 시행 수 1400", dict(mods={(11, "train"): {"n2": (1400, 1500, 3000)}}), "판정: 보류(조작검증 실패)"),
    ("특징 점검 시행 끝 발화", dict(mods={(12, "train"): {"chk": ((40, 0), (40, 2), (40, 0))}}), "판정: 보류(조작검증 실패)"),
    ("평가 포화", dict(mods={(13, "learn"): {"sat": True}}), "판정: 보류(조작검증 실패)"),
    ("평가 pushed 0(학습)", dict(mods={(14, "learn"): {"pushed": 0}}), "판정: 보류(조작검증 실패)"),
    ("평가 줄 결측(D new)", dict(mods={(10, "none"): {"drop": ("D", "new")}}), "판정: 보류(조작검증 실패)"),
    ("평가에 특징 가중치 적재", dict(mods={(11, "learn"): {"fw": True}}), "판정: 보류(조작검증 실패)"),
    ("첫 쌍 기준 결측", dict(nopre=True), "판정: 보류(결측"),
    ("평가 로그 결측", dict(miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-34s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
