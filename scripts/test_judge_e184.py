#!/usr/bin/env python3
"""judge_e184.py 합성 시험 — 정답을 아는 가짜 학습·평가 로그로 판정 범주·경계·조작검증·결측 확인.
실행: python3 scripts/test_judge_e184.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e184 as J  # noqa: E402


def train_log(nB=750, n=1500, chk=((40, 0), (38, 0), (41, 0)), done=True, loaded=False):
    t = "[범주 과제] catrule — 가짜\n"
    if loaded:
        t += "[E184 특징 가중치 적재] x 검증 일치 — y\n"
    for i, (a, b) in enumerate(chk):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, a, b)
    if done:
        t += "[학습] 15ep 완료, 보상 900회 (탐색 주입 100회, ε=0.60)\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
    return t


def eval_log(mods, pushed, trials=96, feat_on=96, loaded=False, sat=False, drop=None):
    t = ""
    if loaded:
        t += "[E184 특징 가중치 적재] x 검증 일치 — y\n"
    for c in "AB":
        for s in ("train", "new"):
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[x]\n" % pushed
    return t


def run(eff=None, per=None, mods=None, miss=False):
    """eff: 기본 효과 {(c,s): 1e-4 정수}; per: {뇌: eff 덮어쓰기}; mods: {(뇌, 종류): 인자}"""
    eff = eff or {("A", "train"): -2000, ("A", "new"): -1500, ("B", "train"): 2000, ("B", "new"): 1500}
    per = per or {}
    mods = mods or {}
    base = {("A", "train"): 150, ("A", "new"): 140, ("B", "train"): 160, ("B", "new"): 150}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E184"))
        for arm in ("U", "F"):
            for b in J.BRAINS:
                e = per.get(b, eff)
                tk = dict(mods.get((b, "train"), {})) if arm == "U" else dict(mods.get((b, "trainF"), {}))
                tk.setdefault("loaded", arm == "F")
                open(os.path.join(td, "logs", "E184", "train_b%d_%s.log" % (b, arm)), "w", encoding="utf-8").write(train_log(**tk))
                for w in ("learn", "none"):
                    if miss and b == 12 and w == "none" and arm == "U":
                        continue
                    m = {k: base[k] + (e[k] if w == "learn" else 0) for k in base}
                    ek = dict(mods.get((b, w), {})) if arm == "U" else dict(mods.get((b, w + "F"), {}))
                    ek.setdefault("loaded", arm == "F")
                    pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                    open(os.path.join(td, "logs", "E184", "ev_b%d_%s_%s.log" % (b, arm, w)), "w", encoding="utf-8").write(eval_log(m, pushed, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


E = lambda aT, aN, bT, bN: {("A", "train"): aT, ("A", "new"): aN, ("B", "train"): bT, ("B", "new"): bN}
cases = [
    ("개념 수준(획득·일반화)", dict(), "판정: 개념 수준(H105)"),
    ("경계: train ±0.10·new ±0.05·비 0.5 정확", dict(eff=E(-1000, -500, 1000, 500)), "판정: 개념 수준(H105)"),
    ("새 사례 비 0.49 → 사례 한정", dict(eff=E(-2000, -980, 2000, 1500)), "판정: 사례 한정(H105-null)"),
    ("새 사례 B 0.0499 → 사례 한정", dict(eff=E(-1000, -800, 1000, 499)), "판정: 사례 한정(H105-null)"),
    ("B train 0.0998 → A만", dict(eff=E(-1000, -800, 998, 499)), "판정: 한 범주만 A(교차)"),
    ("새 사례 효과 작음(+0.04) → 사례 한정", dict(eff=E(-2000, -1500, 2000, 400)), "판정: 사례 한정(H105-null)"),
    ("A만(B train 0.05)", dict(eff=E(-2000, -1500, 500, 300)), "판정: 한 범주만 A(교차)"),
    ("B만(A train −0.05)", dict(eff=E(-500, -300, 2000, 1500)), "판정: 한 범주만 B(같은 쪽)"),
    ("획득 실패", dict(eff=E(-300, -200, 300, 200)), "판정: 획득 실패"),
    ("3/5 개념 → 보류", dict(per={13: E(-300, -200, 300, 200), 14: E(-300, -200, 300, 200)}), "판정: 보류"),
    ("학습 범주 B 수 500(범위 밖)", dict(mods={(11, "train"): {"nB": 500}}), "판정: 보류(조작검증 실패)"),
    ("학습 시행 1400", dict(mods={(11, "train"): {"n": 1400, "nB": 700}}), "판정: 보류(조작검증 실패)"),
    ("학습 특징 점검: 시행 끝 발화 3", dict(mods={(12, "train"): {"chk": ((40, 0), (38, 3), (41, 0))}}), "판정: 보류(조작검증 실패)"),
    ("학습 특징 점검: 행동 창 끝 0", dict(mods={(12, "train"): {"chk": ((0, 0), (38, 0), (41, 0))}}), "판정: 보류(조작검증 실패)"),
    ("학습 [학습] 줄 없음", dict(mods={(10, "train"): {"done": False}}), "판정: 보류(조작검증 실패)"),
    ("U 팔에 특징 가중치 적재(오배선)", dict(mods={(10, "learn"): {"loaded": True}}), "판정: 보류(조작검증 실패)"),
    ("평가 시행 95", dict(mods={(13, "none"): {"trials": 95}}), "판정: 보류(조작검증 실패)"),
    ("평가 특징 켬 0", dict(mods={(13, "learn"): {"feat_on": 0}}), "판정: 보류(조작검증 실패)"),
    ("pushed 7", dict(mods={(14, "learn"): {"pushed": 7}}), "판정: 보류(조작검증 실패)"),
    ("motor 포화", dict(mods={(14, "none"): {"sat": True}}), "판정: 보류(조작검증 실패)"),
    ("RCEVAL 한 줄 결측", dict(mods={(10, "none"): {"drop": ("B", "new")}}), "판정: 보류(조작검증 실패)"),
    ("F 팔 학습에 적재 줄 없음", dict(mods={(12, "trainF"): {"loaded": False}}), "판정: 보류(조작검증 실패)"),
    ("F 팔 평가 적재 줄 없음", dict(mods={(13, "noneF"): {"loaded": False}}), "판정: 보류(조작검증 실패)"),
    ("평가 로그 결측", dict(miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-36s → %-28s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
