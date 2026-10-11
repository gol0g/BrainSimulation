#!/usr/bin/env python3
"""judge_e185.py 합성 시험 — 정답을 아는 가짜 학습·평가 로그로 확증·용량-반응·경계·조작검증·결측 확인.
실행: python3 scripts/test_judge_e185.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e185 as J  # noqa: E402


def train_log(n, nB, chk=((40, 0), (38, 0), (41, 0)), done=True, kt=True, fw=False):
    t = "[범주 과제] catrule — 가짜\n"
    if kt:
        t += "[E153 종류 입력 적재] x 검증 일치 — y\n"
    if fw:
        t += "[E184 특징 가중치 적재] x 검증 일치 — y\n"
    for i, (a, b) in enumerate(chk):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, a, b)
    if done:
        t += "[학습] 가짜 완료, 보상 300회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
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
D500 = E(-1200, -800, 1300, 700)
D1500 = E(-1800, -1400, 2000, 1200)


def run(e500=D500, e1500=D1500, per=None, mods=None, miss=False):
    per = per or {}
    mods = mods or {}
    base = {("A", "train"): 150, ("A", "new"): 140, ("B", "train"): 160, ("B", "new"): 150}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E185"))
        for b in J.BRAINS:
            p5, p15 = per.get(b, (e500, e1500))
            for T, n, nB in (("500", 500, 250), ("1500", 1500, 750)):
                kw = dict(n=n, nB=nB); kw.update(mods.get((b, "train" + T), {}))
                open(os.path.join(td, "logs", "E185", "train_b%d_%s.log" % (b, T)), "w", encoding="utf-8").write(train_log(**kw))
            for w, e in (("learn500", p5), ("learn1500", p15), ("none", None)):
                if miss and b == 18 and w == "learn500":
                    continue
                m = {k: base[k] + (e[k] if e else 0) for k in base}
                ek = dict(mods.get((b, w), {}))
                pushed = ek.pop("pushed", 0 if w == "none" else 8)
                open(os.path.join(td, "logs", "E185", "ev_b%d_%s.log" % (b, w)), "w", encoding="utf-8").write(eval_log(m, pushed, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("확증·용량-반응", dict(), "판정: 확증·용량-반응(개념 조건 1~4, 범주 수준)"),
    ("용량-반응 경계 ±0.02 정확", dict(e500=E(-1200, -1200, 1300, 1000), e1500=E(-1800, -1400, 2000, 1200)), "판정: 확증·용량-반응(개념 조건 1~4, 범주 수준)"),
    ("용량-반응 B 증가 0.0199 → 확증만", dict(e500=E(-1200, -800, 1300, 1001), e1500=E(-1800, -1400, 2000, 1200)), "판정: 확증(용량-반응 미확인)"),
    ("500 이 더 큼 → 확증만", dict(e500=E(-2000, -1600, 2100, 1400), e1500=D1500), "판정: 확증(용량-반응 미확인)"),
    ("1,500 사례 한정", dict(e1500=E(-1800, -600, 2000, 1200)), "판정: 확증 실패(사례 한정)"),
    ("1,500 A만", dict(e1500=E(-1800, -1400, 500, 300)), "판정: 확증 실패(한 범주만)"),
    ("1,500 B만", dict(e1500=E(-500, -300, 2000, 1200)), "판정: 확증 실패(한 범주만)"),
    ("1,500 획득 실패", dict(e1500=E(-300, -200, 300, 200)), "판정: 확증 실패(획득 실패)"),
    ("3/5 확증 → 보류", dict(per={19: (D500, E(-300, -200, 300, 200)), 20: (D500, E(-300, -200, 300, 200))}), "판정: 보류"),
    ("확증 4/5·용량 3/5 → 확증만", dict(per={19: (D500, E(-300, -200, 300, 200)), 20: (E(-2000, -1600, 2100, 1400), D1500)}), "판정: 확증(용량-반응 미확인)"),
    ("학습 500 범주 B 150", dict(mods={(16, "train500"): {"nB": 150}}), "판정: 보류(조작검증 실패)"),
    ("학습 1500 시행 1400", dict(mods={(17, "train1500"): {"n": 1400, "nB": 700}}), "판정: 보류(조작검증 실패)"),
    ("학습 점검 시행 끝 발화", dict(mods={(18, "train1500"): {"chk": ((40, 0), (38, 2), (41, 0))}}), "판정: 보류(조작검증 실패)"),
    ("학습 종류 적재 줄 없음", dict(mods={(19, "train500"): {"kt": False}}), "판정: 보류(조작검증 실패)"),
    ("학습에 특징 가중치 적재(오배선)", dict(mods={(20, "train1500"): {"fw": True}}), "판정: 보류(조작검증 실패)"),
    ("평가 시행 95", dict(mods={(16, "none"): {"trials": 95}}), "판정: 보류(조작검증 실패)"),
    ("평가 pushed 0(학습)", dict(mods={(17, "learn1500"): {"pushed": 0}}), "판정: 보류(조작검증 실패)"),
    ("평가 포화", dict(mods={(18, "learn500"): {"sat": True}}), "판정: 보류(조작검증 실패)"),
    ("평가 종류 적재 줄 없음", dict(mods={(19, "none"): {"kt": False}}), "판정: 보류(조작검증 실패)"),
    ("RCEVAL 한 줄 결측", dict(mods={(20, "learn1500"): {"drop": ("A", "new")}}), "판정: 보류(조작검증 실패)"),
    ("평가 로그 결측", dict(miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "판정: 보류" or last == "판정: 보류")
    ok_all &= good
    print("%-30s → %-44s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
