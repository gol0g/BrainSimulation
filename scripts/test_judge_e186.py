#!/usr/bin/env python3
"""judge_e186.py 합성 시험 — 규칙 방향 효과를 정한 가짜 평가 로그로 판정 범주·경계(±0.005)·재현 표기·조작검증·결측 확인.
실행: python3 scripts/test_judge_e186.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e186 as J  # noqa: E402

SETS = ("train", "new", "proto", "far")


def eval_log(mods, pushed, trials=96, feat_on=96, kt=True, fw=False, sat=False, drop=None):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    if fw:
        t += "[E184 특징 가중치 적재] x 검증 일치\n"
    for c in "AB":
        for s in SETS:
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=초기값 kc_means[x]\n" % pushed
    return t


def A(proto, train, new, far):
    """규칙 방향 효과(양수 = 정답 방향) → 두 범주 같은 값"""
    return {"proto": proto, "train": train, "new": new, "far": far}


def run(eff=A(2200, 2000, 1500, 900), per=None, mods=None, miss=False):
    per = per or {}
    mods = mods or {}
    base = {(c, s): 100 + 10 * i for i, (c, s) in enumerate((c, s) for c in "AB" for s in SETS)}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E186"))
        for brains in J.SAMPLES.values():
            for b in brains:
                a = per.get(b, eff)
                for w in ("learn", "none"):
                    if miss and b == 18 and w == "none":
                        continue
                    m = {}
                    for (c, s), v0 in base.items():
                        e = (-a[s] if c == "A" else a[s]) if w == "learn" else 0
                        m[(c, s)] = v0 + e
                    ek = dict(mods.get((b, w), {}))
                    pushed = ek.pop("pushed", 8 if w == "learn" else 0)
                    open(os.path.join(td, "logs", "E186", "ev_b%d_%s.log" % (b, w)), "w", encoding="utf-8").write(eval_log(m, pushed, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


R10 = (10, 11, 12, 13, 14)
cases = [
    ("원형 추상화(재현)", dict(), "판정: 원형 추상화 (재현)"),
    ("경계: 원형 = 노출 − 0.005 → 강", dict(eff=A(1950, 2000, 1500, 900)), "판정: 원형 추상화 (재현)"),
    ("원형 = 노출 − 0.0051 → 우위·기울기", dict(eff=A(1949, 2000, 1500, 900)), "판정: 원형 우위·기울기 (재현)"),
    ("경계: 원형 = 새 + 0.005 → 우위", dict(eff=A(1550, 2000, 1500, 900)), "판정: 원형 우위·기울기 (재현)"),
    ("원형 = 새 + 0.0049 → 기울기만", dict(eff=A(1549, 2000, 1500, 900)), "판정: 기울기만 (재현)"),
    ("기울기 경계 새 = 먼 + 0.005", dict(eff=A(2200, 2000, 950, 900)), "판정: 원형 추상화 (재현)"),
    ("기울기 없음(새 = 먼) → 보류", dict(eff=A(2200, 2000, 900, 900)), "판정: 보류 (재현)"),
    ("원형 열세", dict(eff=A(1000, 2000, 1500, 1500)), "판정: 원형 열세 (재현)"),
    ("효과 모두 반대 방향 → 보류", dict(eff=A(-100, -200, -300, -900)), "판정: 보류 (재현)"),
    ("재현 표본만 다름", dict(per={b: A(1000, 2000, 1500, 1500) for b in R10}), "판정: 원형 추상화 (재현 표본: 원형 열세)"),
    ("주 표본 기울기 3/5 → 보류", dict(per={19: A(2200, 2000, 900, 900), 20: A(2200, 2000, 900, 900)}), "판정: 보류 (재현 표본: 원형 추상화)"),
    ("RCEVAL 결측(먼 사례)", dict(mods={(17, "learn"): {"drop": ("B", "far")}}), "판정: 보류(조작검증 실패)"),
    ("특징 켬 95", dict(mods={(16, "none"): {"feat_on": 95}}), "판정: 보류(조작검증 실패)"),
    ("pushed 0(학습)", dict(mods={(11, "learn"): {"pushed": 0}}), "판정: 보류(조작검증 실패)"),
    ("종류 적재 없음", dict(mods={(12, "none"): {"kt": False}}), "판정: 보류(조작검증 실패)"),
    ("특징 가중치 적재(오배선)", dict(mods={(13, "learn"): {"fw": True}}), "판정: 보류(조작검증 실패)"),
    ("포화", dict(mods={(20, "learn"): {"sat": True}}), "판정: 보류(조작검증 실패)"),
    ("결측", dict(miss=True), "판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want)
    ok_all &= good
    print("%-30s → %-46s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
