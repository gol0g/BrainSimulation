#!/usr/bin/env python3
"""verify_e190_independent.py 합성 시험 — test_judge_e190.py 와 같은 가짜 로그 생성(복사)으로 독립 대조 확인.
실행: python3 scripts/test_verify_e190.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e190_independent as J  # noqa: E402

PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")


def form_log(expo=816, cats="A,B,C,D", rcw=((150000.0, 1), (151000.0, 1)), drop_pair=None, rawcat=1, npres=288, gsame=1):
    t = "[원시 특징 세계] 가짜\n"
    for tag in ("pre", "pre2", "post"):
        t += ("RCCHK tag=%s npres=%d feat_active_frac=1.0000 feat_inactive_spikes=0 feat_active_spikes=115200 feat_other_side=0 goodbad=0 g_same=%d base_l=1 base_r=1\n"
              % (tag, npres, gsame))
    for sd, (d, g) in zip("lr", rcw):
        t += "RCW side=%s nsyn=60000 g_mean0=24.0000 g_mean1=21.4000 dg_abs=%.4f frac_changed=0.3500 wsel0=0.6600 wsel1=0.6600 kc_in_mean0=480.0 kc_in_mean1=430.0 g0_init=%d\n" % (sd, d, g)
    t += "[원시 특징 노출] 제시 %d(주기 17 × 사례 6 × 범주 %d × 쪽 2), 노출 eta 0.02, Oja 켬, 도파민 호출 없음%s\n" % (
        expo, len(cats.split(",")), "" if cats == "A,B" else " — 노출 범주 %s" % cats)
    for sd in "lr":
        for p in PAIRS:
            if drop_pair == (sd, p):
                continue
            t += "RCJ side=%s pair=%s J_pre=0.3000 J_post=0.0200 n_pre=120/120 n_post=90/90\n" % (sd, p)
    t += "=> RAWCAT 가짜\n" * rawcat
    return t


def train_log(phase2, start=1500, n2=(1500, 1500, 3000), n=None, nB=None, chk=((120, 0),) * 3, done=True, kt=True, fw=True, extra=""):
    n = n if n is not None else (3000 if phase2 else 1500)
    nB = nB if nB is not None else n // 2
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    t += "[E184 특징 가중치 적재] f.npz 검증 일치 — z\n" if fw else ""
    t += "[범주 과제] catrule — 가짜\n"
    for i in range(3):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 %d · 시행 끝 %d\n" % (i, chk[i][0], chk[i][1])
    if phase2 and start is not None:
        t += "[둘째 범주 쌍] 시행 %d 부터 범주 C(같은 쪽)·D(교차) — C 는 A 와, D 는 B 와 원형 단위 공유\n" % start
    if done:
        t += "[학습] 가짜 완료, 보상 900회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, nB, n - nB)
    if phase2 and n2 is not None:
        t += "[둘째 범주 쌍] 둘째 쌍 시행 %d (시행 %d 부터, 전체 %d)\n" % n2
    return t + extra


def eval_log(mods, pushed, trials=96, feat_on=96, kt=True, fw=True, sat=False, drop=None):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n" if kt else ""
    t += "[E184 특징 가중치 적재] f.npz 검증 일치 — z\n" if fw else ""
    for c in "ABCD":
        for s in ("train", "new"):
            if drop == (c, s):
                continue
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=%d feat_on=%d | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, trials, feat_on, mot))
    t += "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=f.npz kc_means[x]\n" % pushed
    return t


def E(aT, bT, cT, dT):
    return {("A", "train"): aT, ("A", "new"): aT // 2, ("B", "train"): bT, ("B", "new"): bT // 2,
            ("C", "train"): cT, ("C", "new"): cT // 2, ("D", "train"): dT, ("D", "new"): dT // 2}


RBASE = E(-1800, 2000, -600, 700)        # 기준(첫 쌍 뒤): C 는 A 처럼 교차 쪽 전이, D 는 B 처럼
PGOOD = E(-1500, 1600, 1400, -1300)      # 둘째 쌍 뒤: 유지 몫 0.83·0.80, 획득


def run(post=PGOOD, per=None, rbase=None, mods=None, miss=False):
    per, rbase, mods = per or {}, rbase or {}, mods or {}
    base = {(c, s): 150 + 10 * i for i, (c, s) in enumerate((c, s) for c in "ABCD" for s in ("train", "new"))}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E190"))
        for b in J.BRAINS:
            w = lambda nm, txt: open(os.path.join(td, "logs", "E190", nm % b), "w", encoding="utf-8").write(txt)
            w("form_b%d.log", form_log(**mods.get((b, "form"), {})))
            w("train_R_b%d.log", train_log(False, **mods.get((b, "trainR"), {})))
            w("train_P_b%d.log", train_log(True, **mods.get((b, "trainP"), {})))
            for nm, eff, pu in (("R_learn", rbase.get(b, RBASE), 8), ("P_learn", per.get(b, post), 8), ("none", None, 0)):
                if miss and b == 12 and nm == "none":
                    continue
                m = {k: base[k] + (eff[k] if eff else 0) for k in base}
                ek = dict(mods.get((b, nm), {}))
                pushed = ek.pop("pushed", pu)
                w("ev_b%d_" + nm + ".log", eval_log(m, pushed, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


cases = [
    ("유지", dict(), "독립 판정: 유지(H111)"),
    ("경계 r 0.5", dict(post=E(-900, 1000, 1000, -1000)), "독립 판정: 유지(H111)"),
    ("A r 0.4994 → 간섭", dict(post=E(-899, 1600, 1400, -1300)), "독립 판정: 간섭(H111-null)"),
    ("유지 3·간섭 2 → 보류", dict(per={13: E(-300, 400, 1400, -1300), 14: E(-200, 300, 1400, -1300)}), "독립 판정: 보류"),
    ("기준 미달 2 뇌", dict(rbase={13: E(-999, 2000, 0, 0), 14: E(-1800, 999, 0, 0)}), "독립 판정: 보류(기준 미달"),
    ("전제 미달 2 뇌", dict(per={13: E(-1500, 1600, 999, -1300), 14: E(-1500, 1600, 1400, -999)}), "독립 판정: 보류(전제 미달"),
    ("형성 노출 408", dict(mods={(10, "form"): {"expo": 408, "cats": "A,B"}}), "독립 판정: 보류(조작검증 실패)"),
    ("형성 측정 제시 144", dict(mods={(12, "form"): {"npres": 144}}), "독립 판정: 보류(조작검증 실패)"),
    ("기준 학습에 둘째 쌍 줄", dict(mods={(10, "trainR"): {"extra": "[둘째 범주 쌍] 시행 700 부터 가짜\n"}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 특징 가중치 적재 없음", dict(mods={(14, "none"): {"fw": False}}), "독립 판정: 보류(조작검증 실패)"),
    ("평가 로그 결측", dict(miss=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-24s → %-30s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
