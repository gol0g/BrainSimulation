#!/usr/bin/env python3
"""verify_e191_independent.py 합성 시험 — test_judge_e191.py 와 같은 가짜 로그 생성(복사)으로 독립 대조 확인.
실행: python3 scripts/test_verify_e191.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e191_independent as J  # noqa: E402

PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")


def form_log(expo=816, cats="A,B,C,D", npres=288):
    t = "[원시 특징 세계] 가짜\n"
    for tag in ("pre", "pre2", "post"):
        t += "RCCHK tag=%s npres=%d feat_active_frac=1.0000 feat_inactive_spikes=0 feat_active_spikes=115200 feat_other_side=0 goodbad=0 g_same=1 base_l=1 base_r=1\n" % (tag, npres)
    for sd in "lr":
        t += "RCW side=%s nsyn=60000 g_mean0=24.0000 g_mean1=21.4000 dg_abs=150000.0000 frac_changed=0.3500 wsel0=0.6600 wsel1=0.6600 kc_in_mean0=480.0 kc_in_mean1=430.0 g0_init=1\n" % sd
    t += "[원시 특징 노출] 제시 %d(주기 17 × 사례 6 × 범주 %d × 쪽 2), 노출 eta 0.02, Oja 켬, 도파민 호출 없음%s\n" % (
        expo, len(cats.split(",")), "" if cats == "A,B" else " — 노출 범주 %s" % cats)
    for sd in "lr":
        for p in PAIRS:
            t += "RCJ side=%s pair=%s J_pre=0.3500 J_post=0.0250 n_pre=120/120 n_post=140/140\n" % (sd, p)
    return t + "=> RAWCAT 가짜\n"


def train_log(kind, n2=(1500, 1500, 3000), nrev=(1500, 1500, 3000), fw=True, extra=""):
    n = 1500 if kind == "R" else 3000
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n"
    t += "[E184 특징 가중치 적재] f.npz 검증 일치 — z\n" if fw else ""
    t += "[범주 과제] catrule — 가짜\n"
    for i in range(3):
        t += "[범주 과제 점검] 시행 %d 범주 A 사례 0 쪽 left 정답 교차 — 행동 창 끝 특징 발화 120 · 시행 끝 0\n" % i
    if kind == "P":
        t += "[둘째 범주 쌍] 시행 1500 부터 범주 C(같은 쪽)·D(교차) — 가짜\n"
    if kind == "V":
        t += "[범주 반전] 시행 1500 부터 정답 A → 같은 쪽 · B → 교차\n"
    t += "[학습] 가짜 완료, 보상 900회\n"
    t += "[범주 과제] 시행 %d 중 범주 B(같은 쪽) %d · A(교차) %d\n" % (n, n // 2, n - n // 2)
    if kind == "P" and n2:
        t += "[둘째 범주 쌍] 둘째 쌍 시행 %d (시행 %d 부터, 전체 %d)\n" % n2
    if kind == "V" and nrev:
        t += "[범주 반전] 반전 시행 %d (시행 %d 부터, 전체 %d)\n" % nrev
    return t + extra


def eval_log(mods, pushed, fw=True, sat=False):
    t = "[E153 종류 입력 적재] x 검증 일치 — y\n"
    t += "[E184 특징 가중치 적재] f.npz 검증 일치 — z\n" if fw else ""
    for c in "ABCD":
        for s in ("train", "new"):
            mot = "0.6650/0.6670" if sat else "0.5000/0.5200"
            t += ("RCEVAL cat=%s set=%s mod=%+.4f off=+0.0100 acc=50.0 trials=96 feat_on=96 | side=left n=480 motor L/R 0.5100/0.4900 KC L/R 0.006/0.003 side=right n=480 motor L/R %s KC L/R 0.003/0.006\n"
                  % (c, s, mods[(c, s)] / 1e4, mot))
    return t + "=> DECOMP mode=x rceval A_train=+0.0 pushed=%d feat=f.npz kc_means[x]\n" % pushed


def E(aT, aN, bT, bN, cT=0, dT=0):
    return {("A", "train"): aT, ("A", "new"): aN, ("B", "train"): bT, ("B", "new"): bN,
            ("C", "train"): cT, ("C", "new"): cT // 2, ("D", "train"): dT, ("D", "new"): dT // 2}


RG = E(-1800, -1100, 1800, 1200, 100, 300)          # 기준: 획득·일반화(비 0.61·0.67)
PG = E(-1600, -900, 1600, 1000, 2000, -1800)        # 둘째 쌍 획득, 유지 몫 0.89
VG = E(1200, 800, -1300, -800)                      # 반전 성공(비 0.67·0.62)


def run(R=RG, P=PG, V=VG, per=None, mods=None, miss=False):
    per, mods = per or {}, mods or {}
    base = {(c, s): 150 + 10 * i for i, (c, s) in enumerate((c, s) for c in "ABCD" for s in ("train", "new"))}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E191"))
        for b in J.BRAINS:
            w = lambda nm, txt: open(os.path.join(td, "logs", "E191", nm), "w", encoding="utf-8").write(txt)
            w("form_b%d.log" % b, form_log(**mods.get((b, "form"), {})))
            for k in "RPV":
                w("train_%s_b%d.log" % (k, b), train_log(k, **mods.get((b, "train" + k), {})))
            effs = {"R": per.get((b, "R"), R), "P": per.get((b, "P"), P), "V": per.get((b, "V"), V)}
            for k in "RPV":
                if miss and b == 18 and k == "V":
                    continue
                ek = dict(mods.get((b, "ev" + k), {}))
                pu = ek.pop("pushed", 8)
                w("ev_b%d_%s_learn.log" % (b, k), eval_log({kk: base[kk] + effs[k][kk] for kk in base}, pu, **ek))
            ek = dict(mods.get((b, "none"), {}))
            pu = ek.pop("pushed", 0)
            w("ev_b%d_none.log" % b, eval_log(base, pu, **ek))
        J.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            J.main()
    return buf.getvalue()


two = lambda key, eff: {(19, key): eff, (20, key): eff}
two = lambda key, eff: {(19, key): eff, (20, key): eff}
cases = [
    ("모두 보존", dict(), "독립 판정: 기준선 확증(H112)"),
    ("일반화 경계 비 0.5", dict(R=E(-1800, -900, 1800, 900, 100, 300), P=E(-1600, -800, 1600, 800, 2000, -1800)), "독립 판정: 기준선 확증(H112)"),
    ("일반화 약화", dict(R=E(-1800, -899, 1800, 1200, 100, 300)), "독립 판정: 부분(유지만 확증 — 일반화 약화(사례 한정)·반전 보존)"),
    ("반전 고착", dict(V=E(-1700, -1000, 1700, 1100)), "독립 판정: 부분(유지만 확증 — 일반화 보존·반전 고착)"),
    ("반전 부분", dict(V=E(900, 600, -1300, -800)), "독립 판정: 부분(유지만 확증 — 일반화 보존·반전 부분)"),
    ("유지 몫 0.4994 → 불확증", dict(P=E(-899, -500, 1600, 1000, 2000, -1800)), "독립 판정: 불확증(H112-null)"),
    ("기준 미달 2 뇌", dict(per=two("R", E(-999, -600, 1800, 1200, 100, 300))), "독립 판정: 보류"),
    ("V 반전 시행 수 1400", dict(mods={(18, "trainV"): {"nrev": (1400, 1500, 3000)}}), "독립 판정: 보류(조작검증 실패)"),
    ("P 에 반전 줄", dict(mods={(19, "trainP"): {"extra": "[범주 반전] 시행 1500 부터 가짜\n"}}), "독립 판정: 보류(조작검증 실패)"),
    ("무학습 평가 적재 없음", dict(mods={(18, "none"): {"fw": False}}), "독립 판정: 보류(조작검증 실패)"),
    ("V 평가 결측", dict(miss=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-22s → %-44s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
