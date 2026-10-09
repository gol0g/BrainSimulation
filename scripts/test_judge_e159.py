#!/usr/bin/env python3
"""judge_e159.py 합성 시험(조건 1): 네 범주·경계(O·C 정확히 0.70)·4/5 규칙·섞임 보류, 조작검증 R1~R6 경계·실패, 결측, 원 로그 파싱.
실행: python3 scripts/test_judge_e159.py (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e159 as J


def build(O=0.5, C=0.9, per=None, over=None, drop=None):
    """v 를 R1~R4 재현값에 맞추고, E(W0,R25) = O·E(W0,R0), E(W25,R0) = C·E(W0,R0) 가 되게 만든다(1e-4 정수)."""
    X = {}
    for b in J.BRAINS:
        p = {"O": O, "C": C}
        p.update((per or {}).get(b, {}))
        n0, n25 = J.i4(J.R2[b]), J.i4(J.R1[b])
        base = J.i4(J.R3[b]) - n0
        v = {"none_R0": n0, "none_R25": n25, "W0_R0": J.i4(J.R3[b]), "W25_R25": J.i4(J.R4[b]),
             "W0_R25": n25 + int(round(p["O"] * base)), "W25_R0": n0 + int(round(p["C"] * base))}
        for c in J.CELLS:
            X[(c, b)] = {"mode": "none" if c.startswith("none") else "all", "v": v[c], "pushed": 0 if c.startswith("none") else 8, "ld": 2, "sc": 2}
    for k, upd in (over or {}).items():
        X[k] = dict(X[k], **upd)
    if drop:
        del X[drop]
    return X


ok_all = True


def chk(name, X, want):
    global ok_all
    c, r = J.judge(X)
    got = r["verdict"] if r else c[0]
    good = got == want if want != "결측" else (r is None and "결측" in got)
    ok_all &= good
    print("%-34s 기대 %-22s → %-26s %s" % (name, want, got[:26], "✓" if good else "✗ %s" % c))


chk("출력", build(), "출력(H082)")
chk("내용", build(O=0.9, C=0.5), "내용(H082-content)")
chk("둘 다", build(O=0.5, C=0.5), "둘 다(H082-both)")
chk("둘 다 아님", build(O=0.9, C=0.9), "둘 다 아님(H082-neither)")
chk("O 음수(반사25 에서 역효과) → 출력", build(O=-0.2), "출력(H082)")
# 경계: E(W0,R0) b10 = −4316 − 169 = −4485. O = 0.70 정확 ⇔ 10·E = 7·(−4485) = −31395 → E = −3139.5 (정수 불가) — 정수로 되는 뇌별 값으로 따로 시험
X = build(); base = {b: J.i4(J.R3[b]) - J.i4(J.R2[b]) for b in J.BRAINS}
for b in J.BRAINS:   # E(W0,R25) = 7·base/10 를 정수로: base 가 10 의 배수가 아니면 내림 쪽(더 음수 = O 큼)과 올림 쪽을 나눠 시험
    X[("W0_R25", b)]["v"] = X[("none_R25", b)]["v"] + (7 * base[b]) // 10 + 1   # 7·base/10 보다 큼(덜 음수) → O < 0.70 쪽(≤)
c, r = J.judge(X); g = r["verdict"] == "출력(H082)"; ok_all &= g
print("%-34s → %s %s" % ("O 경계 바로 안쪽(≤0.70)", r["verdict"], "✓" if g else "✗"))
X = build()
for b in J.BRAINS:
    X[("W0_R25", b)]["v"] = X[("none_R25", b)]["v"] + (7 * base[b]) // 10 - 1   # 더 음수 → O > 0.70
c, r = J.judge(X); g = r["verdict"] == "둘 다 아님(H082-neither)"; ok_all &= g
print("%-34s → %s %s" % ("O 경계 바로 바깥(>0.70)", r["verdict"], "✓" if g else "✗"))
X = build(O=0.9)
X2 = {k: dict(v) for k, v in X.items()}
for b in J.BRAINS:   # base 를 10 의 배수로 만들 수 있는 정확 경계: 10·E = 7·base 인 정수 E 가 있으면 그 값(≤ 이므로 C ≤ 0.70)
    if (7 * base[b]) % 10 == 0:
        X2[("W25_R0", b)]["v"] = X2[("none_R0", b)]["v"] + (7 * base[b]) // 10
c, r = J.judge(X2)
print("%-34s → (정확 정수 경계 뇌 %d개) %s" % ("C 정확 경계(정수 가능 뇌)", sum((7 * base[b]) % 10 == 0 for b in J.BRAINS), r["verdict"]))
chk("4/5 출력", build(per={14: {"O": 0.9, "C": 0.9}}), "출력(H082)")
chk("3/5 → 보류", build(per={13: {"O": 0.9, "C": 0.5}, 14: {"O": 0.9, "C": 0.5}}), "보류")
chk("R1 재현 어긋남(0.0021)", build(over={("none_R25", 12): {"v": J.i4(J.R1[12]) + 21}}), "보류(조작검증 실패)")
chk("R1 재현 경계(0.0020)", build(over={("none_R25", 12): {"v": J.i4(J.R1[12]) + 20}}), "출력(H082)")
chk("R2 재현 어긋남", build(over={("none_R0", 10): {"v": J.i4(J.R2[10]) - 21}}), "보류(조작검증 실패)")
chk("R3 재현 어긋남", build(over={("W0_R0", 11): {"v": J.i4(J.R3[11]) + 25}}), "보류(조작검증 실패)")
chk("R4 재현 어긋남", build(over={("W25_R25", 13): {"v": J.i4(J.R4[13]) - 30}}), "보류(조작검증 실패)")
chk("R5 pushed 7", build(over={("W0_R25", 14): {"pushed": 7}}), "보류(조작검증 실패)")
chk("R5 none 에 pushed 8", build(over={("none_R0", 14): {"pushed": 8}}), "보류(조작검증 실패)")
chk("R5 배율 1줄", build(over={("W25_R0", 10): {"sc": 1}}), "보류(조작검증 실패)")
chk("R5 적재 1줄", build(over={("none_R25", 11): {"ld": 1}}), "보류(조작검증 실패)")
chk("결측", build(drop=("W25_R0", 12)), "결측")
# R6: 기준 효과가 작으면 실패 — R3 재현을 지키면서는 만들 수 없으므로 judge 내부 식을 직접 시험
X = build()
for b in J.BRAINS:
    X[("none_R0", b)]["v"] = X[("W0_R0", b)]["v"] + 999   # E(W0,R0) = −999 (R2 는 깨짐 — 두 검사 모두 실패해야)
c, r = J.judge(X); g = r["verdict"] == "보류(조작검증 실패)" and "R6 기준 효과 0/5" in c[0]; ok_all &= g
print("%-34s → %s %s" % ("R6 기준 효과 −0.0999", c[0][-40:], "✓" if g else "✗"))
# 원 로그 파싱
with tempfile.TemporaryDirectory() as td:
    os.makedirs(os.path.join(td, "logs", "E159"))
    open(os.path.join(td, "logs", "E159", "W0_R25_b10.log"), "w", encoding="utf-8").write(
        "[E153 종류 입력 적재] k 검증 일치 — x\n[E157 종류 입력 배율] k=0.7000 검증 일치 — x\n" * 2
        + "=> DECOMP mode=all mod=-0.1234 acc=60.0 off=+0.0010 pushed=8 kc_means[kc_l>l=148.6000]\n")
    J.EXP = td
    Xp = J.load()
g = Xp[("W0_R25", 10)] == {"mode": "all", "v": -1234, "pushed": 8, "ld": 2, "sc": 2} and ("W0_R0", 10) not in Xp
ok_all &= g
print("%-34s → %s" % ("원 로그 파싱", "✓" if g else "✗ %s" % Xp))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
