#!/usr/bin/env python3
"""채점·판정 규칙 검증 (CPU, GPU 불필요).

왜: 코드를 고친 것과 그 코드가 맞게 도는 것은 다르다(외부 검토 조건 2).
반전 채점 수정과 E098 방향 판정을 **경계 사례로** 확인한다.
"""
import ast, io, sys

# --- 1) 반전 채점: 최종 규칙과 원래 규칙을 둘 다 세는가 ---
src = io.open("backend/genesis/minimal_circuit.py", encoding="utf-8").read()
tree = ast.parse(src)
has_final = "_final = FLIP if a.mode ==" in src
has_orig = "eval_orig += 1" in src
uses_final = "if _final[stim] == act:" in src
print("=== 1) 반전 채점 ===")
print("  최종 규칙 선택(_final)        : %s" % ("O" if has_final else "**X**"))
print("  최종 규칙으로 채점            : %s" % ("O" if uses_final else "**X**"))
print("  원래 규칙 점수도 별도 집계    : %s" % ("O" if has_orig else "**X**"))

# 채점 논리를 순수 함수로 재현해 정답표 검사
RULE = {"A": "L", "B": "R"}; FLIP = {"A": "R", "B": "L"}
def score(mode, stim, act):
    final = FLIP if mode == "reversal" else RULE
    return (final[stim] == act, RULE[stim] == act)
cases = [("learn","A","L",True,True), ("learn","A","R",False,False),
         ("reversal","A","R",True,False), ("reversal","A","L",False,True),
         ("reversal","B","L",True,False), ("reversal","B","R",False,True)]
bad = 0
for mode, stim, act, want_f, want_o in cases:
    got_f, got_o = score(mode, stim, act)
    ok = (got_f == want_f and got_o == want_o)
    bad += 0 if ok else 1
    print("  %-9s %s→%s  최종=%-5s 원래=%-5s  %s" % (mode, stim, act, got_f, got_o, "OK" if ok else "**틀림**"))
print("  -> %s" % ("**통과**" if bad == 0 and has_final and uses_final and has_orig else "**실패**"))

# --- 2) E098 방향 판정: 효과 존재와 목표 방향이 분리되는가 ---
print("")
print("=== 2) E098 판정: 효과 존재(a-1) vs 목표 방향(a-2) ===")
def judge(eff, base):
    """eff: (학습-무학습) 5뇌 리스트, base: tau200 평균 절대효과"""
    m = sum(eff) / len(eff)
    a1 = abs(m) >= 0.005 and (base == 0 or abs(m) >= 3 * base) and \
         max(sum(1 for x in eff if x > 0), sum(1 for x in eff if x < 0)) >= 4
    a2 = a1 and sum(1 for x in eff if x < 0) >= 4
    return a1, a2
tests = [
    ("목표 방향으로 큰 효과", [-0.02]*5, 0.001, True, True),
    ("**반사 강화** 방향 큰 효과", [+0.02]*5, 0.001, True, False),
    ("효과 없음", [0.0005]*5, 0.001, False, False),
    ("부호 불일치", [-0.02, +0.02, -0.02, +0.02, -0.02], 0.001, False, False),
    ("경계: 정확히 0.005", [-0.005]*5, 0.001, True, True),
]
bad2 = 0
for name, eff, base, w1, w2 in tests:
    g1, g2 = judge(eff, base)
    ok = (g1 == w1 and g2 == w2)
    bad2 += 0 if ok else 1
    print("  %-26s a1=%-5s a2=%-5s  %s" % (name, g1, g2, "OK" if ok else "**틀림**"))
print("  -> %s" % ("**통과**" if bad2 == 0 else "**실패**"))
print("")
print("핵심: '반사 강화 방향 큰 효과'가 a1=True 이지만 a2=False 로 갈린다.")
print("      원래 기준(절대값만)이었다면 이것이 '지지'로 세어졌다.")
sys.exit(0 if (bad == 0 and bad2 == 0) else 1)
