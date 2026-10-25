#!/usr/bin/env python3
"""E208 뒤 사후 기술(탐색 — 판정 밖, 통합 병목 정리 8절): 첫 쌍 규칙이 둘째 쌍으로 번지는 길이 '사례 절반 이상' 집합 겹침 밖에 있는가.
측정 로그의 RCKC tag=post set=train 줄(쪽·범주마다 노출 사례 6개의 반응 KC)에서 KC 마다 반응 비율 p(0~1)를 구해
  집합 J   = p ≥ 0.5 집합의 자카드(RCJ 와 같은 정의 — 로그의 RCJ J_post 와 같아야 한다: 자기 점검),
  가중 J   = Σ min(p_X, p_Y) / Σ max(p_X, p_Y)(문턱 아래 약한 반응까지),
  코사인   = Σ p_X p_Y / (|p_X| |p_Y|)
를 A·C, B·D, A·B 쌍마다, 두 쪽마다 계산한다. 합 S 는 두 쪽 J(A,C)+J(B,D)(1e-4 정수), 코사인은 네 값(두 쪽 × AC·BD) 평균.
조건: E207 F·맥락 없음(meas_F_none), E207 F·맥락 X 흥분 20(meas_F_X), E208 흥분 0·맥락 X(meas_F_X). 뇌 16~20.
읽기(가설 생성 — 확인 아님): 흥분 20 → 0 에서 집합 S 는 약 1/4 로 줄었다(E208 보고). 가중 겹침·코사인이 그만큼 줄지 않았다면 문턱 아래 약한 반응의 공유가
남은 것(둘째 쌍 전이 크기가 그대로인 것과 맞음), 함께 줄었다면 전이의 길은 KC 반응 겹침 밖(판독 다시 쓰기 등).
실행: python3 scripts/posthoc_e208_graded.py (저장소 루트에서)"""
import math
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
RX = re.compile(r"^RCKC tag=post side=([lr]) set=train cat=([ABCD]) i=(\d+) n=(\d+) ids=([0-9,]*)")
RX_J = re.compile(r"^RCJ side=([lr]) pair=([A-D]{2}) J_pre=[0-9.]+ J_post=([0-9.]+)")


def read(path):
    """→ ({(side, cat): [set, …]}, {(side, pair): J_post 1e-4 정수})."""
    cases, rcj = {}, {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RX.match(ln)
        if m:
            ids = set(int(x) for x in m.group(5).split(",") if x)
            if len(ids) != int(m.group(4)):
                raise ValueError("n 불일치 %s" % ln[:60])
            cases.setdefault((m.group(1), m.group(2)), []).append(ids)
            continue
        m = RX_J.match(ln)
        if m:
            rcj[(m.group(1), m.group(2))] = int(round(float(m.group(3)) * 1e4))
    return cases, rcj


def pvec(case_sets):
    n = len(case_sets)
    out = {}
    for s in case_sets:
        for k in s:
            out[k] = out.get(k, 0) + 1
    return {k: v / n for k, v in out.items()}


def metrics(px, py):
    keys = set(px) | set(py)
    sx = {k for k, v in px.items() if v >= 0.5 - 1e-12}; sy = {k for k, v in py.items() if v >= 0.5 - 1e-12}
    u = sx | sy
    jset = len(sx & sy) / len(u) if u else 0.0
    mn = sum(min(px.get(k, 0.0), py.get(k, 0.0)) for k in keys); mx = sum(max(px.get(k, 0.0), py.get(k, 0.0)) for k in keys)
    wj = mn / mx if mx else 0.0
    dot = sum(px.get(k, 0.0) * py.get(k, 0.0) for k in keys)
    nx = math.sqrt(sum(v * v for v in px.values())); ny = math.sqrt(sum(v * v for v in py.values()))
    cos = dot / (nx * ny) if nx and ny else 0.0
    return jset, wj, cos


def brain_stats(path):
    cases, rcj = read(path)
    if set(cases) != {(s, c) for s in "lr" for c in "ABCD"} or any(len(v) != 6 for v in cases.values()):
        raise ValueError("사례 줄 수 %s" % {k: len(v) for k, v in cases.items()})
    p = {k: pvec(v) for k, v in cases.items()}
    r = {}
    for s in "lr":
        for pr in ("AC", "BD", "AB"):
            r[(s, pr)] = metrics(p[(s, pr[0])], p[(s, pr[1])])
    chk = all(int(round(r[(s, pr)][0] * 1e4)) == rcj.get((s, pr)) for s in "lr" for pr in ("AC", "BD", "AB"))
    S = lambda i: int(round(sum(r[(s, pr)][i] for s in "lr" for pr in ("AC", "BD")) * 1e4))  # noqa: E731
    cos = sum(r[(s, pr)][2] for s in "lr" for pr in ("AC", "BD")) / 4
    ab = [r[(s, "AB")] for s in "lr"]
    return {"S_set": S(0), "S_w": S(1), "cos": cos, "AB_set": sum(x[0] for x in ab) / 2, "AB_w": sum(x[1] for x in ab) / 2,
            "AB_cos": sum(x[2] for x in ab) / 2, "self_check": chk}


def main():
    conds = (("E207 맥락 없음", "E207", "meas_F_none_b%d.log"), ("E207 흥분 20·X", "E207", "meas_F_X_b%d.log"), ("E208 흥분 0·X", "E208", "meas_F_X_b%d.log"))
    print("E208 뒤 사후 기술(탐색 — 판정 밖): 집합 겹침(p ≥ 0.5) 대 활동 가중 겹침(Σmin/Σmax)·코사인 — 노출 사례 6개, 둘째 측정(post)")
    allok = True
    rows = {}
    for b in BRAINS:
        parts = []
        for nm, e, f in conds:
            st = brain_stats(os.path.join(EXP, "logs", e, f % b))
            rows[(b, nm)] = st
            allok = allok and st["self_check"]
            parts.append("%s S 집합 %d · 가중 %d · 코사인 %.3f | A·B 집합 %.3f · 가중 %.3f · 코사인 %.3f" % (nm, st["S_set"], st["S_w"], st["cos"], st["AB_set"], st["AB_w"], st["AB_cos"]))
        print("b%d: %s" % (b, " || ".join(parts)))
    print("자기 점검(집합 J = 로그 RCJ J_post, 18쌍 × 3조건 × 5뇌): %s" % ("통과" if allok else "실패"))
    for key, lab in (("S_set", "집합 S"), ("S_w", "가중 S"), ("cos", "코사인")):
        r20 = [rows[(b, "E208 흥분 0·X")][key] / rows[(b, "E207 흥분 20·X")][key] for b in BRAINS]
        rn = [rows[(b, "E207 맥락 없음")][key] / rows[(b, "E207 흥분 20·X")][key] for b in BRAINS]
        print("요약 %s: 흥분 0 ÷ 흥분 20 = %.2f~%.2f · 맥락 없음 ÷ 흥분 20 = %.2f~%.2f" % (lab, min(r20), max(r20), min(rn), max(rn)))
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
