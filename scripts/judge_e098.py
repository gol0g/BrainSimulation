#!/usr/bin/env python3
"""E098 판정 — 사전등록(research/experiments/E098.md 4절)을 그대로 구현한다.

문서와 판정 코드가 갈라지지 않게 하려는 것이다(E096에서 판정 코드가 기준보다 느슨했다).
  --selftest : 경계 사례로 판정 논리를 검증한다(데이터 불필요)
  (인자 없음) : E098.log 를 판정한다. **40런이 다 모이기 전에는 수치를 출력하지 않는다.**

효과 정의: 뇌 b 마다  e_b = (학습 변조폭 변화) − (무학습 변조폭 변화).  tau 별 평균 m = mean(e_b).
(a-1) 존재 : tau 3 또는 12 중 하나라도  |m| >= 0.005  AND  |m| >= 3*|m_200|  AND  같은 부호 >= 4/5
      기각 : 모든 tau 에서 |m| <= 0.003
      그 외 : 보류
(a-2) 방향 : (a-1)을 만족한 tau 에서 음수 부호 >= 4/5 -> 지지 / 양수 -> '반대'(반사 강화, 목표 실패)
(b)  용량-반응 : |m_200| < |m_50| < |m_12|  (**크기=절대값 기준**, 2026-09-26 명시)
(c)  무학습 대조 : 모든 tau·뇌에서 무학습 변화 == 0.0000 (표시 정밀도). 위반 시 전체 해석 보류
"""
import io, re, sys

TAUS = (200, 50, 12, 3)
BRAINS = (0, 1, 2, 3, 4)
LOG = "research/experiments/E098.log"


def parse(text):
    d = {}
    for line in text.splitlines():
        m = re.search(r"^\s+tau(\d+) (학습|무학습) b(\d): =>.*?변조폭 변화 ([+-][\d.]+)", line)
        if m:
            d[(int(m.group(1)), m.group(2), int(m.group(3)))] = float(m.group(4))
    return d


def judge(d):
    out = {"missing": [], "c": None, "tau": {}}
    for t in TAUS:
        for b in BRAINS:
            for arm in ("학습", "무학습"):
                if (t, arm, b) not in d:
                    out["missing"].append((t, arm, b))
    if out["missing"]:
        return out
    out["c"] = all(abs(d[(t, "무학습", b)]) < 5e-5 for t in TAUS for b in BRAINS)
    for t in TAUS:
        e = [d[(t, "학습", b)] - d[(t, "무학습", b)] for b in BRAINS]
        m = sum(e) / len(e)
        out["tau"][t] = {"e": e, "m": m, "neg": sum(1 for x in e if x < 0),
                         "pos": sum(1 for x in e if x > 0)}
    base = abs(out["tau"][200]["m"])
    a1_taus = []
    for t in (3, 12):
        r = out["tau"][t]
        same = max(r["neg"], r["pos"]) >= 4
        if abs(r["m"]) >= 0.005 and abs(r["m"]) >= 3 * base and same:
            a1_taus.append(t)
    if a1_taus:
        out["a1"] = "지지"
    elif all(abs(out["tau"][t]["m"]) <= 0.003 for t in TAUS):
        out["a1"] = "기각"
    else:
        out["a1"] = "보류"
    out["a1_taus"] = a1_taus
    if not a1_taus:
        out["a2"] = "해당 없음"
    elif any(out["tau"][t]["neg"] >= 4 for t in a1_taus):
        out["a2"] = "지지"
    else:
        out["a2"] = "반대(반사 강화 — 목표 실패)"
    a = [abs(out["tau"][t]["m"]) for t in (200, 50, 12)]
    out["b"] = "지지" if a[0] < a[1] < a[2] else "미충족"
    return out


def selftest():
    def mk(eff, noreward=0.0):
        d = {}
        for t in TAUS:
            for i, b in enumerate(BRAINS):
                d[(t, "무학습", b)] = noreward
                d[(t, "학습", b)] = noreward + eff[t][i]
        return d
    z = [0.0005] * 5
    cases = [
        ("목표 방향 큰 효과(tau12)", {200: z, 50: [-0.002]*5, 12: [-0.02]*5, 3: z},
         {"a1": "지지", "a2": "지지", "b": "지지", "c": True}),
        ("반사 강화 방향 큰 효과", {200: z, 50: [0.002]*5, 12: [0.02]*5, 3: z},
         {"a1": "지지", "a2": "반대(반사 강화 — 목표 실패)", "b": "지지"}),
        ("전부 효과 없음", {200: z, 50: z, 12: z, 3: z},
         {"a1": "기각", "a2": "해당 없음", "b": "미충족"}),
        ("부호 불일치(3:2)", {200: z, 50: z, 12: [-0.02, -0.02, -0.02, 0.02, 0.02], 3: z},
         {"a1": "보류", "a2": "해당 없음"}),
        ("경계: |m|=0.005 정확히", {200: [0.0]*5, 50: z, 12: [-0.005]*5, 3: z},
         {"a1": "지지", "a2": "지지"}),
        ("경계: 3배 미달", {200: [-0.004]*5, 50: z, 12: [-0.011]*5, 3: z},
         {"a1": "보류"}),
        ("경계: 4/5 같은 부호", {200: z, 50: z, 12: [-0.03, -0.03, -0.03, -0.03, 0.01], 3: z},
         {"a1": "지지", "a2": "지지"}),
    ]
    bad = 0
    for name, eff, want in cases:
        got = judge(mk(eff))
        ok = all(got.get(k) == v for k, v in want.items())
        bad += 0 if ok else 1
        print("  %-26s %s  %s" % (name, "OK" if ok else "**틀림**",
              "" if ok else {k: got.get(k) for k in want}))
    # (c) 무학습 대조 위반
    got = judge(mk({200: z, 50: z, 12: [-0.02]*5, 3: z}, noreward=0.0003))
    ok = got["c"] is False
    bad += 0 if ok else 1
    print("  %-26s %s" % ("무학습 대조 ≠ 0", "OK" if ok else "**틀림**"))
    # 결측
    d = mk({200: z, 50: z, 12: z, 3: z}); del d[(3, "학습", 4)]
    ok = judge(d)["missing"] == [(3, "학습", 4)]
    bad += 0 if ok else 1
    print("  %-26s %s" % ("결측 1런", "OK" if ok else "**틀림**"))
    print("  -> %s" % ("**통과**" if bad == 0 else "**실패 %d건**" % bad))
    return bad == 0


def main():
    if "--selftest" in sys.argv:
        print("=== E098 판정 코드 자체 검증 ===")
        sys.exit(0 if selftest() else 1)
    d = parse(io.open(LOG, encoding="utf-8").read())
    r = judge(d)
    done = 40 - len(r["missing"])
    if r["missing"]:
        print("[E098] %d/40런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % done)
        return
    print("[E098] 40/40런 완료")
    for t in TAUS:
        x = r["tau"][t]
        print("  tau %-3d  효과 %s  평균 %+.5f  음수 %d/5" % (
            t, " ".join("%+.4f" % v for v in x["e"]), x["m"], x["neg"]))
    print("(c) 무학습 대조 전부 0.0000: %s%s" % (r["c"], "" if r["c"] else "  -> **전체 해석 보류**"))
    print("(a-1) 효과 존재: %s  (해당 tau: %s)" % (r["a1"], r["a1_taus"] or "없음"))
    print("(a-2) 목표 방향: %s" % r["a2"])
    print("(b)  용량-반응(절대값): %s" % r["b"])


if __name__ == "__main__":
    main()
