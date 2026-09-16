#!/usr/bin/env python3
"""E076 분석: 좌우 대칭 강제가 환경 간 효과 분산을 줄이는가 (H008 인과 검증)."""
import re, statistics as st
log = "research/experiments/E076.log"
pat = re.compile(r"env(\d+) (원본|대칭) b(\d+) (수리|정적): .*변조폭 변화 ([+-][\d.]+)")
d = {}
for line in open(log, encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if not m:
        continue
    env, mode, br, kind, val = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
    d.setdefault((mode, env), {}).setdefault(br, {})[kind] = val

print("%-6s %-5s %9s %9s %s" % ("모드", "env", "평균효과", "표준오차", "n"))
print("-" * 42)
eff = {}
for (mode, env), brains in sorted(d.items()):
    diffs = [v["수리"] - v["정적"] for v in brains.values() if "수리" in v and "정적" in v]
    if not diffs:
        continue
    m_ = st.mean(diffs)
    se = (st.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else float("nan")
    eff.setdefault(mode, {})[env] = m_
    print("%-6s %-5s %+9.4f %9.4f %d" % (mode, env, m_, se, len(diffs)))

print("\n=== 환경 간 효과 분산 (사전기준: 대칭이 원본의 절반 이하면 H008 지지) ===")
for mode in ("원본", "대칭"):
    vals = list(eff.get(mode, {}).values())
    if len(vals) > 1:
        print("  %-4s: 표준편차 %.5f   값 %s" % (mode, st.stdev(vals),
              " ".join("%+.4f" % v for v in vals)))
o = st.stdev(list(eff["원본"].values())) if len(eff.get("원본", {})) > 1 else float("nan")
s = st.stdev(list(eff["대칭"].values())) if len(eff.get("대칭", {})) > 1 else float("nan")
if o == o and s == s:
    r = s / o
    verdict = "H008 지지" if r <= 0.5 else ("H008 기각" if r >= 0.8 else "부분 기여(0.5~0.8) → 확장 필요")
    print("\n  대칭/원본 비율 = %.3f  →  **%s**" % (r, verdict))
