#!/usr/bin/env python3
"""E074 분석: 환경별 뇌 반복의 (수리-정적) 평균과 표준오차."""
import re, sys, statistics as st
log = "research/experiments/E074.log"
pat = re.compile(r"env(\d+) b(\d+) (\S+)\s+(수리|정적):.*변조폭 변화 ([+-][\d.]+)")
d = {}
for line in open(log, encoding="utf-8", errors="ignore"):
    m = pat.search(line)
    if not m: continue
    env, br, cond, kind, val = m.group(1), m.group(2), m.group(3), m.group(4), float(m.group(5))
    d.setdefault((env, cond), {}).setdefault(br, {})[kind] = val

print("%-6s %-10s %8s %8s %8s %s" % ("env", "조건", "평균", "표준편차", "표준오차", "n"))
print("-" * 58)
res = {}
for (env, cond), brains in sorted(d.items()):
    diffs = [v["수리"] - v["정적"] for v in brains.values() if "수리" in v and "정적" in v]
    if not diffs: continue
    m = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else float("nan")
    se = sd / (len(diffs) ** 0.5) if len(diffs) > 1 else float("nan")
    res[(env, cond)] = (m, se, len(diffs))
    print("%-6s %-10s %+8.4f %8.4f %8.4f %d" % (env, cond, m, sd, se, len(diffs)))

print("\n=== 환경별 (11.25x − 기준) ===")
print("사전기준: 0.008 이상 더 음수면 지지 / 세 환경 모두 0.004 이내면 기각")
for env in sorted({e for e, _ in res}):
    base = res.get((env, "기준1.0x")); wide = res.get((env, "11.25x"))
    if not base or not wide: continue
    diff = wide[0] - base[0]
    verdict = "지지방향" if diff <= -0.008 else ("기각권" if abs(diff) <= 0.004 else "중간")
    print("  env%s: %+.4f  (기준 %+.4f → 11.25x %+.4f)  [%s]" % (env, diff, base[0], wide[0], verdict))
