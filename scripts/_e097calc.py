import io, re
txt = io.open("research/experiments/E097.log", encoding="utf-8").read()
A, B = {}, {}
for line in txt.splitlines():
    mA = re.search(r"^\s+A_d(\d+)_tau(\d+) w(\d) t\d+: => MINCIRC .*?\*\*eval=([\d.]+)\*\*", line)
    mB = re.search(r"^\s+B_act(\d+)_tau(\d+) w(\d) t\d+: => MINCIRC .*?\*\*eval=([\d.]+)\*\*", line)
    if mA: A.setdefault((int(mA.group(1)), int(mA.group(2))), []).append((int(mA.group(3)), float(mA.group(4))))
    elif mB: B.setdefault((int(mB.group(1)), int(mB.group(2))), []).append((int(mB.group(3)), float(mB.group(4))))
def rate(v):
    n = sum(1 for _, e in v if e >= 99.9)
    per = " ".join("w%d:%d/%d" % (s, sum(1 for d,e in v if d==s and e>=99.9),
                                  len([1 for d,_ in v if d==s])) for s in (0,3,4))
    return n/len(v)*100, n, len(v), per
print("=== 축 A: 행동 창 15 고정, 지연만 변경 ===")
print("%-8s %-10s %-20s %s" % ("delay", "tau", "평가 성공률", "시드별"))
best = {}
for d in (0, 30, 90):
    for tau in (12, 25, 50):
        if (d, tau) not in A: continue
        r = rate(A[(d, tau)])
        print("%-8s %-10s %5.1f%% (%2d/%2d)        %s" % (d, tau, r[0], r[1], r[2], r[3]))
        if d not in best or r[0] > best[d][1]: best[d] = (tau, r[0])
    if d in best: print("   -> delay %d 최고: tau %d (%.1f%%)" % (d, best[d][0], best[d][1]))
print("\n=== 축 B: E096 미탐색 범위 보완 ===")
print("%-8s %-10s %-20s %s" % ("act", "tau", "평가 성공률", "시드별"))
E096 = {(5,3):12.5,(5,6):66.7,(5,12):83.3,(45,12):0.0,(45,25):8.3,(45,50):20.8}
Bres = {}
for act, taus in ((5,(12,25,50)),(45,(50,100,200))):
    for tau in taus:
        if (act, tau) not in B: continue
        r = rate(B[(act, tau)])
        Bres[(act,tau)] = r[0]
        print("%-8s %-10s %5.1f%% (%2d/%2d)        %s" % (act, tau, r[0], r[1], r[2], r[3]))
print("\n=== 조작검증 1: 회귀 검증 (act15 delay0 tau12 = 100%) ===")
r = rate(A[(0,12)])
print("  %.1f%% (%d/%d)  -> %s" % (r[0], r[1], r[2],
      "**통과**" if r[0] >= 99.9 else "**실패 — 전체 해석 보류**"))
print("\n=== 사전 판정 ===")
if 0 in best and 90 in best:
    d0, d90 = best[0][1], best[90][1]
    print("(a) 지연 효과: delay0 최고 %.1f%% -> delay90 최고 %.1f%%  차이 %+.1f%%p"
          % (d0, d90, d90-d0))
    print("    H026 지지(30%p 이상 하락) -> %s / H026-window(15%p 이내) -> %s"
          % ("**해당**" if d0-d90 >= 30 else "해당 없음", "**해당**" if abs(d0-d90) <= 15 else "해당 없음"))
if len(best) == 3:
    taus = [best[d][0] for d in (0,30,90)]
    print("(b) 최적 tau: delay 0/30/90 = %s  -> %s" % (taus,
          "**지지(단조증가·전부 다름)**" if taus[0]<taus[1]<taus[2] else
          ("**기각(전부 같음)**" if len(set(taus))==1 else "부분(일부만 다름)")))
print("\n(c) 범위 보완 — 내부 최적점을 감쌌는가")
for act, taus, prev in ((5,(12,25,50),{3:12.5,6:66.7}), (45,(50,100,200),{12:0.0,25:8.3})):
    seq = [(t, Bres.get((act,t))) for t in taus if (act,t) in Bres]
    full = sorted(list(prev.items()) + seq)
    vals = [v for _, v in full]
    if not vals: continue
    imax = vals.index(max(vals))
    inner = 0 < imax < len(vals)-1
    print("  act %d: %s" % (act, " ".join("tau%d=%.1f" % (t, v) for t, v in full)))
    print("     최고 tau=%d  -> %s" % (full[imax][0], "**내부 최적점(감쌈)**" if inner else "**여전히 경계값 — 이 축을 접는다**"))
