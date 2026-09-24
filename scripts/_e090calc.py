import io, re
txt = io.open("research/experiments/E090.log", encoding="utf-8").read()
# 태그와 결과를 함께 잡는다
rows = re.findall(r"\n  (\S+) (w\d) (t\d+)?: => MINCIRC mode=(\w+) seed=(\d+).*?\*\*eval=([\d.]+)\*\*", txt)
froz, data = {}, {}
for tag, w, t, mode, sd, ev in rows:
    sd, ev = int(sd), float(ev)
    if mode == "frozen":
        froz[sd] = ev
    else:
        data.setdefault(tag, {}).setdefault(sd, []).append(ev)
print("배선 기준선(frozen):", {k: froz[k] for k in sorted(froz)})
room = [s for s in sorted(froz) if froz[s] < 90]
ceil = [s for s in sorted(froz) if froz[s] >= 90]
print("개선 여지 시드 %s / 천장 시드 %s\n" % (room, ceil))
print("%-12s %-9s %-9s %s" % ("조건", "성공률", "파괴율", "시드별 100% 도달 수"))
out = {}
for tag in sorted(data):
    succ = tot_s = dest = tot_d = 0
    per = []
    for sd, vals in sorted(data[tag].items()):
        n100 = sum(1 for v in vals if v >= 99.9)
        per.append("w%d:%d/%d" % (sd, n100, len(vals)))
        if sd in room:
            succ += n100; tot_s += len(vals)
        else:
            dest += sum(1 for v in vals if v <= froz[sd] - 15); tot_d += len(vals)
    sr = succ/tot_s*100 if tot_s else float("nan")
    dr = dest/tot_d*100 if tot_d else float("nan")
    out[tag] = (sr, dr)
    print("%-12s %5.1f%% (%d/%d)  %5.1f%% (%d/%d)  %s"
          % (tag, sr, succ, tot_s, dr, dest, tot_d, " ".join(per)))
print("\n=== 사전 판정 ===")
base = out.get("eta0.02")
low  = out.get("eta0.001")
if base and low:
    print("(a) 성공률: eta0.02 %.1f%% -> eta0.001 %.1f%%  (지지=2배 이상 AND 20%% 이상)"
          % (base[0], low[0]))
    print("    -> %s" % ("**지지**" if (base[0] > 0 and low[0] >= 2*base[0] and low[0] >= 20)
                        else ("**기각**" if abs(low[0]-base[0]) <= 5 else "보류")))
    print("(b) 파괴율: eta0.02 %.1f%% -> eta0.001 %.1f%%  (지지=절반 이하)" % (base[1], low[1]))
    print("    -> %s" % ("**지지**" if low[1] <= base[1]/2 else
                        ("**기각**" if abs(low[1]-base[1]) <= 5 else "보류")))
    mid = out.get("eta0.005")
    if mid:
        print("(c) 단조성: %.1f -> %.1f -> %.1f  -> %s"
              % (base[0], mid[0], low[0],
                 "지지" if base[0] <= mid[0] <= low[0] else "**위반**"))
if "long" in out:
    print("\n보조(eta0.001, 시행 4배): 성공률 %.1f%%  파괴율 %.1f%%" % out["long"])
