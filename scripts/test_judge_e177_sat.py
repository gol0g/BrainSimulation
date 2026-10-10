#!/usr/bin/env python3
"""judge_e177_sat.py 합성 정답 시험 — 포화 경계(0.66)·재현 차(1e-4 정수 2)·결측·rc·진단 줄·사전 판정 보존·마지막 줄 우선,
그리고 실제 뇌 10 프로브 요약(logs/E177/sat_check/summary.out)을 정규식 파서와 독립 분할 파서로 각각 읽어 대조.
실행: python3 scripts/test_judge_e177_sat.py (저장소 루트에서)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e177_sat as JS  # noqa: E402

BR = (10, 11, 12, 13, 14)
WN = {"learn": "all", "none": "none"}
BASE_EV = {("learn", "off"): -0.4186, ("learn", "on"): -0.0000, ("none", "off"): 0.0080, ("none", "on"): 0.0007}
n_ok = n_bad = 0


def line(b, w, c, mod, rc=0, on_rates=None, off_rates=(0.57, 0.64, 0.64, 0.55), sides=("left", "right")):
    r = on_rates if c == "on" else off_rates
    dg = {"left": "variant=base side=left n=250 motor L/R %.6f/%.6f KC L/R 0.006926/0.003237" % (r[0], r[1]),
          "right": "variant=base side=right n=250 motor L/R %.6f/%.6f KC L/R 0.003227/0.007001" % (r[2], r[3])}
    return "b%d %s %s rc=%d | => DECOMP mode=%s mod=%+.4f acc=0.0 off=-0.0000 pushed=8 kc_means | %s " % (
        b, WN[w], c, rc, WN[w], mod, " ".join(dg[s] for s in sides))


def build(on_rates=(0.6664, 0.6662, 0.6663, 0.6661), per=None, drop=None, ev_shift=None):
    """per[b] = 그 뇌의 켬 motor 발화율 4개(좌 자극 L/R, 우 자극 L/R), drop = 빼는 (b, w, c), ev_shift[(b,w,c)] = 재측정 변조폭에 더할 값."""
    EV, lines = {}, []
    for b in BR:
        for (w, c), m in BASE_EV.items():
            EV[(b, w, c)] = m
            if drop and (b, w, c) in drop:
                continue
            mm = m + (ev_shift or {}).get((b, w, c), 0.0)
            lines.append(line(b, w, c, mm, on_rates=(per or {}).get(b, on_rates)))
    return EV, "\n".join(lines)


def res(ok=True, verdict="부분(H098-partial) — 한 맥락만(켬만 0·끔만 5)"):
    return {"ok": ok, "verdict": verdict, "eoff": {b: -4266 for b in BR}}


def chk(name, got, want_sub):
    global n_ok, n_bad
    good = want_sub in got
    n_ok += good
    n_bad += (not good)
    print("  %-40s → %s %s" % (name, got[:90], "✓" if good else "✗ (기대 포함: %s)" % want_sub))


def run(ev, txt, r=None):
    return JS.amend(res() if r is None else r, ev, JS.parse_summary(txt))[0]


EV, T = build()
chk("모두 포화(0.6661~0.6664)", run(EV, T), "켬 평가 포화 5/5")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59))
chk("모두 비포화 → 사전 판정 그대로", run(EV, T), "부분(H098-partial)")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59), per={12: (0.6664, 0.6662, 0.6663, 0.6661)})
chk("한 뇌만 포화 → 보류", run(EV, T), "켬 평가 포화 1/5")
EV, T = build(on_rates=(0.6664, 0.6662, 0.6663, 0.6599))
chk("경계: 8개 중 하나 0.6599 → 비포화", run(EV, T), "부분(H098-partial)")
EV, T = build(on_rates=(0.6600, 0.6600, 0.6600, 0.6600))
chk("경계: 모두 정확히 0.6600 → 포화", run(EV, T), "켬 평가 포화 5/5")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59), ev_shift={(11, "learn", "off"): 0.0002})
chk("재현 차 2(1e-4) → 통과", run(EV, T), "부분(H098-partial)")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59), ev_shift={(11, "learn", "off"): 0.0003})
chk("재현 차 3 → 재현 실패 보류", run(EV, T), "측정 타당성 실패 1/5")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59), drop={(13, "none", "on")})
chk("재측정 한 줄 결측 → 보류", run(EV, T), "측정 타당성 실패 1/5")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59))
T = T.replace("b14 none on rc=0", "b14 none on rc=1")
chk("rc=1 → 보류", run(EV, T), "측정 타당성 실패 1/5")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59))
T = "\n".join(ln if not ln.startswith("b10 all on") else line(10, "learn", "on", -0.0, on_rates=(0.60, 0.62, 0.62, 0.59), sides=("left",)) for ln in T.splitlines())
chk("진단 줄 한쪽만 → 보류", run(EV, T), "측정 타당성 실패 1/5")
EV, T = build()
chk("사전 판정 조작검증 실패는 그대로", run(EV, T, r=res(ok=False, verdict="보류(조작검증 실패)")), "보류(조작검증 실패)")
chk("사전 판정 결측 → 보류", JS.amend(None, EV, JS.parse_summary(T))[0], "보류(사전 판정 결측)")
EV, T = build(on_rates=(0.60, 0.62, 0.62, 0.59))
# 포화 판정은 켬 평가 두 개(학습·무학습) 8개 값이 모두 ≥ 0.66 이어야 하므로 두 줄을 모두 덧붙인다(첫 판 시험은 학습 켬만 덧붙여 기대가 틀렸음).
T = T + "\n" + line(10, "learn", "on", -0.0, on_rates=(0.6664, 0.6662, 0.6663, 0.6661)) + "\n" + line(10, "none", "on", 0.0007, on_rates=(0.6664, 0.6662, 0.6663, 0.6661))
chk("같은 키 두 번 → 마지막 줄(포화) 사용", run(EV, T), "켬 평가 포화 1/5")

# 실제 뇌 10 프로브 요약: 정규식 파서 vs 독립 분할 파서
f = os.path.join("research", "experiments", "logs", "E177", "sat_check", "summary.out")
txt = open(f, encoding="utf-8").read()
P = JS.parse_summary(txt)


def split_parse(t):
    out = {}
    for ln in t.splitlines():
        if not ln.startswith("b10 "):
            continue
        head, rest = ln.split(" | ", 1)
        b, w, c, _ = head.split()
        vals = []
        for seg in rest.split("motor L/R ")[1:]:
            a, bb = seg.split()[0].split("/")
            vals += [float(a), float(bb)]
        out[(w, c)] = vals
    return out


S2 = split_parse(txt)
reg = {(w, c): [x for s in ("left", "right") for x in P[(10, {"all": "learn", "none": "none"}[w], c)]["diag"][s][:2]] for (w, c) in S2}
chk("실제 b10: 두 파서의 motor 값 16개 일치", "일치" if reg == S2 else "불일치 %s vs %s" % (reg, S2), "일치")
on_vals = S2[("all", "on")] + S2[("none", "on")]
chk("실제 b10: 켬 motor 8개 최소 ≥ 0.66", "최소 %.4f" % min(on_vals) + (" 포화" if min(on_vals) >= 0.66 else " 비포화"), "포화")
EVr = {(10, w, c): m for (w, c), m in BASE_EV.items()}
ok10, why10, sat10 = JS.brain_check(10, EVr, P)
chk("실제 b10: brain_check(재현 4개·포화)", "포화" if (sat10 and not ok10 and len(why10) == 1) else "기대 밖 %s" % why10, "포화")
print("test_judge_e177_sat ✓%d ✗%d" % (n_ok, n_bad))
sys.exit(1 if n_bad else 0)
