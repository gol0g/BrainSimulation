#!/usr/bin/env python3
"""E196 정정 1: 본실험 지수용 측정 로그(logs/E196/meas_b{B}_T4900.log)의 조작검증 — 경로 검사 (a) 와 같은 항목.
RCKX 288 · RCKXCHK npres 288 · 맥락 켬 스파이크 > 0 · 끔 0 · 비활성 특징 0 · g_same 1 · C2(ctx·ctx2 w 20.00·p 0.020) · motor 기저값 −4900.0(모든 동작점 줄).
하나라도 어기면 E196 판정은 보류(조작검증 실패). 실행: python3 scripts/e196_meas_check.py [--selftest]"""
import os
import re
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
NPRES = 288


def check(path):
    """→ (통과 여부, 사유 목록)."""
    if not os.path.exists(path):
        return False, ["로그 없음"]
    n_rckx = 0
    chk = None
    tonic = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCKX "):
                n_rckx += 1
            elif ln.startswith("RCKXCHK "):
                chk = dict(x.split("=", 1) for x in ln.split()[1:] if "=" in x)
            elif "[motor 동작점]" in ln:
                m = re.search(r"기저값 (-?[0-9.]+)", ln)
                tonic.append(m.group(1) if m else "?")
    why = []
    if n_rckx != NPRES:
        why.append("RCKX %d" % n_rckx)
    if chk is None:
        why.append("RCKXCHK 없음")
    else:
        try:
            if int(chk.get("npres", -1)) != NPRES:
                why.append("npres %s" % chk.get("npres"))
            if int(chk.get("ctx_on_spikes", 0)) <= 0:
                why.append("맥락 켬 스파이크 %s" % chk.get("ctx_on_spikes"))
            if int(chk.get("ctx_off_spikes", -1)) != 0:
                why.append("맥락 끔 스파이크 %s" % chk.get("ctx_off_spikes"))
            if int(chk.get("feat_inactive_spikes", -1)) != 0:
                why.append("비활성 특징 스파이크 %s" % chk.get("feat_inactive_spikes"))
            if chk.get("g_same") != "1":
                why.append("g_same %s" % chk.get("g_same"))
            for k, v in (("ctx_w", "20.00"), ("ctx_p", "0.020"), ("ctx2_w", "20.00"), ("ctx2_p", "0.020")):
                if chk.get(k) != v:
                    why.append("%s %s" % (k, chk.get(k)))
        except ValueError as e:
            why.append("RCKXCHK 값 %s" % e)
    if not tonic or any(t != "-4900.0" for t in tonic):
        why.append("motor 기저값 %s" % (",".join(tonic) or "줄 없음"))
    return not why, why


def main(root="."):
    ok = 0
    for b in BRAINS:
        p = os.path.join(root, EXP, "logs/E196/meas_b%d_T4900.log" % b)
        good, why = check(p)
        ok += good
        print("b%d 측정 %s%s" % (b, "통과" if good else "실패", "" if good else " — " + "; ".join(why)))
    print("측정 조작검증 %s %d/%d" % ("통과" if ok == len(BRAINS) else "실패(→ E196 판정 보류)", ok, len(BRAINS)))
    return ok == len(BRAINS)


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    good_chk = "RCKXCHK npres=288 ctx_on_spikes=576000 ctx_off_spikes=0 feat_inactive_spikes=0 base_l=1 base_r=1 g_same=1 ctx_w=20.00 ctx_p=0.020 ctx2_w=20.00 ctx2_p=0.020\n"
    tonic = "  [motor 동작점] motor 좌·우 Ioffset 기저값 -4900.0(긴장성 억제)\n"

    def mk(name, n=288, chk=good_chk, ton=tonic, ton2=tonic):
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(ton + "빌드\n" + ton2 + "".join("RCKX set=train side=l i=%d\n" % i for i in range(n)) + (chk or ""))
        return p

    cases = [
        ("정상", mk("a"), True),
        ("RCKX 287", mk("b", n=287), False),
        ("RCKXCHK 없음", mk("c", chk=None), False),
        ("npres 287", mk("d", chk=good_chk.replace("npres=288", "npres=287")), False),
        ("맥락 켬 0", mk("e", chk=good_chk.replace("ctx_on_spikes=576000", "ctx_on_spikes=0")), False),
        ("맥락 끔 > 0", mk("f", chk=good_chk.replace("ctx_off_spikes=0", "ctx_off_spikes=5")), False),
        ("비활성 특징 > 0", mk("g", chk=good_chk.replace("feat_inactive_spikes=0", "feat_inactive_spikes=2")), False),
        ("g 바뀜", mk("h", chk=good_chk.replace("g_same=1", "g_same=0")), False),
        ("C1(w 10)", mk("i", chk=good_chk.replace("ctx_w=20.00", "ctx_w=10.00")), False),
        ("기저값 −4000", mk("j", ton=tonic.replace("-4900.0", "-4000.0"), ton2=tonic.replace("-4900.0", "-4000.0")), False),
        ("기저값 한 줄만 −4000", mk("k", ton2=tonic.replace("-4900.0", "-4000.0")), False),
        ("기저값 줄 없음", mk("l", ton="", ton2=""), False),
        ("로그 없음", os.path.join(d, "없음.log"), False),
    ]
    ok = 0
    for nm, p, want in cases:
        got, why = check(p)
        good = got == want
        ok += good
        print("[%s] %s → %s %s" % ("통과" if good else "실패", nm, got, "; ".join(why)))
    # 실제 형식: 경로 검사 (a) 로그(있으면) — −4900 은 통과, −4000 은 기저값으로만 실패해야 한다
    real = [(os.path.join(EXP, "logs/E196/pathcheck/meas_b15_T4900.log"), True), (os.path.join(EXP, "logs/E196/pathcheck/meas_b15_T4000.log"), False)]
    for p, want in real:
        if os.path.exists(p):
            got, why = check(p)
            good = got == want and (want or why == ["motor 기저값 -4000.0,-4000.0"] or (len(why) == 1 and why[0].startswith("motor 기저값")))
            ok += good
            cases.append((p, p, want))
            print("[%s] 실제 %s → %s %s" % ("통과" if good else "실패", os.path.basename(p), got, "; ".join(why)))
    print("합성 %d/%d" % (ok, len(cases)))
    return ok == len(cases)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    sys.exit(0 if main() else 1)
