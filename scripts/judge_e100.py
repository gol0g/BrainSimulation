#!/usr/bin/env python3
"""E100 판정 — research/experiments/E100.md 4절 사전기준을 그대로 구현한다.

  (인자 없음) : E100.log + logs/E100/ 원본으로 판정. **96런이 다 모이기 전에는 수치를 출력하지 않는다.**
  --selftest  : 합성 자료로 경계 사례 검증.
"""
import os
import re
import sys

LOG = "research/experiments/E100.log"
RAW = "research/experiments/logs/E100"
LINE = re.compile(r"^\s*(L4|rev|L8) w(\d) t(\d+): => MINCIRC .*?first=([0-9.]+) last=([0-9.]+) .*?\*\*eval=([0-9.]+)\*\*")
REVL = re.compile(r"\(reversal\) 최종 규칙 기준 ([0-9.]+)% \| 원래 규칙 기준 ([0-9.]+)%")
MAIN_W = [5, 6, 7]
ALL_W = [5, 6, 7, 8]
TS = list(range(300, 308))


def load():
    r = {}
    for ln in open(LOG, encoding="utf-8").read().splitlines():
        m = LINE.match(ln)
        if m:
            c, s, t, first, last, ev = m.groups()
            d = {"first": float(first), "last": float(last), "eval": float(ev)}
            if c == "rev":
                f = os.path.join(RAW, "rev_w%s_t%s.log" % (s, t))
                mm = REVL.search(open(f, encoding="utf-8").read()) if os.path.exists(f) else None
                if not mm:
                    continue          # 원래 규칙 점수 없으면 미완으로 취급
                d["new"], d["orig"] = float(mm.group(1)), float(mm.group(2))
            r[(c, int(s), int(t))] = d
    return r


def judge(r):
    out = []
    valid, twin_bad, rev_ok, fails = [], [], [], []
    for s in MAIN_W:
        for t in TS:
            l4, rv = r[("L4", s, t)], r[("rev", s, t)]
            if l4["first"] != rv["first"]:
                twin_bad.append((s, t))
                continue
            if l4["eval"] < 90.0:
                continue
            valid.append((s, t))
            if rv["new"] >= 90.0:
                rev_ok.append((s, t))
            else:
                fails.append((s, t, rv["new"], rv["orig"]))
    # 반전 조작 확인: rev 후반 구간 == L8 후반 구간 이면 의심
    same_second = [(s, t) for s in ALL_W for t in TS if r[("rev", s, t)]["last"] == r[("L8", s, t)]["last"]]
    out.append("조작검증: 쌍둥이 첫 구간 불일치 %d/24 %s | 반전 후반=대조 후반 동일 %d/32%s" % (
        len(twin_bad), twin_bad if twin_bad else "", len(same_second),
        " (반전 조작 의심)" if len(same_second) > 16 else ""))
    for s in ALL_W:
        out.append("  배선 %d%s: L4 %s | rev 새규칙 %s | rev 원래 %s | L8 %s" % (
            s, " (별도 보고)" if s == 8 else "",
            " ".join("%.0f" % r[("L4", s, t)]["eval"] for t in TS),
            " ".join("%.0f" % r[("rev", s, t)]["new"] for t in TS),
            " ".join("%.0f" % r[("rev", s, t)]["orig"] for t in TS),
            " ".join("%.0f" % r[("L8", s, t)]["eval"] for t in TS)))
    nv = len(valid)
    if nv < 16:
        a = "판정 불가"
        out.append("(a) 유연성: 판정 불가 — 유효 쌍 %d/24 (<16, 획득 자체가 약함)" % nv)
    else:
        rate = len(rev_ok) / nv
        a = "지지" if rate >= 0.8 else ("기각" if rate <= 0.4 else "보류")
        out.append("(a) 유연성: %s — 반전 성공 %d/%d (%.1f%%)" % (a, len(rev_ok), nv, rate * 100))
    if a in ("기각", "보류") and fails:
        kinds = {"고착": 0, "혼란": 0, "기타": 0}
        for _, _, new, orig in fails:
            if orig >= 90.0:
                kinds["고착"] += 1
            elif 10.0 < new < 90.0 and 10.0 < orig < 90.0:
                kinds["혼란"] += 1
            else:
                kinds["기타"] += 1
        out.append("(b) 실패 양상: 고착 %d / 혼란 %d / 기타 %d → 다수: %s" % (
            kinds["고착"], kinds["혼란"], kinds["기타"], max(kinds, key=kinds.get)))
    else:
        out.append("(b) 실패 양상: 해당 없음 (실패 %d건)" % len(fails))
    p4 = sum(r[("L4", s, t)]["eval"] >= 90.0 for s in MAIN_W for t in TS) / 24 * 100
    p8 = sum(r[("L8", s, t)]["eval"] >= 90.0 for s in MAIN_W for t in TS) / 24 * 100
    out.append("(c) 긴 훈련 대조: learn400 %.1f%% vs learn800 %.1f%% → %s" % (
        p4, p8, "긴 훈련이 연합을 흔든다(>=20%p 하락)" if p4 - p8 >= 20.0 else "흔들림 없음"))
    return out, a


def selftest():
    def mk(n_valid, n_rev_ok, fail_orig=95.0, twin_break=0, l8_drop=False):
        r = {}
        i = 0
        for s in ALL_W:
            for t in TS:
                main = s in MAIN_W
                idx = MAIN_W.index(s) * 8 + (t - 300) if main else 99
                l4ev = 100.0 if idx < n_valid else 50.0
                ok = idx < n_rev_ok
                r[("L4", s, t)] = {"first": 60.0, "last": 60.0, "eval": l4ev}
                r[("rev", s, t)] = {"first": 60.0 + (0.1 if idx < twin_break else 0.0), "last": 30.0,
                                    "eval": 100.0 if ok else 5.0, "new": 100.0 if ok else 100.0 - fail_orig, "orig": 0.0 if ok else fail_orig}
                r[("L8", s, t)] = {"first": 60.0, "last": 65.0, "eval": 50.0 if (l8_drop and idx < 12) else l4ev}
        return r
    cases = [
        (mk(20, 16), "지지"), (mk(20, 15), "보류"), (mk(20, 9), "보류"), (mk(20, 8), "기각"),
        (mk(15, 15), "판정 불가"), (mk(20, 20, twin_break=5), "판정 불가"),
    ]
    npass = 0
    for r, exp in cases:
        _, a = judge(r)
        npass += (a == exp)
        if a != exp:
            print("실패:", exp, "→", a)
    out, _ = judge(mk(20, 8, fail_orig=95.0))
    npass += any("다수: 고착" in x for x in out)
    out, _ = judge(mk(20, 8, fail_orig=50.0))
    npass += any("다수: 혼란" in x for x in out)
    out, _ = judge(mk(24, 24, l8_drop=True))
    npass += any("흔든다" in x for x in out)
    total = len(cases) + 3
    print("자체 검증 %d/%d 통과" % (npass, total))
    return npass == total


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    if not os.path.exists(LOG):
        print("[E100] 로그 없음 — 미실행"); sys.exit(0)
    r = load()
    need = {(c, s, t) for c in ("L4", "rev", "L8") for s in ALL_W for t in TS}
    have = need & set(r)
    if len(have) < len(need):
        print("[E100] %d/%d런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % (len(have), len(need)))
        sys.exit(0)
    print("[E100] %d/%d런 완료" % (len(have), len(need)))
    out, _ = judge(r)
    print("\n".join(out))
