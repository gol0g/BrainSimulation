#!/usr/bin/env python3
"""E099 판정 — research/experiments/E099.md 4절 사전기준을 그대로 구현한다.

  (인자 없음)  : E099.log 를 판정한다. **144런이 다 모이기 전에는 수치를 출력하지 않는다.**
  --selftest   : 합성 로그로 경계 사례를 검증한다.

성공 = 평가 정답률 >= 90 (2026-09-27 00:58 선언).
(a) 일반화: learn w5~9 성공 >= 32/40 지지, <= 16/40 기각(과적합), 사이 보류
(b) 귀속: (a) 지지일 때만. shuffled 성공 <= 8/40 그리고 frozen(w5~9) 성공 <= 4/20
(c) 파괴: 배선 1·2, 배선별 frozen 평균 대비 learn 이 10%p 이상 떨어진 런 >= 4/16 이면 파괴
조작검증: shuffled reward == 같은 짝 learn reward, noreward·frozen reward == 0
"""
import re
import sys

LOG = "research/experiments/E099.log"
LINE = re.compile(r"^\s*(frozen|learn|shuf|norew) w(\d) t(\d+): => MINCIRC .*?reward=([0-9.]+) \*\*eval=([0-9.]+)\*\*")
W_NEW = [5, 6, 7, 8, 9]
T_ALL = list(range(200, 208))
T_HALF = list(range(200, 204))


def expected():
    e = set()
    for s in W_NEW + [1, 2]:
        for t in T_HALF:
            e.add(("frozen", s, t))
    for s in W_NEW:
        for t in T_ALL:
            e.add(("learn", s, t))
            e.add(("shuf", s, t))
        for t in T_HALF:
            e.add(("norew", s, t))
    for s in [1, 2]:
        for t in T_ALL:
            e.add(("learn", s, t))
    return e


def parse(text):
    r = {}
    for ln in text.splitlines():
        m = LINE.match(ln)
        if m:
            c, s, t, rw, ev = m.groups()
            r[(c, int(s), int(t))] = (float(rw), float(ev))
    return r


def judge(r):
    out = []
    ok = lambda ev: ev >= 90.0
    # 조작검증
    bad = []
    for s in W_NEW:
        for t in T_ALL:
            if r[("shuf", s, t)][0] != r[("learn", s, t)][0]:
                bad.append("shuf w%d t%d reward %.1f != learn %.1f" % (s, t, r[("shuf", s, t)][0], r[("learn", s, t)][0]))
    for k, v in r.items():
        if k[0] in ("frozen", "norew") and v[0] != 0.0:
            bad.append("%s w%d t%d reward=%.1f (0이어야)" % (k + (v[0],)))
    out.append("조작검증: " + ("통과" if not bad else "위반 %d건 — %s" % (len(bad), "; ".join(bad))))

    def row(c, ws, ts):
        return [r[(c, s, t)][1] for s in ws for t in ts]

    L, SH, FR, NR = row("learn", W_NEW, T_ALL), row("shuf", W_NEW, T_ALL), row("frozen", W_NEW, T_HALF), row("norew", W_NEW, T_HALF)
    n = lambda xs: sum(ok(x) for x in xs)
    n100 = lambda xs: sum(x >= 100.0 for x in xs)
    for name, xs in (("learn w5~9", L), ("shuffled", SH), ("frozen w5~9", FR), ("noreward", NR)):
        out.append("  %-12s 성공 %2d/%d  (100%%: %d)  평균 %.1f  [%s]" % (name, n(xs), len(xs), n100(xs), sum(xs) / len(xs), " ".join("%.0f" % x for x in xs)))
    for s in W_NEW:
        out.append("  배선 %d: frozen %s | learn %s | shuf %s" % (
            s, " ".join("%.0f" % r[("frozen", s, t)][1] for t in T_HALF),
            " ".join("%.0f" % r[("learn", s, t)][1] for t in T_ALL),
            " ".join("%.0f" % r[("shuf", s, t)][1] for t in T_ALL)))

    a = "지지(일반화)" if n(L) >= 32 else ("기각(과적합)" if n(L) <= 16 else "보류")
    out.append("(a) 일반화: %s — learn 성공 %d/40" % (a, n(L)))
    if a.startswith("지지"):
        b_ok = n(SH) <= 8 and n(FR) <= 4
        out.append("(b) 학습 귀속: %s — shuffled %d/40 (<=8), frozen %d/20 (<=4)" % (
            "성립" if b_ok else "성립 안 함 — (a) 성공을 수반성 학습 때문이라고 말할 수 없음", n(SH), n(FR)))
    else:
        out.append("(b) 학습 귀속: 해당 없음((a) 미지지) — 참고 shuffled %d/40, frozen %d/20" % (n(SH), n(FR)))
    drops = []
    for s in [1, 2]:
        base = sum(r[("frozen", s, t)][1] for t in T_HALF) / len(T_HALF)
        for t in T_ALL:
            ev = r[("learn", s, t)][1]
            if ev < base - 10.0:
                drops.append("w%d t%d %.0f<%.1f-10" % (s, t, ev, base))
        out.append("  배선 %d: frozen 평균 %.1f | learn %s" % (s, base, " ".join("%.0f" % r[("learn", s, t)][1] for t in T_ALL)))
    out.append("(c) 파괴 점검: %s — 10%%p 이상 하락 %d/16 %s" % (
        "파괴" if len(drops) >= 4 else "파괴 관측 없음", len(drops), ("(" + ", ".join(drops) + ")") if drops else ""))
    return out, a, bad


def selftest():
    def mk(learn_ok, shuf_ok, fr_ok, w12_learn=90.0, w12_frozen=90.0, shuf_rw_mismatch=False):
        lines = []
        for (c, s, t) in sorted(expected()):
            if c == "learn" and s in W_NEW:
                i = W_NEW.index(s) * 8 + (t - 200)
                ev = 100.0 if i < learn_ok else 50.0
            elif c == "shuf":
                i = W_NEW.index(s) * 8 + (t - 200)
                ev = 95.0 if i < shuf_ok else 50.0
            elif c == "frozen" and s in W_NEW:
                i = W_NEW.index(s) * 4 + (t - 200)
                ev = 92.0 if i < fr_ok else 48.0
            elif c == "frozen":
                ev = w12_frozen
            elif c == "learn":
                ev = w12_learn
            else:
                ev = 50.0
            rw = 60.0 if c in ("learn", "shuf") else 0.0
            if shuf_rw_mismatch and c == "shuf" and s == 5 and t == 200:
                rw = 61.0
            lines.append("  %s w%d t%d: => MINCIRC mode=x seed=%d first=0 last=0 delta=+0 reward=%.1f **eval=%.1f** tie=0" % (c, s, t, s, rw, ev))
        return parse("\n".join(lines))
    cases = [
        (mk(32, 0, 0), "지지", None), (mk(31, 0, 0), "보류", None), (mk(17, 0, 0), "보류", None),
        (mk(16, 0, 0), "기각", None), (mk(40, 8, 4), "지지", "성립"), (mk(40, 9, 0), "지지", "성립 안 함"),
        (mk(40, 0, 5), "지지", "성립 안 함"),
    ]
    npass = 0
    for r, a_exp, b_exp in cases:
        out, a, _ = judge(r)
        bline = [x for x in out if x.startswith("(b)")][0]
        good = a.startswith(a_exp) and (b_exp is None or bline.split(": ", 1)[1].startswith(b_exp))
        npass += good
        if not good:
            print("실패:", a_exp, b_exp, "→", a, bline)
    # 파괴 경계: frozen 90, learn 79.9 (10.1 하락) 4런 이상 → 파괴 / 80.0 은 하락 아님
    for learn_v, exp in ((79.9, "파괴"), (80.0, "파괴 관측 없음")):
        out, _, _ = judge(mk(32, 0, 0, w12_learn=learn_v))
        cl = [x for x in out if x.startswith("(c)")][0]
        good = cl.split(": ", 1)[1].split(" —")[0] == exp
        npass += good
        if not good:
            print("실패(c):", learn_v, cl)
    _, _, bad = judge(mk(32, 0, 0, shuf_rw_mismatch=True))
    good = len(bad) == 1
    npass += good
    print("자체 검증 %d/%d 통과" % (npass, len(cases) + 3))
    return npass == len(cases) + 3


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    r = parse(open(LOG, encoding="utf-8").read())
    exp = expected()
    have = exp & set(r)
    if len(have) < len(exp):
        print("[E099] %d/%d런 완료 — **판정 보류. 결과가 다 모일 때까지 수치를 출력하지 않는다.**" % (len(have), len(exp)))
        sys.exit(0)
    print("[E099] %d/%d런 완료" % (len(have), len(exp)))
    out, _, _ = judge(r)
    print("\n".join(out))
