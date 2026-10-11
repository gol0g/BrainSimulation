#!/usr/bin/env python3
"""verify_e183_independent.py 합성 시험 — test_judge_e183.py 와 같은 가짜 로그 생성(복사), 독립 대조의 범주·조작검증·결측 확인.
사례 KC 집합 = 범주 핵심 c 개(모든 사례 공통) + 사례마다 고유 잡음 u 개 → S = c/(c+u)(다른 범주와 겹침 0), 붕괴는 B 핵심 = A 핵심.
실행: python3 scripts/test_verify_e183.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e183_independent as V  # noqa: E402

NTR, NNEW, REPS = 6, 6, 3


def sets_for(c=40, u_t=40, u_n=40, collapse=False, seed_base=0):
    """{side: {set: {cat: [set...]}}} — 잡음 번호는 seed_base 로 겹치지 않게."""
    out = {}
    nxt = [5000 + seed_base]
    for sd in "lr":
        out[sd] = {}
        for st, m, u in (("train", NTR, u_t), ("new", NNEW, u_n)):
            out[sd][st] = {}
            for cat in "AB":
                core = list(range(0, c)) if (cat == "A" or collapse) else list(range(1000, 1000 + c))
                lst = []
                for _ in range(m):
                    noise = list(range(nxt[0], nxt[0] + u)); nxt[0] += u
                    lst.append(sorted(core + noise))
                out[sd][st][cat] = lst
    return out


def chk_line(tag, frac=0.95, fi=0, fa=4000, fo=0, gb=0, same=1, npres=144):
    return ("RCCHK tag=%s npres=%d feat_active_frac=%.4f feat_inactive_spikes=%d feat_active_spikes=%d feat_other_side=%d goodbad=%d g_same=%d base_l=10 base_r=10\n"
            % (tag, npres, frac, fi, fa, fo, gb, same))


def write_log(path, pre, pre2, post, arm, chk=None, dg=None, done=True):
    chk = chk or {}
    t = "[원시 특징 세계] 가짜\n"
    for tag, S in (("pre", pre), ("pre2", pre2), ("post", post)):
        for sd in "lr":
            for st in ("train", "new"):
                for cat in "AB":
                    for i, s in enumerate(S[sd][st][cat]):
                        t += "RCKC tag=%s side=%s set=%s cat=%s i=%d n=%d ids=%s\n" % (tag, sd, st, cat, i, len(s), ",".join(map(str, s)))
        t += chk_line(tag, **chk.get(tag, {}))
        if tag == "pre2":
            for sd in "lr":
                d = (dg if dg is not None else (12.5 if arm == "O" else 0.0))
                t += "RCW side=%s nsyn=6000 g_mean0=4.0000 g_mean1=4.1000 dg_abs=%.4f frac_changed=%.4f wsel0=0.6000 wsel1=0.7000 kc_in_mean0=8.0 kc_in_mean1=8.2\n" % (sd, d, 0.3 if d > 0 else 0.0)
    if done:
        t += "=> RAWCAT side=l 가짜 | feat_n=200\n"
    open(path, "w", encoding="utf-8").write(t)


def run(o_post=None, mods=None, miss=False, nodone=False):
    mods = mods or {}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E183"))
        for b in V.BRAINS:
            for arm in ("O", "Z"):
                if miss and b == 13 and arm == "Z":
                    continue
                m = mods.get((b, arm), {})
                pre = m.get("pre", sets_for())
                pre2 = m.get("pre2", pre)
                post = m.get("post", (o_post or sets_for()) if arm == "O" else sets_for())
                write_log(os.path.join(td, "logs", "E183", "rc_b%d_%s.log" % (b, arm)), pre, pre2, post, arm,
                          chk=m.get("chk"), dg=m.get("dg"), done=not (nodone and b == 12 and arm == "O"))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


cases = [
    ("지지", dict(o_post=sets_for(u_t=10, u_n=10)), "독립 판정: 지지(H104)"),
    ("경계 0.05 → 지지", dict(o_post=sets_for(c=44, u_t=36, u_n=36)), "독립 판정: 지지(H104)"),
    ("경계 아래 → 이득 없음", dict(o_post=sets_for(c=44, u_t=37, u_n=37)), "독립 판정: 이득 없음(H104-null-1)"),
    ("암기", dict(o_post=sets_for(u_t=0, u_n=35)), "독립 판정: 암기(H104-null-2)"),
    ("이득 없음", dict(), "독립 판정: 이득 없음(H104-null-1)"),
    ("역효과", dict(o_post=sets_for(u_t=80, u_n=80)), "독립 판정: 역효과"),
    ("붕괴", dict(o_post=sets_for(collapse=True)), "독립 판정: 붕괴(H104-null-3)"),
    ("3/5 → 보류", dict(o_post=sets_for(u_t=10, u_n=10), mods={(b, "O"): {"post": sets_for()} for b in (13, 14)}), "독립 판정: 보류"),
    ("충실도 0.89", dict(o_post=sets_for(u_t=10, u_n=10), mods={(11, "O"): {"chk": {"post": {"frac": 0.89}}}}), "독립 판정: 보류(조작검증 실패)"),
    ("Z 노출 g 변함", dict(o_post=sets_for(u_t=10, u_n=10), mods={(11, "Z"): {"dg": 0.5}}), "독립 판정: 보류(조작검증 실패)"),
    ("짝 어긋남", dict(o_post=sets_for(u_t=10, u_n=10), mods={(12, "Z"): {"pre": sets_for(u_t=60, u_n=60, seed_base=90000)}}), "독립 판정: 보류(조작검증 실패)"),
    ("반복성 나쁨", dict(o_post=sets_for(u_t=10, u_n=10), mods={(13, "O"): {"pre2": sets_for(u_t=20, u_n=20, seed_base=70000)}}), "독립 판정: 보류(조작검증 실패)"),
    ("결측", dict(o_post=sets_for(u_t=10, u_n=10), miss=True), "독립 판정: 보류(결측"),
    ("완료 줄 없음", dict(o_post=sets_for(u_t=10, u_n=10), nodone=True), "독립 판정: 보류(결측"),
]
ok_all = True
for name, kw, want in cases:
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith(want) and (want != "독립 판정: 보류" or last == "독립 판정: 보류")
    ok_all &= good
    print("%-18s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
