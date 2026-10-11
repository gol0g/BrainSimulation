#!/usr/bin/env python3
"""e183_pick.py 합성 시험 — K_pre·K_post·RCCHK 를 정한 가짜 보정 로그로 선택·확장·실패를 확인.
실행: python3 scripts/test_e183_pick.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e183_pick as P  # noqa: E402


def log(kpre, kpost, frac=0.95, gb=0, fi=0, done=True, n_pre=48):
    t = ""
    for tag, k, m in (("pre", kpre, n_pre), ("post", kpost, 48)):
        for i in range(m):
            ids = ",".join(str(x) for x in range(int(k)))
            t += "RCKC tag=%s side=l set=train cat=A i=%d n=%d ids=%s\n" % (tag, i, int(k), ids)
    for tag in ("pre", "pre2", "post"):
        t += ("RCCHK tag=%s npres=144 feat_active_frac=%.4f feat_inactive_spikes=%d feat_active_spikes=4000 feat_other_side=0 goodbad=%d g_same=1 base_l=1 base_r=1\n"
              % (tag, frac, fi, gb))
    if done:
        t += "=> RAWCAT 가짜\n"
    return t


def run(spec):
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "logs", "E183", "calib"))
        for w, kw in spec.items():
            open(os.path.join(td, "logs", "E183", "calib", "rc_w%d_b15.log" % w), "w", encoding="utf-8").write(log(**kw))
        P.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main()
        res = open(os.path.join(td, "logs", "E183", "pick.txt"), encoding="utf-8").read().strip()
    return res, buf.getvalue()


cases = [
    ("130 가장 가까운 16", {8: dict(kpre=60, kpost=70), 16: dict(kpre=125, kpost=150), 24: dict(kpre=260, kpost=300)}, "W=16"),
    ("로그 거리(8:65 대 24:260 → 둘 다 ln2, 작은 W)", {8: dict(kpre=65, kpost=70), 16: dict(kpre=30, kpost=40), 24: dict(kpre=260, kpost=300)}, "W=8"),
    ("P2 범위 밖 제외(16: 45)", {8: dict(kpre=20, kpost=25), 16: dict(kpre=45, kpost=50), 24: dict(kpre=90, kpost=100)}, "W=24"),
    ("P3 폭주 제외(16 post 4배)", {8: dict(kpre=60, kpost=70), 16: dict(kpre=130, kpost=521), 24: dict(kpre=300, kpost=310)}, "W=8"),
    ("P1 충실도 제외(16 frac 0.85)", {8: dict(kpre=60, kpost=70), 16: dict(kpre=130, kpost=140, frac=0.85), 24: dict(kpre=380, kpost=390)}, "W=8"),
    ("P1 good·bad 발화 제외", {8: dict(kpre=60, kpost=70, gb=3), 16: dict(kpre=130, kpost=140, gb=1), 24: dict(kpre=300, kpost=310, gb=2)}, "W=none"),
    ("모두 작음 → 확장 32,48", {8: dict(kpre=10, kpost=12), 16: dict(kpre=20, kpost=22), 24: dict(kpre=40, kpost=45)}, "EXTEND=32,48"),
    ("모두 큼 → 확장 4", {8: dict(kpre=500, kpost=520), 16: dict(kpre=900, kpost=950), 24: dict(kpre=1400, kpost=1500)}, "EXTEND=4"),
    ("섞임(작음·큼) → 실패", {8: dict(kpre=30, kpost=32), 16: dict(kpre=40, kpost=45, frac=0.5), 24: dict(kpre=500, kpost=520)}, "W=none"),
    ("확장 뒤 선택(32: 70)", {8: dict(kpre=10, kpost=12), 16: dict(kpre=20, kpost=22), 24: dict(kpre=40, kpost=45),
                              32: dict(kpre=70, kpost=80), 48: dict(kpre=140, kpost=600)}, "W=32"),
    ("확장 뒤에도 없음 → 실패", {8: dict(kpre=10, kpost=12), 16: dict(kpre=20, kpost=22), 24: dict(kpre=40, kpost=45),
                                 32: dict(kpre=45, kpost=50), 48: dict(kpre=48, kpost=50)}, "W=none"),
    ("불완전 로그 제외(16 완료 줄 없음)", {8: dict(kpre=60, kpost=70), 16: dict(kpre=130, kpost=140, done=False), 24: dict(kpre=300, kpost=310)}, "W=8"),
    ("기본 격자 결측 → 실패", {8: dict(kpre=10, kpost=12), 16: dict(kpre=20, kpost=22)}, "W=none"),
]
ok_all = True
for name, spec, want in cases:
    res, out = run(spec)
    good = res == want
    ok_all &= good
    print("%-40s → %-14s %s" % (name, res, "✓" if good else "✗ (기대 %s)\n%s" % (want, out)))
print("전체: %s (%d 사례)" % ("통과" if ok_all else "실패", len(cases)))
sys.exit(0 if ok_all else 1)
