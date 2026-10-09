#!/usr/bin/env python3
"""verify_e167_independent.py 합성 시험: 손실·남음·부분, 경계(비 0.80·평균 5.0%p 정확, 양수 13/16), 발달 불일치·장치 대조 실패, 모드 어긋남, 결측.
실행: python3 scripts/test_verify_e167.py (저장소 루트에서)"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_e167_independent as V


def run(nl=52.0, nf=50.0, hl=90.0, hf=52.5, per=None, devbad=False, dmax="0.00e+00", wrongmode=False, miss=False):
    with tempfile.TemporaryDirectory() as td:
        for d in (("logs", "E167"), ("logs", "E136"), ("traces", "E167"), ("traces", "E136")):
            os.makedirs(os.path.join(td, *d))
        w_ = lambda p, s: open(os.path.join(td, *p), "w", encoding="utf-8").write(s)
        for w in V.WIRES:
            arr = dict(a=np.arange(400) + w, b=np.arange(400), e=np.arange(400) % 100, i=np.arange(400) % 50)
            np.savez_compressed(os.path.join(td, "traces", "E136", "dev_corr_w%d.npz" % w), **arr)
            arr7 = dict(arr, b=(arr["b"] + (1 if (devbad and w == 85) else 0)))
            np.savez_compressed(os.path.join(td, "traces", "E167", "dev_corr_w%d.npz" % w), cpre=np.zeros(20000, int), ipre=np.zeros(20000, int), **arr7)
            a_nl, a_nf = (per or {}).get(w, (nl, nf))
            for ts in (600, 601):
                for mode, v in (("learn", a_nl), ("frozen", a_nf)):
                    if miss and w == 93 and mode == "frozen" and ts == 601:
                        continue
                    mline = "frozen" if (wrongmode and w == 80 and mode == "learn") else mode
                    w_(("logs", "E167", "N_%s_w%d_t%d.log" % (mode, w, ts)),
                       "[KC망안] /x | 흥분 후보 20000개 평균 0.0712 같은 위치 몫 0.0270 | 억제 후보 20000개 크기 평균 0.0578 같은 위치 몫 0.0580 | 균등 0.0200 | 장치 대조 최대 |차| %s\n"
                       "=> SDLAB diff=cyclic rule=samediff mode=%s seed=%d trialseed=%d train_lbal=55.0 novel_lbal=%.1f\n"
                       % (dmax if (w == 81 and mode == "learn") else "0.00e+00", mline, w, ts, v))
                for mode, v in (("learn", hl), ("frozen", hf)):
                    w_(("logs", "E136", "corr_%s_w%d_t%d.log" % (mode, w, ts)),
                       "[KC불러옴] /x | 일치형 같은 위치 317/400\n=> SDLAB diff=cyclic rule=samediff mode=%s seed=%d trialseed=%d train_lbal=84.5 novel_lbal=%.1f\n" % (mode, w, ts, v))
        V.EXP = td
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            V.main()
    return buf.getvalue()


ok_all = True
for name, kw, want in (("손실(Δ_N +2.0)", {}, "손실(H090-null)"),
                       ("남음(Δ_N = Δ_H)", {"nl": 90.0, "nf": 52.5}, "남음(H090)"),
                       ("부분(Δ_N +20)", {"nl": 70.0, "nf": 50.0}, "부분"),
                       ("비 0.80 정확(+30.0) → 남음", {"nl": 80.0, "nf": 50.0}, "남음(H090)"),
                       ("평균 5.0 정확 → 부분", {"nl": 55.0, "nf": 50.0}, "부분"),
                       ("평균 4.9 → 손실", {"nl": 54.9, "nf": 50.0}, "손실(H090-null)"),
                       ("남음 크기·양수 13/16 → 부분", {"nl": 100.0, "nf": 50.0, "per": {78: (40.0, 50.0), 79: (40.0, 50.0), 80: (40.0, 50.0)}}, "부분"),
                       ("발달 불일치(w85)", {"devbad": True}, "보류(조작검증 실패)"),
                       ("장치 대조 차 1e-3", {"dmax": "1.00e-03"}, "보류(조작검증 실패)"),
                       ("모드 어긋남(learn 파일에 frozen 줄)", {"wrongmode": True}, "보류(결측"),
                       ("결측", {"miss": True}, "보류(결측")):
    out = run(**kw)
    last = out.strip().splitlines()[-1]
    good = last.startswith("독립 판정: %s" % want)
    ok_all &= good
    print("%-34s → %-34s %s" % (name, last, "✓" if good else "✗\n" + out))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
