#!/usr/bin/env python3
"""E196: RCKX 측정 로그(rawcat --rc-measure-ctx) → KC 별 지수 파일(npz: sxi_·csi_·resp_·tot_·npc_ {l,r}).
계산은 backend/genesis/kcxor_oracle.sxi_rows(합성 자체 시험), 파싱은 scripts/rcxor.parse.
실행: python3 scripts/kcxor_build.py <RCKX 로그> <출력 npz>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "backend", "genesis"))
import kcxor_oracle as KX  # noqa: E402
import rcxor  # noqa: E402


def build(path, n_kc=3000):
    rows = rcxor.parse(path, n_kc)
    if set(rows) != {"l", "r"}:
        raise ValueError("쪽 %s" % sorted(rows))
    out = {}
    for sd in ("l", "r"):
        d = KX.sxi_rows(rows[sd], n_kc)
        for k in ("sxi", "csi", "resp", "tot"):
            out["%s_%s" % (k, sd)] = d[k]
        out["npc_" + sd] = np.int64(d["npc"])
    return out


def main(argv):
    src, dst = argv[0], argv[1]
    o = build(src)
    np.savez(dst, **o)
    for sd in ("l", "r"):
        cx, sx, rp = KX.counts(o["sxi_" + sd], o["resp_" + sd])
        cc, sc, _ = KX.counts(o["csi_" + sd], o["resp_" + sd])
        print("[E196 지수] %s 쪽 %s | 반응 %d · SXI ≥ 0.5 %d · ≤ −0.5 %d · CSI ≥ 0.5 %d · ≤ −0.5 %d · 칸당 제시 %d"
              % (os.path.basename(dst), sd, rp, cx, sx, cc, sc, int(o["npc_" + sd])))


if __name__ == "__main__":
    main(sys.argv[1:])
