"""posthoc_e204 합성 시험 — 무리 배정 재현(실제 진단 로그 무리 수와 대조)·규칙 일치 계산. 실행: python3 scripts/test_posthoc_e204.py"""
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import posthoc_e204 as P  # noqa: E402
ok = tot = 0


def chk(n, c):
    global ok, tot
    tot += 1; ok += bool(c); print(("OK   " if c else "실패 ") + n)


# 1 무리 배정 재현: 뇌 15 개발 진단·경로 검사 로그 '무리 X 좌 1493·우 1463 / Y 좌 1507·우 1537'
s15 = P.subsets(15)
chk("뇌 15 무리 수 재현(X 1493·1463)", (int(s15["l"].sum()), int(s15["r"].sum())) == (1493, 1463))
chk("뇌 15 Y 무리 수(1507·1537)", (int((~s15["l"]).sum()), int((~s15["r"]).sum())) == (1507, 1537))
# 2~5 규칙 일치(합성): X 무리는 sign(Δ)=sign(CSI), Y 무리는 반대
n = 8
isx = {"l": np.array([1, 1, 1, 1, 0, 0, 0, 0], bool), "r": np.zeros(n, bool)}
Z = {"csi_l": np.array([0.05, 0.3, 0.7, -0.6, 0.05, 0.3, 0.7, -0.6]), "csi_r": np.zeros(n),
     "sxi_l": np.array([0.05, -0.3, 0.7, 0.2, 0.05, 0.3, -0.7, 0.2]), "sxi_r": np.zeros(n),
     "resp_l": np.ones(n), "resp_r": np.zeros(n)}
D = {"l": np.array([+1.0, +2.0, +3.0, -4.0, -1.0, -2.0, -3.0, +4.0]), "r": np.zeros(n)}   # 모두 무리 규칙대로
a = P.rule_agree(D, Z, isx)
chk("모두 맞음 — 구간별 (1/1·2/2... ) 합 8/8", sum(v[0] for v in a.values()) == 8 and sum(v[1] for v in a.values()) == 8)
chk("구간 배정: <0.1 2개·0.1~0.5 2개·≥0.5 4개", [a[i][1] for i in range(3)] == [2, 2, 4])
D2 = {"l": -D["l"], "r": np.zeros(n)}
a2 = P.rule_agree(D2, Z, isx)
chk("모두 반대 → 0/8", sum(v[0] for v in a2.values()) == 0)
Z0 = dict(Z); Z0["resp_l"] = np.zeros(n)
chk("비반응 KC 는 세지 않음", sum(v[1] for v in P.rule_agree(D, Z0, isx).values()) == 0)
# 6 XOR 일치: sign(Δ)=sign(SXI)
ax = P.xor_agree(D, Z)
want = int(np.sum(np.sign(D["l"]) == np.sign(Z["sxi_l"])))
chk("XOR 일치 수 = 직접 계산(%d)" % want, sum(v[0] for v in ax.values()) == want)
print("합성 %d/%d" % (ok, tot))
sys.exit(0 if ok == tot else 1)
