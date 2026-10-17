"""explore_e201_comp.means 합성 시험 — 반응/비반응/활동 가중 평균의 정답을 아는 작은 연결로 검사(결과 뒤 작성한 탐색 도구의 검증)."""
import os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(tempfile.mkdtemp())
import explore_e201_comp as X
ok = 0; bad = 0
def chk(name, got, want, tol=1e-9):
    global ok, bad
    if abs(got - want) <= tol: ok += 1
    else: bad += 1; print("실패", name, got, want)
# KC 쪽마다 10개, 경로마다 KC 하나당 시냅스 2개(순서 섞음). 반응 KC = 0~4, 발화 수 tot = 번호+1
pre = np.array([3, 0, 9, 1, 2, 8, 4, 5, 6, 7] * 2, dtype=np.int32)
W, S, Z = {}, {}, {}
for kn, m in X.PATHS:
    nm = "kc_%s_to_motor_%s" % (kn, m)
    S[nm + "__pre"] = pre
    resp = pre < 5
    if (kn, m) == ("l", "r"):
        w = np.where(resp, 200.0, 100.0)       # 반응 200 · 비반응 100 → 전체 150
    elif (kn, m) == ("l", "l"):
        w = np.where(resp, 100.0, 200.0)       # 반대
    else:
        w = np.full(len(pre), 150.0)
    W[nm] = w.astype(np.float32)
for kn in "lr":
    Z["resp_" + kn] = (np.arange(10) < 5).astype(np.int8)
    Z["tot_" + kn] = (np.arange(10) + 1).astype(np.int32)
np.savez("w.npz", **W); np.savez("s.npz", **S); np.savez("z.npz", **Z)
r = X.means("w.npz", "s.npz", "z.npz")
chk("l→r 전체", r[("l", "r")]["all"], 150.0)
chk("l→r 반응", r[("l", "r")]["resp"], 200.0)
chk("l→r 비반응", r[("l", "r")]["non"], 100.0)
chk("l→r 활동", r[("l", "r")]["act"], (200 * 15 + 100 * 40) / 55, 1e-6)
chk("l→l 반응", r[("l", "l")]["resp"], 100.0)
chk("l→l 활동", r[("l", "l")]["act"], (100 * 15 + 200 * 40) / 55, 1e-6)
chk("r→l 전체", r[("r", "l")]["all"], 150.0)
chk("r→r 비반응", r[("r", "r")]["non"], 150.0)
chk("반응 시냅스 수", r[("l", "r")]["n_resp_syn"], 10)
chk("시냅스 수", r[("l", "r")]["n_syn"], 20)
# 길이 불일치는 거부
S2 = dict(S); S2["kc_l_to_motor_l__pre"] = pre[:-1]; np.savez("s2.npz", **S2)
try:
    X.means("w.npz", "s2.npz", "z.npz"); bad += 1; print("실패 길이 불일치 통과")
except ValueError:
    ok += 1
print("합성 %d/%d" % (ok, ok + bad))
