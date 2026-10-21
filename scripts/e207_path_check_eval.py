#!/usr/bin/env python3
"""E207 경로 검사 재판정(정정 1) — e207_path_check.sh 가 남긴 로그(logs/E207/pathcheck, 뇌 15)를 정정 1 규칙으로 다시 검사한다. GPU 실행 없음.
정정 1(2026-10-22, criteria_fixed.txt): 노출 0 에서도 한 프로세스의 두 측정(J_pre = 새 뇌 첫 측정, J_post = 둘째 측정)이 이력 때문에 달라
'J_pre = J_post' 를 조작검증에서 빼고 (보고) 측정 이력 차로 돌렸다. 나머지 점검(회귀·RCCHK·노출 0·세계·동작점·F 적재·측정 맥락·같은 값)은 첫 판 그대로 —
같은 값 점검만 J_pre·J_post 각각. 코드 경로(과제 진입점)는 바뀌지 않았으므로 첫 판 로그를 그대로 쓴다.
실행: python3 scripts/e207_path_check_eval.py (저장소 루트에서)"""
import os
import sys

import numpy as np

R = os.getcwd()
sys.path.insert(0, os.path.join(R, "scripts"))
import judge_e207 as J7  # noqa: E402

OUT = os.path.join(R, "research/experiments/logs/E207/pathcheck")
WD = os.path.join(R, "research/experiments/traces/E207/pathcheck")
FORM6 = os.path.join(R, "research/experiments/logs/E206/pathcheck/form_f_b15.log")
F15 = os.path.join(R, "research/experiments/traces/E206/pathcheck/feat_formed_b15.npz")


def L(fn):
    return os.path.join(OUT, fn)


def main():
    ok = True
    # (a) 회귀 — 첫 판과 같은 계산
    A, B = np.load(os.path.join(WD, "F15_regress.npz")), np.load(F15)
    same_w = sorted(A.files) == sorted(B.files) and all(np.array_equal(A[k], B[k]) for k in A.files)
    rj = lambda p: [ln for ln in open(p, encoding="utf-8", errors="replace") if ln.startswith("RCJ ")]  # noqa: E731
    same_j = rj(L("a_regress_b15.log")) == rj(FORM6) and len(rj(FORM6)) == 12
    no207 = not any("E207 측정 맥락" in ln for ln in open(L("a_regress_b15.log"), encoding="utf-8", errors="replace"))
    a_ok = same_w and same_j and no207
    ok &= a_ok
    print("[재검사 (a)] 회귀 — 저장 배열 %d개 같음 %s · RCJ 12줄 같음 %s · E207 줄 없음 %s → %s" % (len(A.files), same_w, same_j, no207, "통과" if a_ok else "실패"))
    res = {}
    for N, rep, c, T in (("b", "F", "X", -1300.0), ("c", "F", "none", -1300.0), ("d", "U", "X", -2800.0)):
        r = J7.read_meas(L("%s_meas_%s_%s_b15.log" % (N, rep, c)))
        res[N] = r
        feat_ok = (len(r["feat"]) >= 1 and all(p and p.replace("\\", "/").endswith("traces/E206/pathcheck/feat_formed_b15.npz") for p in r["feat"])) if rep == "F" else (r["feat"] == [])
        base_ok = (r["rawcat"] == 1 and len(r["rcj"]) == 12 and set(r["chk"]) == {"pre", "post"}
                   and all(v == (288, 0, 0, 0, 1) for v in r["chk"].values()) and r["expo"] == 0 and r["world"] == [(38300, "5,14,15,19,29")]
                   and r["tonic"] and all(t == T for t in r["tonic"]) and feat_ok)
        if c == "X":
            cx = r["ctx"]
            ctx_ok = set(cx) == {"pre", "post"} and all(v and v[0] == "X" and v[1] == 288 and v[2] > 0 and v[3] == 0 and v[6] == "있음" and v[4] > 0 and 100 * v[5] <= v[4] for v in cx.values())
        else:
            ctx_ok = not r["ctx"]
        k_ok = base_ok and ctx_ok
        ok &= k_ok
        print("[재검사 (%s)] %s·%s — RAWCAT %d · RCJ %d · RCCHK %s · 노출 %s · 세계 %s · 동작점 %s · F 적재 %d · 측정 맥락 %s · S 첫/둘째 %d/%d · (보고) 측정 이력 차 최대 %d → %s"
              % (N, rep, c, r["rawcat"], len(r["rcj"]), sorted(r["chk"].items()), r["expo"], r["world"], sorted(set(r["tonic"])), len(r["feat"]), r["ctx"],
                 J7.score(r, 0), J7.score(r, 1), J7.hist_diff(r), "통과" if k_ok else "실패"))
    for k, nm in ((0, "J_pre"), (1, "J_post")):
        same_bc = [res["b"]["rcj"][(s, p)][k] for s in "lr" for p in J7.PAIRS] == [res["c"]["rcj"][(s, p)][k] for s in "lr" for p in J7.PAIRS]
        ok &= not same_bc
        print("[재검사 같은 값] F15 X 대 none RCJ %s 12값 같음 %s(같으면 조작 무효 의심)" % (nm, same_bc))
    c0 = J7.classify(J7.score(res["b"], 0), J7.score(res["c"], 0)); c1 = J7.classify(J7.score(res["b"], 1), J7.score(res["c"], 1))
    print("[보고] 뇌 15(표본 밖) F: 첫 측정 S_none %d · S_X %d → %s · 둘째 측정 S_none %d · S_X %d → %s ⇒ %s · J(A,B) 좌/우 첫 none %d/%d X %d/%d"
          % (J7.score(res["c"], 0), J7.score(res["b"], 0), c0, J7.score(res["c"], 1), J7.score(res["b"], 1), c1, J7.classify2(c0, c1),
             res["c"]["rcj"][("l", "AB")][0], res["c"]["rcj"][("r", "AB")][0], res["b"]["rcj"][("l", "AB")][0], res["b"]["rcj"][("r", "AB")][0]))
    print("[E207 경로 검사 재판정 — 정정 1] %s" % ("통과" if ok else "실패"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
