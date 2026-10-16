#!/usr/bin/env python3
"""E198 판정 — L4 학습 규칙 형식: 학습 가중치의 방향(부호)만 살리면 범주 × 맥락 XOR 이 행동에 서는가. 규칙은 research/experiments/logs/E198/criteria_fixed.txt.
입력(뇌 10~14): 평가 logs/E198/ev_b{B}_{none|sgnR|lsgnR|lsgnA}_{X|Y}.log, 지수 traces/E196/sxi_b{B}.npz(E196 −4900 측정),
  학습 가중치 traces/E193/w_b{B}.npz(E193 3,000시행), 초기 연결 스냅숏 traces/E176/snap/conn_b{B}.npz(KC→motor __pre), 회귀 기준 logs/E197/ev_b{B}_{none|sgnR}_{X|Y}.log.
학습 차 Δ_k = mean(w_교차[k]) − mean(w_같은쪽[k]) — 스냅숏 pre 로 KC 별 묶음(행 묶음만 쓰므로 행 안 순서 무관), 9자리 반올림, 한쪽 시냅스 없는 KC 는 0.
지표(1e-4 정수): e = m(팔) − m(none). XOR 방향 a = −e(X·A, Y·B), +e(X·B, Y·A).
조작검증(하나라도 어기면 보류): E196·E197 과 같은 평가 줄 검사(judge_e196.read_eval), pushed 0(none)·4(팔), '[E196 오라클]' 줄 arm·θ 0.50·파일·쪽마다 교차/같은 쪽/대상 수
  = 다시 센 값(sgnR: 반응·SXI 부호; lsgnR: 반응·Δ 부호; lsgnA: Δ ≠ 0·Δ 부호), 장치 검증 줄; lsgn 팔은 '[E198 학습 부호]' 줄(가중치 파일 = w_b{B}.npz, agree_l·r = 다시 센 부호 일치 수)과
  '[E198 연결 확인]' 줄(KC→motor 4집단 n·pre_sum = 스냅숏 __pre 의 크기·합); none·sgnR 회귀(E197) 뇌마다 8값 |차| ≤ 2; 조작 무효 의심: lsgnR 효과 8값 모두 |e| ≤ 2 인 뇌.
뇌 단위(lsgnR·lsgnA·sgnR): 획득 칸 n = #{a_train ≥ 1000}; 획득 n 4 / 부분 2~3 / 실패 ≤ 1.
종합(우선순위): 조작검증 실패 → 보류 / lsgnR 획득 ≥ 4/5 → 학습 방향 충분(크기 병목, H119) / lsgnR 실패 ≥ 4/5 → 학습 방향 부족(H119-null) / lsgnR 부분 ≥ 4/5 → 부분 / 그 밖 보류.
(보고) 반응 KC 부호 일치율 = agree / #(반응·Δ ≠ 0·SXI ≠ 0), lsgnA·sgnR 획득 수.
실행: python3 scripts/judge_e198.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from judge_e196 import CELLS, THETA, read_eval  # noqa: E402 — 검증된 평가 줄 파서

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = ("sgnR", "lsgnR", "lsgnA")
REG = ("none", "sgnR")


def deltas(wfile, snapfile, n_kc=3000):
    W = np.load(wfile)
    S = np.load(snapfile)
    out, conn = {}, {}
    for kn in "lr":
        oc = "r" if kn == "l" else "l"
        g = {}
        for m in (oc, kn):
            nm = "kc_%s_to_motor_%s" % (kn, m)
            pre = S[nm + "__pre"].astype(np.int64)
            w = W[nm].astype(np.float64)
            if pre.size != w.size:
                raise ValueError("%s 크기 %d vs %d" % (nm, pre.size, w.size))
            cnt = np.bincount(pre, minlength=n_kc)
            g[m] = (np.bincount(pre, weights=w, minlength=n_kc) / np.maximum(cnt, 1), cnt > 0)
            conn[nm] = (int(pre.size), int(pre.sum()))
        both = g[oc][1] & g[kn][1]
        out[kn] = np.round(np.where(both, g[oc][0] - g[kn][0], 0.0), 9)
    return out, conn


def expect(npz, arm, D=None):
    Z = np.load(npz)
    res, agr = {}, {}
    for sd in "lr":
        s = np.round(Z["sxi_" + sd], 9)
        if arm == "sgnR":
            v, el = s, Z["resp_" + sd].astype(bool)
        else:
            v = D[sd]
            el = (v != 0) if arm == "lsgnA" else Z["resp_" + sd].astype(bool)
        res["cross_" + sd] = int((el & (v > 0)).sum())
        res["same_" + sd] = int((el & (v < 0)).sum())
        res["resp_" + sd] = int(el.sum())
        if arm != "sgnR":
            agr["agree_" + sd] = int((el & (v != 0) & (s != 0) & (np.sign(v) == np.sign(s))).sum())
            agr["den_" + sd] = int((el & (v != 0) & (s != 0)).sum())
    return res, agr


def extra_lines(path):
    lr, cn = None, None
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E198 학습 부호]"):
            m = re.search(r"가중치=(\S+) .*agree_l=(\d+) agree_r=(\d+)", ln)
            lr = (m.group(1), int(m.group(2)), int(m.group(3))) if m else ("?", -1, -1)
        elif ln.startswith("[E198 연결 확인]"):
            cn = {nm: (int(a), int(b)) for nm, a, b in re.findall(r"(kc_[lr]_to_motor_[lr]) n=(\d+) pre_sum=(\d+)", ln)}
    return lr, cn


def main():
    ok, why, E, AG = True, [], {}, {}
    for b in BRAINS:
        npz = os.path.join(EXP, "traces", "E196", "sxi_b%d.npz" % b)
        wf = os.path.join(EXP, "traces", "E193", "w_b%d.npz" % b)
        sf = os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b)
        try:
            D, conn = deltas(wf, sf)
            ev = {(w, x): read_eval(os.path.join(EXP, "logs", "E198", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in ("none",) + ARMS for x in "XY"}
            ref = {(w, x): read_eval(os.path.join(EXP, "logs", "E197", "ev_b%d_%s_%s.log" % (b, w, x)), x) for w in REG for x in "XY"}
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        m = {}
        for (w, x), r in ev.items():
            good = (set(r["rc"]) == {(c, s) for c in "AB" for s in ("train", "new")} and not r["bad"] and not r["sat"] and r["kt"] >= 1
                    and r["fw"] == 0 and r["pushed"] == (0 if w == "none" else 4))
            note = ""
            if w != "none":
                exp_c, exp_a = expect(npz, w, D)
                o = r["orc"] or {}
                try:
                    if o.get("arm") != w or abs(float(o.get("theta", "nan")) - THETA) > 1e-9 or os.path.basename(o.get("file", "")) != os.path.basename(npz):
                        good, note = False, "팔·θ·파일"
                    elif any(int(o.get(k, -1)) != v for k, v in exp_c.items()):
                        good, note = False, "오라클 수 %s 대 %s" % ({k: o.get(k) for k in exp_c}, exp_c)
                except ValueError:
                    good, note = False, "오라클 줄 값"
                good = good and r["ver"] >= 1
                if w in ("lsgnR", "lsgnA"):
                    lr, cn = extra_lines(os.path.join(EXP, "logs", "E198", "ev_b%d_%s_%s.log" % (b, w, x)))
                    if lr is None or os.path.basename(lr[0]) != os.path.basename(wf) or (lr[1], lr[2]) != (exp_a["agree_l"], exp_a["agree_r"]):
                        good, note = False, "학습 부호 줄 %s 대 %s" % (lr, exp_a)
                    if cn is None or cn != conn:
                        good, note = False, "연결 확인 %s 대 스냅숏 %s" % (cn, conn)
                    if w == "lsgnR":
                        AG[b] = exp_a
            if not good:
                ok = False
                why.append("b%d %s %s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s·검증 %d %s)"
                           % (b, w, x, len(r["rc"]), r["bad"], r["sat"], r["kt"], r["fw"], r["pushed"], r["ver"], note))
                continue
            for (c, s), v in r["rc"].items():
                m[(w, x, c, s)] = v
        for w in REG:
            for x in "XY":
                for c in "AB":
                    for s in ("train", "new"):
                        if (w, x, c, s) in m and (c, s) in ref[(w, x)]["rc"] and abs(m[(w, x, c, s)] - ref[(w, x)]["rc"][(c, s)]) > 2:
                            ok = False
                            why.append("b%d %s 회귀 %s %s %s %d vs E197 %d" % (b, w, x, c, s, m[(w, x, c, s)], ref[(w, x)]["rc"][(c, s)]))
        if len(m) == 32:
            E[b] = {(w, x, c, s): m[(w, x, c, s)] - m[("none", x, c, s)] for w in ARMS for x in "XY" for c in "AB" for s in ("train", "new")}
            if all(abs(E[b][("lsgnR", x, c, s)]) <= 2 for x in "XY" for c in "AB" for s in ("train", "new")):
                ok = False
                why.append("b%d lsgnR 효과 8값 모두 |e| ≤ 2 — 조작 무효 의심" % b)
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cnt = {w + k: 0 for w in ARMS for k in ("_acq", "_part", "_fail")}
    for b in BRAINS:
        e = E[b]
        ag = AG[b]
        line = "b%d 반응 KC 부호 일치 %d/%d(%.3f)·%d/%d(%.3f)" % (b, ag["agree_l"], ag["den_l"], ag["agree_l"] / max(ag["den_l"], 1), ag["agree_r"], ag["den_r"], ag["agree_r"] / max(ag["den_r"], 1))
        for w in ("lsgnR", "lsgnA", "sgnR"):
            a = {(x, c, s): sg * e[(w, x, c, s)] for x, c, sg in CELLS for s in ("train", "new")}
            n = sum(a[(x, c, "train")] >= 1000 for x, c, _ in CELLS)
            cnt[w + "_acq"] += n == 4
            cnt[w + "_part"] += 2 <= n <= 3
            cnt[w + "_fail"] += n <= 1
            line += " | %s X·A %+d/%+d X·B %+d/%+d Y·A %+d/%+d Y·B %+d/%+d 획득 칸 %d" % (
                w, a[("X", "A", "train")], a[("X", "A", "new")], a[("X", "B", "train")], a[("X", "B", "new")],
                a[("Y", "A", "train")], a[("Y", "A", "new")], a[("Y", "B", "train")], a[("Y", "B", "new")], n)
        print(line)
    print("조작검증 통과(none·sgnR 회귀 뇌마다 8값·Δ 재계산·연결 확인 포함) | lsgnR 획득 %d · 부분 %d · 실패 %d | lsgnA 획득 %d · 부분 %d · 실패 %d | sgnR 획득 %d · 부분 %d · 실패 %d"
          % tuple(cnt[w + k] for w in ("lsgnR", "lsgnA", "sgnR") for k in ("_acq", "_part", "_fail")))
    if cnt["lsgnR_acq"] >= 4:
        v = "학습 방향 충분 — 크기 병목(H119)"
    elif cnt["lsgnR_fail"] >= 4:
        v = "학습 방향 부족(H119-null)"
    elif cnt["lsgnR_part"] >= 4:
        v = "부분"
    else:
        v = "보류"
    print("판정: " + v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
