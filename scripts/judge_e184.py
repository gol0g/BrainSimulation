#!/usr/bin/env python3
"""E184 판정 — 원시 특징 범주 의존 행동 학습과 새 사례 일반화. 규칙은 research/experiments/logs/E184/criteria_fixed.txt.
입력(팔 a ∈ ARMS, 뇌 10~14):
  학습 로그 logs/E184/train_b{B}_{a}.log — '[범주 과제] 시행 N 중 범주 B(같은 쪽) n · A(교차) m', '[범주 과제 점검]' 3줄(행동 창 끝 특징 발화 > 0, 시행 끝 0), '[학습]' 줄.
  평가 로그 logs/E184/ev_b{B}_{a}_{learn|none}.log — RCEVAL 4줄(cat A·B × set train·new: mod, trials, feat_on, 진단), '=> DECOMP ... rceval ... pushed=N'.
지표: e_{c,s} = m(learn, c, s) − m(none, c, s) (변조폭: + = 같은 쪽, − = 교차; 범주 A 정답 교차, B 정답 같은 쪽).
뇌 단위(주 팔 PRIMARY):
  획득 = e_A,train ≤ −0.10 이면서 e_B,train ≥ +0.10
  일반화 = e_A,new ≤ −0.05 이면서 e_B,new ≥ +0.05 이면서 e_A,new ≥ 0.5·e_A,train 크기(즉 e_A,new/e_A,train ≥ 0.5) 이면서 e_B,new/e_B,train ≥ 0.5
판정(우선순위): 조작검증 실패 → 보류 / 개념 수준(H105) 획득·일반화 ≥ 4/5 / 사례 한정(H105-null) 획득·일반화 아님 ≥ 4/5
  / 한 범주만 A(e_A,train ≤ −0.10, e_B,train < +0.10) 또는 B ≥ 4/5 / 획득 실패(둘 다 아님) ≥ 4/5 / 그 밖 보류.
경계 비교는 1e-4 단위 정수(로그 4자리).
실행: python3 scripts/judge_e184.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
ARMS = ("U", "F")      # 사전등록 고정: U = 형성 전(초기 가중치, 주 팔 — E183 사전 결정), F = 형성(E183 O 절차 재실행 --rc-save, 부 팔 — 판정 밖 비교)
PRIMARY = "U"
TRIALS = 96
NTRIAL = 1500


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    rc, pushed, loaded = {}, None, 0
    sat = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                rc[(kv["cat"], kv["set"])] = (q4(kv["mod"]), int(kv["trials"]), int(kv["feat_on"]))
                for seg in ln.split(" | ")[1:]:
                    for part in seg.split("motor L/R ")[1:]:
                        a, b = part.split()[0].split("/")
                        if q4(a) >= 6600 and q4(b) >= 6600:
                            sat = True
            elif ln.startswith("=> DECOMP ") and "rceval" in ln:
                pushed = int(ln.split("pushed=")[1].split()[0])
            elif ln.startswith("[E184 특징 가중치 적재]") and "검증 일치" in ln:
                loaded += 1
    return rc, pushed, loaded, sat


def read_train(path):
    nB = nA = None; chk = []; done = False; loaded = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[범주 과제] 시행 "):
                p = ln.split()
                n = int(p[3]); nB = int(ln.split("범주 B(같은 쪽) ")[1].split()[0]); nA = int(ln.split("A(교차) ")[1].split()[0])
                if nA + nB != n:
                    raise ValueError("범주 시행 수 불일치")
            elif ln.startswith("[범주 과제 점검]"):
                a = int(ln.split("행동 창 끝 특징 발화 ")[1].split()[0]); b = int(ln.split("시행 끝 ")[1].split()[0])
                chk.append((a, b))
            elif ln.startswith("[학습] "):
                done = True
            elif ln.startswith("[E184 특징 가중치 적재]") and "검증 일치" in ln:
                loaded += 1
    return nA, nB, chk, done, loaded


def main():
    ok, why, rows = True, [], {}
    for a in ARMS:
        for b in BRAINS:
            try:
                nA, nB, chk, done, ldt = read_train(os.path.join(EXP, "logs", "E184", "train_b%d_%s.log" % (b, a)))
                ev = {w: read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_%s_%s.log" % (b, a, w))) for w in ("learn", "none")}
            except (OSError, ValueError, IndexError, KeyError) as ex:
                print("판정: 보류(결측 — b%d %s %s)" % (b, a, type(ex).__name__))
                return 0
            if not done or nA is None or nA + nB != NTRIAL or not (600 <= nB <= 900):
                ok = False; why.append("b%d %s 학습 범주 시행(%s/%s)" % (b, a, nA, nB))
            if len(chk) != 3 or not all(x > 0 and y == 0 for x, y in chk):
                ok = False; why.append("b%d %s 학습 특징 점검 %s" % (b, a, chk))
            if (a == "F") != (ldt >= 1):
                ok = False; why.append("b%d %s 학습 특징 가중치 적재 %d" % (b, a, ldt))
            m = {}
            for w in ("learn", "none"):
                rc, pushed, ld, sat = ev[w]
                if set(rc) != {("A", "train"), ("A", "new"), ("B", "train"), ("B", "new")}:
                    ok = False; why.append("b%d %s %s RCEVAL 결측" % (b, a, w)); continue
                if any(t != TRIALS or fo != TRIALS for _, t, fo in rc.values()):
                    ok = False; why.append("b%d %s %s 시행·특징 켬 수" % (b, a, w))
                if pushed != (8 if w == "learn" else 0):
                    ok = False; why.append("b%d %s %s pushed %s" % (b, a, w, pushed))
                if (a == "F") != (ld >= 1):
                    ok = False; why.append("b%d %s %s 특징 가중치 적재 %d" % (b, a, w, ld))
                if sat:
                    ok = False; why.append("b%d %s %s motor 포화" % (b, a, w))
                for k, v in rc.items():
                    m[(w,) + k] = v[0]
            if len(m) != 8:
                continue
            e = {(c, s): m[("learn", c, s)] - m[("none", c, s)] for c in "AB" for s in ("train", "new")}
            rows[(a, b)] = e
    for (a, b), e in sorted(rows.items()):
        print("%s b%d e_A train %+d new %+d · e_B train %+d new %+d (1e-4)" % (a, b, e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]))
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    acq = gen = memo = onlyA = onlyB = fail = 0
    for b in BRAINS:
        e = rows[(PRIMARY, b)]
        aT, aN, bT, bN = e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]
        acq_b = aT <= -1000 and bT >= 1000
        gen_b = aN <= -500 and bN >= 500 and 2 * aN <= aT and 2 * bN >= bT   # aT<0: aN/aT ≥ 0.5 ⇔ 2·aN ≤ aT
        acq += acq_b
        gen += acq_b and gen_b
        memo += acq_b and not gen_b
        onlyA += aT <= -1000 and bT < 1000
        onlyB += bT >= 1000 and aT > -1000
        fail += aT > -1000 and bT < 1000
    print("조작검증 통과 | 주 팔 %s: 획득 %d/5 · 획득+일반화 %d/5 · 사례 한정 %d/5 · A만 %d/5 · B만 %d/5 · 획득 실패 %d/5"
          % (PRIMARY, acq, gen, memo, onlyA, onlyB, fail))
    v = ("개념 수준(H105)" if gen >= 4 else "사례 한정(H105-null)" if memo >= 4 else "한 범주만 A(교차)" if onlyA >= 4
         else "한 범주만 B(같은 쪽)" if onlyB >= 4 else "획득 실패" if fail >= 4 else "보류")
    # 부 팔(판정 밖): 형성 F 의 같은 분류와 새 사례 효과 F − U — 판정 줄 앞에 출력(판정 줄은 항상 마지막)
    if "F" in ARMS:
        fg = sum(1 for b in BRAINS if rows[("F", b)][("A", "train")] <= -1000 and rows[("F", b)][("B", "train")] >= 1000
                 and rows[("F", b)][("A", "new")] <= -500 and rows[("F", b)][("B", "new")] >= 500
                 and 2 * rows[("F", b)][("A", "new")] <= rows[("F", b)][("A", "train")] and 2 * rows[("F", b)][("B", "new")] >= rows[("F", b)][("B", "train")])
        d = ["b%d A %+d B %+d" % (b, rows[("F", b)][("A", "new")] - rows[("U", b)][("A", "new")], rows[("F", b)][("B", "new")] - rows[("U", b)][("B", "new")]) for b in BRAINS]
        print("부 팔 F(판정 밖): 획득+일반화 %d/5 | 새 사례 효과 F − U(1e-4): %s" % (fg, " · ".join(d)))
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
