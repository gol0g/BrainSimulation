#!/usr/bin/env python3
"""E185 판정 — 원시 특징 범주 개념의 독립 확증(뇌 16~20)과 용량-반응(500 대 1,500시행). 규칙은 research/experiments/logs/E185/criteria_fixed.txt.
입력(뇌 16~20, 형성 전 U 표현):
  학습 로그 logs/E185/train_b{B}_{500|1500}.log — '[범주 과제] 시행 N 중 범주 B(같은 쪽) n · A(교차) m', '[범주 과제 점검]' 3줄, '[학습]', '[E153 종류 입력 적재] … 검증 일치'.
  평가 로그 logs/E185/ev_b{B}_{learn500|learn1500|none}.log — RCEVAL 4줄, '=> DECOMP … rceval … pushed=N', 종류 입력 적재 줄.
지표: e^T_{c,s} = m(학습T, c, s) − m(무학습, c, s), 1e-4 정수(로그 4자리). + = 같은 쪽, − = 교차.
뇌 단위:
  확증(1,500) = 획득(e_A,train ≤ −1000 이면서 e_B,train ≥ +1000) 이면서 일반화(e_A,new ≤ −500·e_B,new ≥ +500·2·e_A,new ≤ e_A,train·2·e_B,new ≥ e_B,train) — E184 규칙
  용량-반응 = e1500_A,new − e500_A,new ≤ −200 이면서 e1500_B,new − e500_B,new ≥ +200
판정(우선순위): 조작검증 실패 → 보류 / 확증 ≥ 4/5 이면서 용량-반응 ≥ 4/5 → 확증·용량-반응 / 확증 ≥ 4/5 → 확증(용량-반응 미확인)
  / 1,500 사례 한정 ≥ 4/5 → 확증 실패(사례 한정) / 1,500 한 범주만 A·B ≥ 4/5 → 확증 실패(한 범주만) / 1,500 획득 실패 ≥ 4/5 → 확증 실패(획득 실패) / 그 밖 보류.
실행: python3 scripts/judge_e185.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
TRIALS = 96
NT = {"500": (500, 200, 300), "1500": (1500, 600, 900)}


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    rc, pushed, kt, sat, fw = {}, None, 0, False, 0
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
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                kt += 1
            elif ln.startswith("[E184 특징 가중치 적재]"):
                fw += 1
    return rc, pushed, kt, sat, fw


def read_train(path):
    cnt = None; chk = []; done = False; kt = 0; fw = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[범주 과제] 시행 "):
                p = ln.split()
                cnt = (int(p[3]), int(ln.split("범주 B(같은 쪽) ")[1].split()[0]), int(ln.split("A(교차) ")[1].split()[0]))
            elif ln.startswith("[범주 과제 점검]"):
                chk.append((int(ln.split("행동 창 끝 특징 발화 ")[1].split()[0]), int(ln.split("시행 끝 ")[1].split()[0])))
            elif ln.startswith("[학습] "):
                done = True
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                kt += 1
            elif ln.startswith("[E184 특징 가중치 적재]"):
                fw += 1
    return cnt, chk, done, kt, fw


def classify(e):
    aT, aN, bT, bN = e[("A", "train")], e[("A", "new")], e[("B", "train")], e[("B", "new")]
    acq = aT <= -1000 and bT >= 1000
    gen = aN <= -500 and bN >= 500 and 2 * aN <= aT and 2 * bN >= bT
    return {"conf": acq and gen, "memo": acq and not gen, "A": aT <= -1000 and bT < 1000,
            "B": bT >= 1000 and aT > -1000, "fail": aT > -1000 and bT < 1000}


def main():
    ok, why, E = True, [], {}
    for b in BRAINS:
        try:
            tr = {T: read_train(os.path.join(EXP, "logs", "E185", "train_b%d_%s.log" % (b, T))) for T in NT}
            ev = {w: read_eval(os.path.join(EXP, "logs", "E185", "ev_b%d_%s.log" % (b, w))) for w in ("learn500", "learn1500", "none")}
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        for T, (cnt, chk, done, kt, fw) in tr.items():
            n, lo, hi = NT[T]
            if cnt is None or cnt[0] != n or cnt[1] + cnt[2] != n or not (lo <= cnt[1] <= hi) or not done:
                ok = False; why.append("b%d 학습 %s 범주 시행 %s" % (b, T, cnt))
            if len(chk) != 3 or not all(x > 0 and y == 0 for x, y in chk):
                ok = False; why.append("b%d 학습 %s 특징 점검 %s" % (b, T, chk))
            if kt < 1 or fw != 0:
                ok = False; why.append("b%d 학습 %s 적재(종류 %d·특징 %d)" % (b, T, kt, fw))
        m = {}
        for w, (rc, pushed, kt, sat, fw) in ev.items():
            if set(rc) != {("A", "train"), ("A", "new"), ("B", "train"), ("B", "new")}:
                ok = False; why.append("b%d %s RCEVAL 결측" % (b, w)); continue
            if any(t != TRIALS or fo != TRIALS for _, t, fo in rc.values()):
                ok = False; why.append("b%d %s 시행·특징 켬 수" % (b, w))
            if pushed != (0 if w == "none" else 8):
                ok = False; why.append("b%d %s pushed %s" % (b, w, pushed))
            if kt < 1 or fw != 0 or sat:
                ok = False; why.append("b%d %s 적재 %d·특징 %d·포화 %s" % (b, w, kt, fw, sat))
            for k, v in rc.items():
                m[(w,) + k] = v[0]
        if len(m) != 12:
            continue
        E[b] = {T: {(c, s): m[("learn" + T, c, s)] - m[("none", c, s)] for c in "AB" for s in ("train", "new")} for T in NT}
    for b in BRAINS:
        if b in E:
            e5, e15 = E[b]["500"], E[b]["1500"]
            print("b%d 500: A %+d/%+d B %+d/%+d · 1500: A %+d/%+d B %+d/%+d (노출/새, 1e-4) · 새 사례 증가 A %+d B %+d"
                  % (b, e5[("A", "train")], e5[("A", "new")], e5[("B", "train")], e5[("B", "new")],
                     e15[("A", "train")], e15[("A", "new")], e15[("B", "train")], e15[("B", "new")],
                     e15[("A", "new")] - e5[("A", "new")], e15[("B", "new")] - e5[("B", "new")]))
    if not ok or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cl = {b: classify(E[b]["1500"]) for b in BRAINS}
    conf = sum(cl[b]["conf"] for b in BRAINS)
    dose = sum(E[b]["1500"][("A", "new")] - E[b]["500"][("A", "new")] <= -200 and E[b]["1500"][("B", "new")] - E[b]["500"][("B", "new")] >= 200 for b in BRAINS)
    memo = sum(cl[b]["memo"] for b in BRAINS); oA = sum(cl[b]["A"] for b in BRAINS); oB = sum(cl[b]["B"] for b in BRAINS); fail = sum(cl[b]["fail"] for b in BRAINS)
    c500 = sum(classify(E[b]["500"])["conf"] for b in BRAINS)
    print("조작검증 통과 | 확증(1,500) %d/5 · 용량-반응 %d/5 · (참고) 500 에서 개념 규칙 %d/5 · 1,500 사례 한정 %d · A만 %d · B만 %d · 획득 실패 %d"
          % (conf, dose, c500, memo, oA, oB, fail))
    v = ("확증·용량-반응(개념 조건 1~4, 범주 수준)" if conf >= 4 and dose >= 4 else "확증(용량-반응 미확인)" if conf >= 4
         else "확증 실패(사례 한정)" if memo >= 4 else "확증 실패(한 범주만)" if (oA >= 4 or oB >= 4) else "확증 실패(획득 실패)" if fail >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
