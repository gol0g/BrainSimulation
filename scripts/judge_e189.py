#!/usr/bin/env python3
"""E189 판정 — 원시 특징 범주의 간섭 하 유지(K95 의 범주판). 규칙은 research/experiments/logs/E189/criteria_fixed.txt.
입력(뇌 10~14): 학습 logs/E189/train_b{B}.log(catrule 1,500 → 둘째 쌍 1,500), 평가 logs/E189/ev_b{B}_{learn|none}.log(RCEVAL 8줄 — A·B·C·D × train·new).
첫 쌍 기준 R = E184 U(같은 뇌·같은 시드 — 앞 1,500 같은 시행열; A·B 평가 자리도 같음): logs/E184/ev_b{B}_U_{learn|none}.log.
지표(1e-4): e = m(학습) − m(무학습). 첫 쌍 정답 A 교차(−)·B 같은 쪽(+), 둘째 쌍 C 같은 쪽(+)·D 교차(−).
뇌 단위: 둘째 쌍 획득(전제) = e_C,train ≥ +1000 이면서 e_D,train ≤ −1000; 첫 쌍 기준 = R_A,train ≤ −1000 이면서 R_B,train ≥ +1000;
  유지 = 첫 쌍 기준·전제 충족 이면서 2·e_A,train ≤ R_A,train 이고 2·e_B,train ≥ R_B,train — 유지 몫 r = e/R ≥ 0.5 두 범주;
  간섭 = 첫 쌍 기준·전제 충족 이면서 유지 아님(한 범주라도 r < 0.5). 기준·전제 미충족 뇌는 어느 쪽에도 세지 않는다. (보고) 완전 유지 r ≥ 0.8 두 범주.
판정(우선순위): 조작검증 실패 → 보류 / 전제 미달(둘째 쌍 획득 < 4/5) → 보류(전제 미달) / 유지(H110) ≥ 4/5 / 간섭(H110-null) ≥ 4/5 / 그 밖 보류.
실행: python3 scripts/judge_e189.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    rc, pushed, kt, fw, sat, bad = {}, None, 0, 0, False, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                rc[(kv["cat"], kv["set"])] = q4(kv["mod"])
                if int(kv["trials"]) != TRIALS or int(kv["feat_on"]) != TRIALS:
                    bad = True
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
    return rc, pushed, kt, fw, sat, bad


def read_train(path):
    r = {"start": None, "n2": None, "cat": None, "chk": [], "done": False, "kt": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[둘째 범주 쌍] 시행 "):
                r["start"] = int(ln.split("시행 ")[1].split()[0])
            elif ln.startswith("[둘째 범주 쌍] 둘째 쌍 시행 "):
                r["n2"] = (int(ln.split("둘째 쌍 시행 ")[1].split()[0]), int(ln.split("(시행 ")[1].split()[0]), int(ln.split("전체 ")[1].split(")")[0]))
            elif ln.startswith("[범주 과제] 시행 "):
                p = ln.split()
                r["cat"] = (int(p[3]), int(ln.split("범주 B(같은 쪽) ")[1].split()[0]))
            elif ln.startswith("[범주 과제 점검]"):
                r["chk"].append((int(ln.split("행동 창 끝 특징 발화 ")[1].split()[0]), int(ln.split("시행 끝 ")[1].split()[0])))
            elif ln.startswith("[학습] "):
                r["done"] = True
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                r["kt"] += 1
    return r


def main():
    ok, why, rows = True, [], {}
    for b in BRAINS:
        try:
            tr = read_train(os.path.join(EXP, "logs", "E189", "train_b%d.log" % b))
            L = read_eval(os.path.join(EXP, "logs", "E189", "ev_b%d_learn.log" % b))
            N = read_eval(os.path.join(EXP, "logs", "E189", "ev_b%d_none.log" % b))
            RL = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_learn.log" % b))
            RN = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_none.log" % b))
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        if not (tr["start"] == 1500 and tr["n2"] == (1500, 1500, 3000) and tr["cat"] and tr["cat"][0] == 3000 and 1200 <= tr["cat"][1] <= 1800
                and tr["done"] and tr["kt"] >= 1):
            ok = False; why.append("b%d 학습 줄(%s %s %s)" % (b, tr["start"], tr["n2"], tr["cat"]))
        if len(tr["chk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["chk"]):
            ok = False; why.append("b%d 특징 점검 %s" % (b, tr["chk"]))
        for nm, (rc, pushed, kt, fw, sat, bad) in (("learn", L), ("none", N)):
            if set(rc) != {(c, s) for c in "ABCD" for s in ("train", "new")} or bad or sat or kt < 1 or fw != 0 or pushed != (8 if nm == "learn" else 0):
                ok = False; why.append("b%d %s 평가(줄 %d)" % (b, nm, len(rc)))
        if len(RL[0]) != 4 or len(RN[0]) != 4:
            ok = False; why.append("b%d 첫 쌍 기준(E184) 결측" % b)
        if ok:
            rows[b] = ({k: L[0][k] - N[0][k] for k in L[0]}, {k: RL[0][k] - RN[0][k] for k in RL[0]})
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {"prem": 0, "ret": 0, "intf": 0, "full": 0, "base": 0}
    for b in BRAINS:
        e, R = rows[b]
        prem = e[("C", "train")] >= 1000 and e[("D", "train")] <= -1000
        base = R[("A", "train")] <= -1000 and R[("B", "train")] >= 1000          # 첫 쌍 기준(E184 U)이 획득 수준 — 아니면 셈 밖
        kept = 2 * e[("A", "train")] <= R[("A", "train")] and 2 * e[("B", "train")] >= R[("B", "train")]
        ret = base and prem and kept
        intf = base and prem and not kept
        full = ret and 5 * e[("A", "train")] <= 4 * R[("A", "train")] and 5 * e[("B", "train")] >= 4 * R[("B", "train")]
        n["prem"] += prem
        n["base"] += base
        n["ret"] += ret
        n["intf"] += intf
        n["full"] += full
        rA = e[("A", "train")] / R[("A", "train")] if R[("A", "train")] else float("nan")
        rB = e[("B", "train")] / R[("B", "train")] if R[("B", "train")] else float("nan")
        print("b%d 둘째 쌍 C %+d/%+d D %+d/%+d | 첫 쌍 뒤 A %+d/%+d B %+d/%+d · 앞(E184) A %+d B %+d | 유지 몫 r_A %.2f r_B %.2f"
              % (b, e[("C", "train")], e[("C", "new")], e[("D", "train")], e[("D", "new")], e[("A", "train")], e[("A", "new")],
                 e[("B", "train")], e[("B", "new")], R[("A", "train")], R[("B", "train")], rA, rB))
    print("조작검증 통과 | 둘째 쌍 획득 %d/5 · 첫 쌍 기준 획득 %d/5 · 유지 %d/5 · 간섭 %d/5 · (보고) 완전 유지 %d/5"
          % (n["prem"], n["base"], n["ret"], n["intf"], n["full"]))
    if n["prem"] < 4:
        print("판정: 보류(전제 미달 — 둘째 쌍 획득 %d/5)" % n["prem"])
        return 0
    v = "유지(H110)" if n["ret"] >= 4 else "간섭(H110-null)" if n["intf"] >= 4 else "보류"
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
