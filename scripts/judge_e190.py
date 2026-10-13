#!/usr/bin/env python3
"""E190 판정 — 형성 표현에서 원시 특징 범주의 간섭 하 유지(K120 의 형성판). 규칙은 research/experiments/logs/E190/criteria_fixed.txt.
입력(뇌 10~14, logs/E190/): 형성 form_b{B}.log(노출 범주 A·B·C·D), 학습 train_R_b{B}.log(catrule 1,500)·train_P_b{B}.log(1,500 → 둘째 쌍 1,500),
평가 ev_b{B}_R_learn.log·ev_b{B}_P_learn.log·ev_b{B}_none.log(RCEVAL 8줄 — A·B·C·D × train·new, 모두 형성 가중치 적재).
지표(1e-4): e_X = m(X 학습) − m(무학습). 정답: A 교차(−)·B 같은 쪽(+)·C 같은 쪽(+)·D 교차(−).
뇌 단위: 첫 쌍 기준 = e_R,A,train ≤ −1000 이면서 e_R,B,train ≥ +1000; 둘째 쌍 획득(전제) = e_P,C,train ≥ +1000 이면서 e_P,D,train ≤ −1000;
  유지 = 기준·전제 충족 이면서 2·e_P,A,train ≤ e_R,A,train 이고 2·e_P,B,train ≥ e_R,B,train(유지 몫 r ≥ 0.5 두 범주); 간섭 = 기준·전제 충족 이면서 유지 아님.
판정(우선순위): 조작검증 실패 → 보류 / 첫 쌍 기준 < 4/5 → 보류(기준 미달) / 둘째 쌍 획득 < 4/5 → 보류(전제 미달) / 유지(H111) ≥ 4/5 / 간섭(H111-null) ≥ 4/5 / 그 밖 보류.
(보고) 같은 뇌 형성 전(E189) 유지 몫과의 차, 형성 전후 범주 쌍 겹침(RCJ), 기준 단계의 C·D 전이(e_R,C·e_R,D), 새 사례 값.
실행: python3 scripts/judge_e190.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
TRIALS = 96
CATS8 = {(c, s) for c in "ABCD" for s in ("train", "new")}
PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")


def q4(s):
    s = s.strip()
    neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_eval(path):
    r = {"rc": {}, "pushed": None, "kt": 0, "fw": 0, "sat": False, "bad": False}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("RCEVAL "):
                kv = {}
                for x in ln.split(" | ")[0].split()[1:]:
                    k, v = x.split("=", 1); kv[k] = v
                r["rc"][(kv["cat"], kv["set"])] = q4(kv["mod"])
                if int(kv["trials"]) != TRIALS or int(kv["feat_on"]) != TRIALS:
                    r["bad"] = True
                for seg in ln.split(" | ")[1:]:
                    for part in seg.split("motor L/R ")[1:]:
                        a, b = part.split()[0].split("/")
                        if q4(a) >= 6600 and q4(b) >= 6600:
                            r["sat"] = True
            elif ln.startswith("=> DECOMP ") and "rceval" in ln:
                r["pushed"] = int(ln.split("pushed=")[1].split()[0])
            elif ln.startswith("[E153 종류 입력 적재]") and "검증 일치" in ln:
                r["kt"] += 1
            elif ln.startswith("[E184 특징 가중치 적재]") and "검증 일치" in ln:
                r["fw"] += 1
    return r


def read_train(path):
    r = {"p2start": None, "n2": None, "cat": None, "chk": [], "done": False, "kt": 0, "fw": 0, "p2lines": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[둘째 범주 쌍]"):
                r["p2lines"] += 1
                if ln.startswith("[둘째 범주 쌍] 시행 "):
                    r["p2start"] = int(ln.split("시행 ")[1].split()[0])
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
            elif ln.startswith("[E184 특징 가중치 적재]") and "검증 일치" in ln:
                r["fw"] += 1
    return r


def read_form(path):
    r = {"expo": None, "expo_cats": None, "rcw": [], "rcj": {}, "rawcat": 0, "chk": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[원시 특징 노출] 제시 "):
                r["expo"] = int(ln.split("제시 ")[1].split("(")[0])
                r["expo_cats"] = ln.split("노출 범주 ")[1].split()[0] if "노출 범주 " in ln else "A,B"
            elif ln.startswith("RCW "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                r["rcw"].append((float(kv["dg_abs"]), int(kv["g0_init"])))
            elif ln.startswith("RCJ "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                r["rcj"][(kv["side"], kv["pair"])] = (q4(kv["J_pre"]), q4(kv["J_post"]))
            elif ln.startswith("=> RAWCAT "):
                r["rawcat"] += 1
            elif ln.startswith("RCCHK "):
                kv = dict(x.split("=", 1) for x in ln.split()[1:])
                r["chk"].append((kv["tag"], int(kv["npres"]), int(kv["feat_inactive_spikes"]), int(kv["feat_other_side"]), int(kv["goodbad"]), int(kv["g_same"])))
    return r


def train_ok(t, phase2):
    if not (t["done"] and t["kt"] >= 1 and t["fw"] >= 1 and len(t["chk"]) == 3 and all(x > 0 and y == 0 for x, y in t["chk"]) and t["cat"]):
        return False
    if phase2:
        return t["p2start"] == 1500 and t["n2"] == (1500, 1500, 3000) and t["cat"][0] == 3000 and 1200 <= t["cat"][1] <= 1800
    return t["p2lines"] == 0 and t["cat"][0] == 1500 and 600 <= t["cat"][1] <= 900


def form_ok(fm):
    return (fm["expo"] == 816 and fm["expo_cats"] == "A,B,C,D" and len(fm["rcw"]) == 2 and all(d > 0 and g == 1 for d, g in fm["rcw"])
            and set(fm["rcj"]) == {(s, p) for s in "lr" for p in PAIRS} and fm["rawcat"] == 1
            and sorted(c[0] for c in fm["chk"]) == ["post", "pre", "pre2"] and all(c[1] == 288 and c[2] == 0 and c[3] == 0 and c[4] == 0 and c[5] == 1 for c in fm["chk"]))


def eval_ok(ev, pushed):
    return set(ev["rc"]) == CATS8 and not ev["bad"] and not ev["sat"] and ev["kt"] >= 1 and ev["fw"] >= 1 and ev["pushed"] == pushed


def e189_retention(b):
    """(보고) 형성 전 유지 몫 — E189 둘째 쌍 뒤 효과 / E184 U 효과(A·B 노출). 없으면 None."""
    try:
        L = read_eval(os.path.join(EXP, "logs", "E189", "ev_b%d_learn.log" % b))["rc"]
        N = read_eval(os.path.join(EXP, "logs", "E189", "ev_b%d_none.log" % b))["rc"]
        RL = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_learn.log" % b))["rc"]
        RN = read_eval(os.path.join(EXP, "logs", "E184", "ev_b%d_U_none.log" % b))["rc"]
        return tuple((L[(c, "train")] - N[(c, "train")]) / float(RL[(c, "train")] - RN[(c, "train")]) for c in "AB")
    except (OSError, KeyError, ValueError, IndexError, ZeroDivisionError):
        return None


def main():
    ok, why, rows = True, [], {}
    for b in BRAINS:
        try:
            lg = lambda n: os.path.join(EXP, "logs", "E190", n % b)
            fm = read_form(lg("form_b%d.log"))
            tR, tP = read_train(lg("train_R_b%d.log")), read_train(lg("train_P_b%d.log"))
            eR, eP, eN = read_eval(lg("ev_b%d_R_learn.log")), read_eval(lg("ev_b%d_P_learn.log")), read_eval(lg("ev_b%d_none.log"))
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        checks = (("형성", form_ok(fm)), ("기준 학습", train_ok(tR, False)), ("둘째 쌍 학습", train_ok(tP, True)),
                  ("기준 평가", eval_ok(eR, 8)), ("둘째 쌍 평가", eval_ok(eP, 8)), ("무학습 평가", eval_ok(eN, 0)))
        for nm, c in checks:
            if not c:
                ok = False; why.append("b%d %s" % (b, nm))
        if ok:
            rows[b] = ({k: eR["rc"][k] - eN["rc"][k] for k in CATS8}, {k: eP["rc"][k] - eN["rc"][k] for k in CATS8}, fm["rcj"])
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {"base": 0, "prem": 0, "ret": 0, "intf": 0, "full": 0}
    for b in BRAINS:
        R, P, J = rows[b]
        base = R[("A", "train")] <= -1000 and R[("B", "train")] >= 1000
        prem = P[("C", "train")] >= 1000 and P[("D", "train")] <= -1000
        kept = 2 * P[("A", "train")] <= R[("A", "train")] and 2 * P[("B", "train")] >= R[("B", "train")]
        ret = base and prem and kept
        n["base"] += base
        n["prem"] += prem
        n["ret"] += ret
        n["intf"] += base and prem and not kept
        n["full"] += ret and 5 * P[("A", "train")] <= 4 * R[("A", "train")] and 5 * P[("B", "train")] >= 4 * R[("B", "train")]
        rA = P[("A", "train")] / float(R[("A", "train")]) if R[("A", "train")] else float("nan")
        rB = P[("B", "train")] / float(R[("B", "train")]) if R[("B", "train")] else float("nan")
        u = e189_retention(b)
        jm = lambda p, i: (J[("l", p)][i] + J[("r", p)][i]) / 2e4
        print("b%d 기준 A %+d/%+d B %+d/%+d (C %+d D %+d) | 둘째 쌍 C %+d/%+d D %+d/%+d | 뒤 A %+d/%+d B %+d/%+d | 유지 몫 r_A %.2f r_B %.2f%s | J AC %.3f→%.3f BD %.3f→%.3f AB %.3f→%.3f"
              % (b, R[("A", "train")], R[("A", "new")], R[("B", "train")], R[("B", "new")], R[("C", "train")], R[("D", "train")],
                 P[("C", "train")], P[("C", "new")], P[("D", "train")], P[("D", "new")], P[("A", "train")], P[("A", "new")], P[("B", "train")], P[("B", "new")],
                 rA, rB, (" (형성 전 E189 %.2f·%.2f → 차 %+.2f·%+.2f)" % (u[0], u[1], rA - u[0], rB - u[1])) if u else " (형성 전 없음)",
                 jm("AC", 0), jm("AC", 1), jm("BD", 0), jm("BD", 1), jm("AB", 0), jm("AB", 1)))
    print("조작검증 통과 | 첫 쌍 기준 %d/5 · 둘째 쌍 획득 %d/5 · 유지 %d/5 · 간섭 %d/5 · (보고) 완전 유지 %d/5" % (n["base"], n["prem"], n["ret"], n["intf"], n["full"]))
    if n["base"] < 4:
        print("판정: 보류(기준 미달 — 첫 쌍 기준 %d/5)" % n["base"])
        return 0
    if n["prem"] < 4:
        print("판정: 보류(전제 미달 — 둘째 쌍 획득 %d/5)" % n["prem"])
        return 0
    v = "유지(H111)" if n["ret"] >= 4 else "간섭(H111-null)" if n["intf"] >= 4 else "보류"
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
