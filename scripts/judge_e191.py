#!/usr/bin/env python3
"""E191 판정 — 형성 표현 원시 특징 범주 기준선의 독립 확증(뇌 16~20). 규칙은 research/experiments/logs/E191/criteria_fixed.txt.
입력(logs/E191/): 형성 form_b{B}.log(노출 범주 A·B·C·D), 학습 train_R_b{B}.log(catrule 1,500)·train_P_b{B}.log(1,500 → 둘째 쌍 1,500)·train_V_b{B}.log(1,500 → 반전 1,500),
평가 ev_b{B}_{R,P,V}_learn.log·ev_b{B}_none.log(RCEVAL 8줄 — A·B·C·D × train·new, 모두 형성 가중치 적재).
지표(1e-4): e_X = m(X 학습) − m(무학습). 정답(첫 쌍) A 교차(−)·B 같은 쪽(+), 둘째 쌍 C 같은 쪽(+)·D 교차(−), 반전 뒤 A 같은 쪽(+)·B 교차(−).
(가) 유지(E190 규칙): 기준 = R_A,train ≤ −1000·R_B,train ≥ +1000; 전제 = P_C,train ≥ +1000·P_D,train ≤ −1000; 유지 = 기준·전제 이면서 2·P_A ≤ R_A·2·P_B ≥ R_B; 간섭 = 기준·전제 이면서 유지 아님.
   판정: 기준 < 4/5 → 보류(기준 미달) / 전제 < 4/5 → 보류(전제 미달) / 유지 ≥ 4/5 → 확증 / 간섭 ≥ 4/5 → 불확증 / 그 밖 보류.
(나) 개념 수준 일반화(E184 규칙, R): 획득 = R_A,train ≤ −1000·R_B,train ≥ +1000; 일반화 = R_A,new ≤ −500·R_B,new ≥ +500·2·R_A,new ≤ R_A,train·2·R_B,new ≥ R_B,train.
   판정: 획득·일반화 ≥ 4/5 → 보존 / 획득·일반화 아님 ≥ 4/5 → 약화(사례 한정) / 그 밖 보류.
(다) 반전(E188 규칙, V 대 R): 반전 획득 = V_A,train ≥ +1000·V_B,train ≤ −1000; 반전 일반화 = V_A,new ≥ +500·V_B,new ≤ −500·2·V_A,new ≥ V_A,train·2·V_B,new ≤ V_B,train;
   부분 = 반전 획득 아님·Δ_A = V_A − R_A ≥ +1000·Δ_B ≤ −1000; 고착 = |Δ_A| < 500·|Δ_B| < 500.
   판정: 반전 성공 ≥ 4/5 → 보존 / 사례 한정 ≥ 4/5 / 부분 ≥ 4/5 / 고착 ≥ 4/5 / 그 밖 보류.
종합: 조작검증 실패 → 보류 / (가) 확증·(나) 보존·(다) 보존 → 기준선 확증(H112) / (가) 확증·(나)·(다) 중 하나라도 보존 아님 → 부분(유지만 확증 — 부판정이 약화·고착 등이면 맞교환) / (가) 불확증 → 불확증(H112-null) / 그 밖 보류.
실행: python3 scripts/judge_e191.py (저장소 루트에서)"""
import os
import sys

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
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
    r = {"p2start": None, "n2": None, "p2lines": 0, "rvstart": None, "nrev": None, "rvlines": 0,
         "cat": None, "chk": [], "done": False, "kt": 0, "fw": 0}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("[둘째 범주 쌍]"):
                r["p2lines"] += 1
                if ln.startswith("[둘째 범주 쌍] 시행 "):
                    r["p2start"] = int(ln.split("시행 ")[1].split()[0])
                elif ln.startswith("[둘째 범주 쌍] 둘째 쌍 시행 "):
                    r["n2"] = (int(ln.split("둘째 쌍 시행 ")[1].split()[0]), int(ln.split("(시행 ")[1].split()[0]), int(ln.split("전체 ")[1].split(")")[0]))
            elif ln.startswith("[범주 반전]"):
                r["rvlines"] += 1
                if ln.startswith("[범주 반전] 시행 "):
                    r["rvstart"] = int(ln.split("시행 ")[1].split()[0])
                elif ln.startswith("[범주 반전] 반전 시행 "):
                    r["nrev"] = (int(ln.split("반전 시행 ")[1].split()[0]), int(ln.split("(시행 ")[1].split()[0]), int(ln.split("전체 ")[1].split(")")[0]))
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


def train_ok(t, kind):
    if not (t["done"] and t["kt"] >= 1 and t["fw"] >= 1 and len(t["chk"]) == 3 and all(x > 0 and y == 0 for x, y in t["chk"]) and t["cat"]):
        return False
    if kind == "P":
        return (t["p2start"] == 1500 and t["n2"] == (1500, 1500, 3000) and t["rvlines"] == 0
                and t["cat"][0] == 3000 and 1200 <= t["cat"][1] <= 1800)
    if kind == "V":
        return (t["rvstart"] == 1500 and t["nrev"] == (1500, 1500, 3000) and t["p2lines"] == 0
                and t["cat"][0] == 3000 and 1200 <= t["cat"][1] <= 1800)
    return t["p2lines"] == 0 and t["rvlines"] == 0 and t["cat"][0] == 1500 and 600 <= t["cat"][1] <= 900


def form_ok(fm):
    return (fm["expo"] == 816 and fm["expo_cats"] == "A,B,C,D" and len(fm["rcw"]) == 2 and all(d > 0 and g == 1 for d, g in fm["rcw"])
            and set(fm["rcj"]) == {(s, p) for s in "lr" for p in PAIRS} and fm["rawcat"] == 1
            and sorted(c[0] for c in fm["chk"]) == ["post", "pre", "pre2"] and all(c[1] == 288 and c[2] == 0 and c[3] == 0 and c[4] == 0 and c[5] == 1 for c in fm["chk"]))


def eval_ok(ev, pushed):
    return set(ev["rc"]) == CATS8 and not ev["bad"] and not ev["sat"] and ev["kt"] >= 1 and ev["fw"] >= 1 and ev["pushed"] == pushed


def main():
    ok, why, rows = True, [], {}
    for b in BRAINS:
        try:
            lg = lambda n: os.path.join(EXP, "logs", "E191", n % b)
            fm = read_form(lg("form_b%d.log"))
            tr = {k: read_train(lg("train_%s_b%%d.log" % k)) for k in "RPV"}
            ev = {k: read_eval(lg("ev_b%%d_%s_learn.log" % k)) for k in "RPV"}
            eN = read_eval(lg("ev_b%d_none.log"))
        except (OSError, ValueError, IndexError, KeyError) as ex:
            print("판정: 보류(결측 — b%d %s)" % (b, type(ex).__name__))
            return 0
        checks = [("형성", form_ok(fm))] + [("%s 학습" % k, train_ok(tr[k], k)) for k in "RPV"] \
            + [("%s 평가" % k, eval_ok(ev[k], 8)) for k in "RPV"] + [("무학습 평가", eval_ok(eN, 0))]
        for nm, c in checks:
            if not c:
                ok = False; why.append("b%d %s" % (b, nm))
        if ok:
            rows[b] = ({k: {kk: ev[k]["rc"][kk] - eN["rc"][kk] for kk in CATS8} for k in "RPV"}, fm["rcj"])
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    n = {k: 0 for k in ("base", "prem", "ret", "intf", "acq", "gen", "memoG", "rsucc", "rmemo", "rpart", "rstuck")}
    for b in BRAINS:
        E, J = rows[b]
        R, P, V = E["R"], E["P"], E["V"]
        rA, rB, rAn, rBn = R[("A", "train")], R[("B", "train")], R[("A", "new")], R[("B", "new")]
        base = rA <= -1000 and rB >= 1000
        prem = P[("C", "train")] >= 1000 and P[("D", "train")] <= -1000
        kept = 2 * P[("A", "train")] <= rA and 2 * P[("B", "train")] >= rB
        gen = rAn <= -500 and rBn >= 500 and 2 * rAn <= rA and 2 * rBn >= rB
        vA, vB, vAn, vBn = V[("A", "train")], V[("B", "train")], V[("A", "new")], V[("B", "new")]
        vacq = vA >= 1000 and vB <= -1000
        vgen = vAn >= 500 and vBn <= -500 and 2 * vAn >= vA and 2 * vBn <= vB
        dA, dB = vA - rA, vB - rB
        n["base"] += base; n["prem"] += prem
        n["ret"] += base and prem and kept; n["intf"] += base and prem and not kept
        n["acq"] += base; n["gen"] += base and gen; n["memoG"] += base and not gen
        n["rsucc"] += vacq and vgen; n["rmemo"] += vacq and not vgen
        n["rpart"] += (not vacq) and dA >= 1000 and dB <= -1000; n["rstuck"] += abs(dA) < 500 and abs(dB) < 500
        jm = lambda p, i: (J[("l", p)][i] + J[("r", p)][i]) / 2e4
        print("b%d R A %+d/%+d B %+d/%+d (비 %.2f·%.2f) | P C %+d D %+d 뒤 A %+d B %+d (r %.2f·%.2f) | V A %+d/%+d B %+d/%+d (Δ %+d·%+d) | J AC %.3f→%.3f BD %.3f→%.3f"
              % (b, rA, rAn, rB, rBn, rAn / float(rA) if rA else float("nan"), rBn / float(rB) if rB else float("nan"),
                 P[("C", "train")], P[("D", "train")], P[("A", "train")], P[("B", "train")],
                 P[("A", "train")] / float(rA) if rA else float("nan"), P[("B", "train")] / float(rB) if rB else float("nan"),
                 vA, vAn, vB, vBn, dA, dB, jm("AC", 0), jm("AC", 1), jm("BD", 0), jm("BD", 1)))
    ga = ("보류(기준 미달)" if n["base"] < 4 else "보류(전제 미달)" if n["prem"] < 4 else "확증" if n["ret"] >= 4 else "불확증" if n["intf"] >= 4 else "보류")
    gb = "보존" if n["gen"] >= 4 else "약화(사례 한정)" if n["memoG"] >= 4 else "보류"
    gc = ("보존" if n["rsucc"] >= 4 else "사례 한정" if n["rmemo"] >= 4 else "부분" if n["rpart"] >= 4 else "고착" if n["rstuck"] >= 4 else "보류")
    print("조작검증 통과 | (가) 기준 %d·전제 %d·유지 %d·간섭 %d → %s | (나) 획득 %d·개념 %d·사례 한정 %d → %s | (다) 성공 %d·사례 한정 %d·부분 %d·고착 %d → %s"
          % (n["base"], n["prem"], n["ret"], n["intf"], ga, n["acq"], n["gen"], n["memoG"], gb, n["rsucc"], n["rmemo"], n["rpart"], n["rstuck"], gc))
    if ga == "확증":
        v = "기준선 확증(H112)" if gb == "보존" and gc == "보존" else "부분(유지만 확증 — 일반화 %s·반전 %s)" % (gb, gc)
    elif ga == "불확증":
        v = "불확증(H112-null)"
    else:
        v = "보류"
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
