#!/usr/bin/env python3
"""E207 판정 — 학습 조건의 범주 KC 겹침(측정, 학습 없음): 형성 표현 F(·U 보고)의 범주 KC 집합을 측정 제시 동안 맥락 없음 대 맥락 X(C2·게이팅 구조 같음)로.
규칙 = research/experiments/logs/E207/criteria_fixed.txt.
입력: logs/E207/meas_{U|F}_{none|X}_b{B}.log(뇌 16~20). 지표 S = J(A,C)_좌 + J(B,D)_좌 + J(A,C)_우 + J(B,D)_우(RCJ, 정수 1e-4).
뇌마다(F): 되살림 = S_X ≥ 6000 이고 S_X − S_none ≥ 4000, 되살리지 않음 = S_X ≤ 3200, 그 밖 가운데. 종합: 조작검증 실패 → 보류 / 되살림 ≥ 4/5 → H128 지지 / 되살리지 않음 ≥ 4/5 → H128-null / 그 밖 보류.
정정 1(2026-10-22, 경로 검사 뒤·본실험 전 — criteria_fixed.txt): 노출 0 에서도 한 프로세스의 두 측정(J_pre = 새 뇌 첫 측정, J_post = 둘째 측정)이 이력 때문에 다르다(뇌 15).
  → 분류를 J_pre 로 한 번·J_post 로 한 번 하고 둘이 같을 때만 그 분류, 다르면 가운데. 조작검증 'J_pre = J_post' 를 빼고 (보고) 측정 이력 차 |J_post − J_pre| 최대. 같은 값 점검은 J_pre·J_post 각각.
조작검증: RAWCAT 1·RCJ 12·RCCHK pre·post(npres 288·비활성 0·반대쪽 0·goodbad 0·g_same 1)·노출 0·세계 38300·원형 A·F 적재(F ≥ 1 그 뇌 파일, U 0)·동작점(U −2800·F −1300)·
  맥락 X 측정 '[E207 측정 맥락]' pre·post(켠 > 0·끈 0·게이팅 있음·100·비활성 ≤ 활성), 맥락 없음 측정에 그 줄 없음·같은 값(X 와 none RCJ 12값 모두 같음 → 무효 의심).
실행: python3 scripts/judge_e207.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
REPS = ("U", "F")
CONDS = ("none", "X")
TONIC = {"U": -2800.0, "F": -1300.0}
WORLD, PROTO_A = 38300, "5,14,15,19,29"
PAIRS = ("AB", "AC", "AD", "BC", "BD", "CD")
NPRES = 288
TH_HI, TH_DIFF, TH_LO = 6000, 4000, 3200


def q4(s):
    s = s.strip(); neg = s.startswith("-")
    a, b = s.lstrip("+-").split(".")
    v = int(a) * 10000 + int((b + "0000")[:4])
    return -v if neg else v


def read_meas(path):
    r = {"rcj": {}, "chk": {}, "ctx": {}, "expo": None, "world": [], "feat": [], "tonic": [], "rawcat": 0, "rckc": {}}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("RCJ "):
            kv = dict(x.split("=", 1) for x in ln.split()[1:])
            r["rcj"][(kv["side"], kv["pair"])] = (q4(kv["J_pre"]), q4(kv["J_post"]), kv["n_post"], kv["n_pre"])
        elif ln.startswith("RCCHK "):
            kv = dict(x.split("=", 1) for x in ln.split()[1:])
            r["chk"][kv["tag"]] = (int(kv["npres"]), int(kv["feat_inactive_spikes"]), int(kv["feat_other_side"]), int(kv["goodbad"]), int(kv["g_same"]))
        elif ln.startswith("[E207 측정 맥락] "):
            m = re.match(r"\[E207 측정 맥락\] tag=(\w+) 맥락 (\w) 켬 — 제시 (\d+) · 켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · KC 발화 활성 무리 (\d+) · 비활성 무리 (\d+) · 게이팅 (\S+)", ln)
            if m:
                r["ctx"][m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7)), m.group(8))
            else:
                r["ctx"]["?"] = None
        elif ln.startswith("[원시 특징 노출] 제시 "):
            r["expo"] = int(ln.split("제시 ")[1].split("(")[0])
        elif ln.startswith("[원시 특징 세계] "):
            m = re.search(r"seed (\d+) \| 원형 A ([0-9]+(?:,[0-9]+){4})", ln)
            r["world"].append((int(m.group(1)), m.group(2)) if m else (-1, ""))
        elif ln.startswith("[E184 특징 가중치 적재]"):
            m = re.match(r"\[E184 특징 가중치 적재\] (\S+) 검증 일치", ln)
            r["feat"].append(m.group(1) if m else None)
        elif "[motor 동작점]" in ln:
            m = re.search(r"기저값 (-?[0-9]+(?:\.[0-9]+)?)\(", ln)
            r["tonic"].append(float(m.group(1)) if m else float("nan"))
        elif ln.startswith("=> RAWCAT "):
            r["rawcat"] += 1
        elif ln.startswith("RCKC tag=post "):
            kv = dict(x.split("=", 1) for x in ln.split()[1:])
            if kv["set"] == "train":
                ids = set(int(x) for x in kv["ids"].split(",") if x) if kv.get("ids") else set()
                r["rckc"].setdefault(kv["side"], {}).setdefault(kv["cat"], []).append(ids)
    return r


def score(r, k=1):
    """k = 0 첫 측정(J_pre), 1 둘째 측정(J_post)."""
    return sum(r["rcj"][(s, p)][k] for s in "lr" for p in ("AC", "BD"))


def hist_diff(r):
    """(보고) 측정 이력 차 — 12쌍 |J_post − J_pre| 최대(1e-4)."""
    return max(abs(v[1] - v[0]) for v in r["rcj"].values())


def check(tag, r, rep, cond, b, why):
    ok = True
    if r["rawcat"] != 1 or set(r["rcj"]) != {(s, p) for s in "lr" for p in PAIRS}:
        ok = False; why.append("%s RAWCAT·RCJ(RAWCAT %d·줄 %d)" % (tag, r["rawcat"], len(r["rcj"])))
    if set(r["chk"]) != {"pre", "post"} or any(v != (NPRES, 0, 0, 0, 1) for v in r["chk"].values()):
        ok = False; why.append("%s RCCHK %s" % (tag, r["chk"]))
    if r["expo"] != 0:
        ok = False; why.append("%s 노출 %s" % (tag, r["expo"]))
    if r["world"] != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 세계 %s" % (tag, r["world"]))
    want = "traces/E206/feat_formed_b%d.npz" % b
    if rep == "F" and (not r["feat"] or not all(p is not None and p.replace("\\", "/").endswith(want) for p in r["feat"])):
        ok = False; why.append("%s F 적재 %s" % (tag, r["feat"]))
    if rep == "U" and r["feat"]:
        ok = False; why.append("%s U 에 F 적재 %s" % (tag, r["feat"]))
    if not r["tonic"] or not all(t == TONIC[rep] for t in r["tonic"]):
        ok = False; why.append("%s 동작점 %s" % (tag, r["tonic"]))
    if cond == "X":
        c = r["ctx"]
        good = set(c) == {"pre", "post"} and all(v is not None and v[0] == "X" and v[1] == NPRES and v[2] > 0 and v[3] == 0 and v[6] == "있음" and 100 * v[5] <= v[4] and v[4] > 0 for v in c.values())
        if not good:
            ok = False; why.append("%s 측정 맥락 %s" % (tag, c))
    elif r["ctx"]:
        ok = False; why.append("%s 맥락 없음 측정에 맥락 줄 %s" % (tag, r["ctx"]))
    return ok


def classify(sx, sn):
    if sx >= TH_HI and sx - sn >= TH_DIFF:
        return "되살림"
    if sx <= TH_LO:
        return "되살리지 않음"
    return "가운데"


def classify2(c0, c1):
    """정정 1: 첫 측정·둘째 측정 분류가 같을 때만 그 분류, 다르면 가운데."""
    return c0 if c0 == c1 else "가운데"


def cat_sets(rckc):
    out = {}
    for sd, cats in rckc.items():
        out[sd] = {}
        for c, lst in cats.items():
            need = max(1, -(-len(lst) // 2))
            cnt = {}
            for st in lst:
                for x in st:
                    cnt[x] = cnt.get(x, 0) + 1
            out[sd][c] = {x for x, n in cnt.items() if n >= need}
    return out


def posthoc(b, rckc):
    """(보고) 맥락 X F 집합으로 본 사후 기술 — R 맞춤 KC(E206 w_R, |d_R| ≥ 50) 중 A∪B 안 몫, P 바뀐 KC 중 C∪D 안 몫. 자료가 없으면 None."""
    try:
        import numpy as np
        import posthoc_kc_rewrite as PK
        PK.EXP = EXP
        S = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
        R = np.load(os.path.join(EXP, "traces", "E206", "w_R_b%d.npz" % b)); P = np.load(os.path.join(EXP, "traces", "E206", "w_P_b%d.npz" % b))
    except (OSError, ImportError, KeyError):
        return None
    sets = cat_sets(rckc)
    acc = {}
    for sd in "lr":
        dR, bR = PK.kc_pref(R, S, sd); dP, bP = PK.kc_pref(P, S, sd)
        st = PK.set_stats(dR, dP, bR & bP, sets.get(sd, {}), 50.0)
        for k, v in st.items():
            acc[k] = acc.get(k, 0) + v
    return (acc["tuned_in_ab"] / acc["tuned"] if acc["tuned"] else float("nan"), acc["changed_in_cd"] / acc["changed"] if acc["changed"] else float("nan"))


def main():
    ok, why, M = True, [], {}
    L = os.path.join(EXP, "logs", "E207")
    try:
        for b in BRAINS:
            for rep in REPS:
                for cond in CONDS:
                    r = read_meas(os.path.join(L, "meas_%s_%s_b%d.log" % (rep, cond, b)))
                    ok = check("%s %s b%d" % (rep, cond, b), r, rep, cond, b, why) and ok
                    M[(b, rep, cond)] = r
    except (OSError, ValueError, KeyError, IndexError) as ex:
        print("판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    for b in BRAINS:
        for rep in REPS:
            for k, nm in ((0, "J_pre"), (1, "J_post")):
                jx = [M[(b, rep, "X")]["rcj"].get((s, p), (0, 0))[k] for s in "lr" for p in PAIRS]
                jn = [M[(b, rep, "none")]["rcj"].get((s, p), (0, 0))[k] for s in "lr" for p in PAIRS]
                if jx == jn:
                    ok = False; why.append("같은 값 %s b%d(X = none RCJ %s 12값) — 조작 무효 의심" % (rep, b, nm))
    if not ok:
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cls = {}
    for b in BRAINS:
        S = {(rep, c, k): score(M[(b, rep, c)], k) for rep in REPS for c in CONDS for k in (0, 1)}
        c0 = classify(S[("F", "X", 0)], S[("F", "none", 0)]); c1 = classify(S[("F", "X", 1)], S[("F", "none", 1)])
        cls[b] = classify2(c0, c1)
        ab = {(rep, c): (M[(b, rep, c)]["rcj"][("l", "AB")][0], M[(b, rep, c)]["rcj"][("r", "AB")][0], M[(b, rep, c)]["rcj"][("l", "AB")][1], M[(b, rep, c)]["rcj"][("r", "AB")][1])
              for rep in REPS for c in CONDS}
        n = {(rep, c): M[(b, rep, c)]["rcj"][("l", "AC")][2] for rep in REPS for c in CONDS}
        hd = {(rep, c): hist_diff(M[(b, rep, c)]) for rep in REPS for c in CONDS}
        ph = posthoc(b, M[(b, "F", "X")]["rckc"])
        ph0 = posthoc(b, M[(b, "F", "none")]["rckc"])
        print("b%d F: 첫 측정 S_none %d · S_X %d · 차 %+d → %s · 둘째 측정 S_none %d · S_X %d · 차 %+d → %s ⇒ %s | U(첫/둘째): S_none %d/%d · S_X %d/%d"
              " | J(A,B) 좌/우 첫·둘째 F none %d/%d·%d/%d · X %d/%d·%d/%d · U none %d/%d·%d/%d · X %d/%d·%d/%d"
              " | (보고) 측정 이력 차 최대 F none %d · X %d · U none %d · X %d | 집합 크기 AC 좌 n_post F none %s · X %s · U none %s · X %s"
              % ((b, S[("F", "none", 0)], S[("F", "X", 0)], S[("F", "X", 0)] - S[("F", "none", 0)], c0,
                  S[("F", "none", 1)], S[("F", "X", 1)], S[("F", "X", 1)] - S[("F", "none", 1)], c1, cls[b],
                  S[("U", "none", 0)], S[("U", "none", 1)], S[("U", "X", 0)], S[("U", "X", 1)])
                 + ab[("F", "none")] + ab[("F", "X")] + ab[("U", "none")] + ab[("U", "X")]
                 + (hd[("F", "none")], hd[("F", "X")], hd[("U", "none")], hd[("U", "X")], n[("F", "none")], n[("F", "X")], n[("U", "none")], n[("U", "X")]))
              + (" | (보고) R 맞춤 KC 중 A∪B 안 몫 F·none %.3f → X %.3f · P 바뀐 KC 중 C∪D 안 몫 %.3f → %.3f(둘째 측정 집합)" % (ph0[0], ph[0], ph0[1], ph[1]) if ph and ph0 else " | (보고) 사후 기술 자료 없음"))
    nr = sum(v == "되살림" for v in cls.values()); nn = sum(v == "되살리지 않음" for v in cls.values()); nm = sum(v == "가운데" for v in cls.values())
    print("F: 되살림 %d/5 · 되살리지 않음 %d/5 · 가운데 %d/5" % (nr, nn, nm))
    print("조작검증 통과")
    if nr >= 4:
        print("판정: H128 지지 — 맥락 X 가 형성 표현의 범주 KC 겹침을 되살림")
    elif nn >= 4:
        print("판정: H128-null — 맥락 X 에서도 형성 표현의 범주 KC 겹침 낮음")
    else:
        print("판정: 보류 — 되살림 %d/5·되살리지 않음 %d/5" % (nr, nn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
