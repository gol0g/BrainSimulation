#!/usr/bin/env python3
"""E205 독립 대조 — judge_e205 와 코드를 나누지 않는 다른 경로(정규식·정수 산술 직접)로 같은 판정을 다시 계산한다.
평가 원 로그의 RCEVAL mod 를 직접 읽어 칸·효과를 만들고, criteria_fixed.txt 의 규칙(L4 범주·m 차, E185·E188·E190 정의, 종합)을 따로 구현한다.
조작검증 중 핵심만 따로 본다: 게이팅 비(학습 요약)·맥락 고정 켬 수·학습 세계/배정 시드·평가 세계/원형·pushed·평가 맥락/ctx_on·반전/둘째 쌍 요약 줄.
실행: python3 scripts/verify_e205_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BR = [16, 17, 18, 19, 20]
RX_EVAL = re.compile(r"^RCEVAL cat=([A-D]) set=(train|new) mod=([+-]\d+\.\d{4}) .*? trials=(\d+) feat_on=(\d+)(?: ctx=(\S+) ctx_on=(\d+))?")
SIGN = {("X", "A"): -1, ("X", "B"): 1, ("Y", "A"): 1, ("Y", "B"): -1}


def i4(s):
    """'+0.1234' → 1234 (정수 1e-4) — 문자열에서 직접."""
    neg = s[0] == "-"
    w, f = s.lstrip("+-").split(".")
    v = int(w) * 10000 + int(f)
    return -v if neg else v


def evlog(path):
    d, info = {}, {"pushed": None, "world": None, "proto": None, "ctx": set(), "ctxon": set(), "trials": set()}
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            m = RX_EVAL.match(ln)
            if m:
                d[(m.group(1), m.group(2))] = i4(m.group(3))
                info["trials"].add((int(m.group(4)), int(m.group(5))))
                info["ctx"].add(m.group(6)); info["ctxon"].add(m.group(7))
                continue
            m = re.search(r"pushed=(\d+)", ln) if ln.startswith("=> DECOMP") else None
            if m:
                info["pushed"] = int(m.group(1))
            m = re.match(r"\[E204 평가 세계\] 세계 시드 (\d+) · 원형 A (\d+),(\d+),(\d+),(\d+),(\d+)", ln)
            if m:
                info["world"] = int(m.group(1)); info["proto"] = tuple(int(x) for x in m.group(2, 3, 4, 5, 6))
    return d, info


def trlog(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    g = re.search(r"^\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*?비\(비활성/활성\) ([0-9.]+)$", t, re.M)
    c = re.search(r"^\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 (X|Y) 켬 (\d+)$", t, re.M)
    w = re.search(r"^\[범주 과제\] catrule — .*?\(세계 시드 (\d+)\).*?배정 지역 난수 (\d+),", t, re.M)
    rv = re.search(r"^\[범주 반전\] 반전 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    p2 = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    return {"gate": (int(g.group(1)), float(g.group(2))) if g else None, "cnt": (int(c.group(1)), c.group(2), int(c.group(3))) if c else None,
            "seed": (int(w.group(1)), int(w.group(2))) if w else None, "rev": rv.groups() if rv else None, "p2": p2.groups() if p2 else None,
            "gate_any": "[E204 맥락 게이팅" in t}


def l4cat(a):
    tr = [a[k][0] for k in SIGN]; nw = [a[k][1] for k in SIGN]
    acq = sum(v >= 1000 for v in tr)
    if acq == 4:
        return ("conc" if all(n_ >= 500 and 2 * n_ >= t_ for t_, n_ in zip(tr, nw)) else "memo"), min(tr)
    return ("part" if acq >= 2 else "fail"), min(tr)


def main():
    L = os.path.join(EXP, "logs", "E205")
    bad = []
    cat, m = {"GW": {}, "UW": {}}, {"GW": {}, "UW": {}}
    st = {"R": {}, "V": {}, "K": {}}
    try:
        for b in BR:
            for arm in ("GW", "UW"):
                tl = trlog(os.path.join(L, "train_%s_b%d.log" % (arm, b)))
                if tl["seed"] != (38300, 38500 + b):
                    bad.append("%s b%d 시드 %s" % (arm, b, tl["seed"]))
                if arm == "GW" and not (tl["gate"] and tl["gate"][0] == 3000 and tl["gate"][1] <= 0.01):
                    bad.append("GW b%d 게이팅 %s" % (b, tl["gate"]))
                if arm == "UW" and tl["gate_any"]:
                    bad.append("UW b%d 게이팅 줄" % b)
                a = {}
                for x in "XY":
                    dl, il = evlog(os.path.join(L, "ev_%s_b%d_learn_%s.log" % (arm, b, x)))
                    dn, inn = evlog(os.path.join(L, "ev_%s_b%d_none_%s.log" % (arm, b, x)))
                    for nm, inf, pw in (("learn", il, 8), ("none", inn, 0)):
                        if inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {x} or inf["ctxon"] != {"96"} or inf["trials"] != {(96, 96)}:
                            bad.append("%s b%d %s %s 평가 %s" % (arm, b, nm, x, inf))
                    for c in "AB":
                        a[(x, c)] = tuple(SIGN[(x, c)] * (dl[(c, s)] - dn[(c, s)]) for s in ("train", "new"))
                cat[arm][b], m[arm][b] = l4cat(a)
            n_ = {}
            for t, nt in (("R", 1500), ("V", 3000), ("P", 3000)):
                tl = trlog(os.path.join(L, "train_%s_b%d.log" % (t, b)))
                if not (tl["gate"] and tl["gate"][0] == nt and tl["gate"][1] <= 0.01 and tl["cnt"] == (nt, "X", nt) and tl["seed"] == (38300, 38500 + b)):
                    bad.append("%s b%d 학습(게이팅 %s·켬 %s·시드 %s)" % (t, b, tl["gate"], tl["cnt"], tl["seed"]))
                if (tl["rev"] is not None) != (t == "V") or (t == "V" and tl["rev"] != ("1500", "1500", "3000")):
                    bad.append("%s b%d 반전 줄 %s" % (t, b, tl["rev"]))
                if (tl["p2"] is not None) != (t == "P") or (t == "P" and tl["p2"] != ("1500", "1500", "3000")):
                    bad.append("%s b%d 둘째 쌍 줄 %s" % (t, b, tl["p2"]))
            D = {}
            for nm, pw in (("R_learn", 8), ("V_learn", 8), ("P_learn", 8), ("none", 0)):
                d, inf = evlog(os.path.join(L, "ev_b%d_%s.log" % (b, nm)))
                if inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {"X"} or inf["ctxon"] != {"96"} or inf["trials"] != {(96, 96)}:
                    bad.append("b%d %s 평가 %s" % (b, nm, inf))
                D[nm] = d
            eR = {k: D["R_learn"][k] - D["none"][k] for k in D["R_learn"]}
            eV = {k: D["V_learn"][k] - D["none"][k] for k in D["V_learn"]}
            eP = {k: D["P_learn"][k] - D["none"][k] for k in D["P_learn"]}
            # 범주 규칙(E185)
            if eR[("A", "train")] <= -1000 and eR[("B", "train")] >= 1000:
                g = eR[("A", "new")] <= -500 and eR[("B", "new")] >= 500 and 2 * eR[("A", "new")] <= eR[("A", "train")] and 2 * eR[("B", "new")] >= eR[("B", "train")]
                st["R"][b] = "개념" if g else "사례"
            elif eR[("A", "train")] <= -1000 or eR[("B", "train")] >= 1000:
                st["R"][b] = "한쪽"
            else:
                st["R"][b] = "실패"
            # 반전(E188)
            if eV[("A", "train")] >= 1000 and eV[("B", "train")] <= -1000:
                g = eV[("A", "new")] >= 500 and eV[("B", "new")] <= -500 and 2 * eV[("A", "new")] >= eV[("A", "train")] and 2 * eV[("B", "new")] <= eV[("B", "train")]
                st["V"][b] = "성공" if g else "사례"
            else:
                dA, dB = eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")]
                st["V"][b] = "부분" if (dA >= 1000 and dB <= -1000) else ("고착" if (abs(dA) < 500 and abs(dB) < 500) else "그밖")
            # 유지(E190)
            base = eR[("A", "train")] <= -1000 and eR[("B", "train")] >= 1000
            pre = eP[("C", "train")] >= 1000 and eP[("D", "train")] <= -1000
            st["K"][b] = (base, pre, (2 * eP[("A", "train")] <= eR[("A", "train")] and 2 * eP[("B", "train")] >= eR[("B", "train")]) if base and pre else None)
    except (OSError, KeyError, ValueError, IndexError, TypeError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    for arm in ("GW", "UW"):
        print("독립 %s: 범주 %s · 가장 약한 칸 %s" % (arm, " ".join("b%d %s" % (b, cat[arm][b]) for b in BR), " ".join("%+d" % m[arm][b] for b in BR)))
    print("독립 앞 능력: 범주 규칙 %s | 반전 %s | 유지 %s" % (" ".join(st["R"][b] for b in BR), " ".join(st["V"][b] for b in BR),
                                                    " ".join(("유지" if k[2] else "간섭") if k[2] is not None else ("기준미달" if not k[0] else "전제미달") for k in (st["K"][b] for b in BR))))
    if bad:
        print("독립 조작검증 실패: " + "; ".join(bad))
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    ngw = sum(cat["GW"][b] == "conc" for b in BR)
    ngate = sum(m["GW"][b] - m["UW"][b] >= 500 for b in BR)
    l4 = ("확증" if ngate >= 4 else "체계 확증(게이팅 효과 미확증)") if ngw >= 4 else "미확증"
    cnt = lambda d, v: sum(x == v for x in d.values())  # noqa: E731
    sr = "보존" if cnt(st["R"], "개념") >= 4 else ("잃음" if cnt(st["R"], "실패") >= 4 else "부분")
    sv = "보존" if cnt(st["V"], "성공") >= 4 else ("잃음" if cnt(st["V"], "고착") >= 4 else "부분")
    K = st["K"]
    if sum(k[0] for k in K.values()) < 4:
        sp = "보류(기준 미달)"
    elif sum(k[1] for k in K.values()) < 4:
        sp = "보류(전제 미달)"
    else:
        nk = sum(k[2] is True for k in K.values()); ni = sum(k[2] is False for k in K.values())
        sp = "보존" if nk >= 4 else ("잃음" if ni >= 4 else "부분")
    print("독립 L4: GW 범주 %d/5 · 게이팅 효과 %d/5 → %s" % (ngw, ngate, l4))
    if l4 == "확증" and sr == sv == sp == "보존":
        print("독립 판정: 통합 관문 통과")
    else:
        print("독립 판정: 통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (l4, sr, sv, sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
