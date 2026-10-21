#!/usr/bin/env python3
"""E206 독립 대조 — judge_e206(과 그것이 쓰는 judge_e191·e193·e204·e205)과 코드를 나누지 않는 다른 경로(정규식·정수 산술 직접)로 같은 판정을 다시 계산한다.
평가 원 로그의 RCEVAL mod 를 직접 읽어 칸·효과를 만들고, criteria_fixed.txt 의 규칙(L4 범주, E185·E188·E190 정의, 종합·가설 줄)을 따로 구현한다.
조작검증 중 핵심만 따로 본다: 형성(세계 줄 seed 38300·원형 A, RCJ AC·BD J_post ≤ 0.1000 두 쪽, 제시 816), F 적재 줄(그 뇌 파일·검증 일치)·동작점 줄(= T_F) — 모든 학습·평가 로그,
  게이팅 비(학습 요약)·맥락 고정 켬 수·학습 세계/배정 시드·평가 세계/원형·pushed·평가 맥락/ctx_on·반전/둘째 쌍 요약 줄, 같은 값(F-GW learn = E205 GW learn).
실행: python3 scripts/verify_e206_independent.py (저장소 루트에서)"""
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


def flines(t, b, tf):
    """로그 본문 t — F 적재 줄이 모두 그 뇌 파일·검증 일치이고 ≥ 1, 동작점 줄이 모두 T_F 이고 ≥ 1."""
    fl = re.findall(r"^\[E184 특징 가중치 적재\] (\S+)( 검증 일치)?", t, re.M)
    tn = re.findall(r"\[motor 동작점\] .*?기저값 (-?\d+)\.0\(", t)
    f_ok = bool(fl) and all(g == " 검증 일치" and p.replace("\\", "/").endswith("traces/E206/feat_formed_b%d.npz" % b) for p, g in fl)
    t_ok = bool(tn) and all(int(x) == tf for x in tn)
    return f_ok and t_ok


def evlog(path, b=None, tf=None):
    t = open(path, encoding="utf-8", errors="replace").read()
    d, info = {}, {"pushed": None, "world": None, "proto": None, "ctx": set(), "ctxon": set(), "trials": set(), "f": None}
    for ln in t.splitlines():
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
    if b is not None:
        info["f"] = flines(t, b, tf)
    return d, info


def trlog(path, b, tf):
    t = open(path, encoding="utf-8", errors="replace").read()
    g = re.search(r"^\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*?비\(비활성/활성\) ([0-9.]+)$", t, re.M)
    c = re.search(r"^\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 (X|Y) 켬 (\d+)$", t, re.M)
    w = re.search(r"^\[범주 과제\] catrule — .*?\(세계 시드 (\d+)\).*?배정 지역 난수 (\d+),", t, re.M)
    rv = re.search(r"^\[범주 반전\] 반전 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    p2 = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    return {"gate": (int(g.group(1)), float(g.group(2))) if g else None, "cnt": (int(c.group(1)), c.group(2), int(c.group(3))) if c else None,
            "seed": (int(w.group(1)), int(w.group(2))) if w else None, "rev": rv.groups() if rv else None, "p2": p2.groups() if p2 else None,
            "f": flines(t, b, tf)}


def formlog(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    w = re.search(r"^\[원시 특징 세계\] .*? seed (\d+) \| 원형 A (\d+),(\d+),(\d+),(\d+),(\d+),", t, re.M)
    e = re.search(r"^\[원시 특징 노출\] 제시 (\d+)\(", t, re.M)
    j = {(s, p): (i4("+" + a), i4("+" + z)) for s, p, a, z in re.findall(r"^RCJ side=([lr]) pair=(AC|BD) J_pre=(\d+\.\d{4}) J_post=(\d+\.\d{4})", t, re.M)}
    return {"world": (int(w.group(1)), tuple(int(x) for x in w.group(2, 3, 4, 5, 6))) if w else None, "expo": int(e.group(1)) if e else None, "J": j}


def l4cat(a):
    tr = [a[k][0] for k in SIGN]; nw = [a[k][1] for k in SIGN]
    acq = sum(v >= 1000 for v in tr)
    if acq == 4:
        return ("conc" if all(n_ >= 500 and 2 * n_ >= t_ for t_, n_ in zip(tr, nw)) else "memo"), min(tr)
    return ("part" if acq >= 2 else "fail"), min(tr)


def main():
    L = os.path.join(EXP, "logs", "E206")
    L5 = os.path.join(EXP, "logs", "E205")
    bad = []
    cat, m = {}, {}
    st = {"R": {}, "V": {}, "K": {}}
    try:
        tf = int(re.search(r"T_F=(-\d+)\b", open(os.path.join(L, "calib_tonic.txt"), encoding="utf-8").read()).group(1))
        for b in BR:
            fm = formlog(os.path.join(L, "form_b%d.log" % b))
            if fm["world"] != (38300, (5, 14, 15, 19, 29)) or fm["expo"] != 816 or len(fm["J"]) != 4 or not all(v[1] <= 1000 for v in fm["J"].values()):
                bad.append("형성 b%d %s" % (b, fm))
            tl = trlog(os.path.join(L, "train_GW_b%d.log" % b), b, tf)
            if tl["seed"] != (38300, 38500 + b) or not (tl["gate"] and tl["gate"][0] == 3000 and tl["gate"][1] <= 0.01) or not tl["f"]:
                bad.append("F-GW b%d 학습(시드 %s·게이팅 %s·F/동작점 %s)" % (b, tl["seed"], tl["gate"], tl["f"]))
            a, same = {}, True
            for x in "XY":
                dl, il = evlog(os.path.join(L, "ev_GW_b%d_learn_%s.log" % (b, x)), b, tf)
                dn, inn = evlog(os.path.join(L, "ev_GW_b%d_none_%s.log" % (b, x)), b, tf)
                d5, _ = evlog(os.path.join(L5, "ev_GW_b%d_learn_%s.log" % (b, x)))
                same = same and dl == d5
                for nm, inf, pw in (("learn", il, 8), ("none", inn, 0)):
                    if inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {x} or inf["ctxon"] != {"96"} or inf["trials"] != {(96, 96)} or not inf["f"]:
                        bad.append("F-GW b%d %s %s 평가 %s" % (b, nm, x, inf))
                for c in "AB":
                    a[(x, c)] = tuple(SIGN[(x, c)] * (dl[(c, s)] - dn[(c, s)]) for s in ("train", "new"))
            if same:
                bad.append("F-GW b%d learn = E205 GW learn(같은 값)" % b)
            cat[b], m[b] = l4cat(a)
            for t, nt in (("R", 1500), ("V", 3000), ("P", 3000)):
                tl = trlog(os.path.join(L, "train_%s_b%d.log" % (t, b)), b, tf)
                if not (tl["gate"] and tl["gate"][0] == nt and tl["gate"][1] <= 0.01 and tl["cnt"] == (nt, "X", nt) and tl["seed"] == (38300, 38500 + b) and tl["f"]):
                    bad.append("F-%s b%d 학습(게이팅 %s·켬 %s·시드 %s·F/동작점 %s)" % (t, b, tl["gate"], tl["cnt"], tl["seed"], tl["f"]))
                if (tl["rev"] is not None) != (t == "V") or (t == "V" and tl["rev"] != ("1500", "1500", "3000")):
                    bad.append("F-%s b%d 반전 줄 %s" % (t, b, tl["rev"]))
                if (tl["p2"] is not None) != (t == "P") or (t == "P" and tl["p2"] != ("1500", "1500", "3000")):
                    bad.append("F-%s b%d 둘째 쌍 줄 %s" % (t, b, tl["p2"]))
            D = {}
            for nm, pw in (("R_learn", 8), ("V_learn", 8), ("P_learn", 8), ("none", 0)):
                d, inf = evlog(os.path.join(L, "ev_b%d_%s.log" % (b, nm)), b, tf)
                if inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {"X"} or inf["ctxon"] != {"96"} or inf["trials"] != {(96, 96)} or not inf["f"]:
                    bad.append("F b%d %s 평가 %s" % (b, nm, inf))
                D[nm] = d
            eR = {k: D["R_learn"][k] - D["none"][k] for k in D["R_learn"]}
            eV = {k: D["V_learn"][k] - D["none"][k] for k in D["V_learn"]}
            eP = {k: D["P_learn"][k] - D["none"][k] for k in D["P_learn"]}
            if eR[("A", "train")] <= -1000 and eR[("B", "train")] >= 1000:
                g = eR[("A", "new")] <= -500 and eR[("B", "new")] >= 500 and 2 * eR[("A", "new")] <= eR[("A", "train")] and 2 * eR[("B", "new")] >= eR[("B", "train")]
                st["R"][b] = "개념" if g else "사례"
            elif eR[("A", "train")] <= -1000 or eR[("B", "train")] >= 1000:
                st["R"][b] = "한쪽"
            else:
                st["R"][b] = "실패"
            if eV[("A", "train")] >= 1000 and eV[("B", "train")] <= -1000:
                g = eV[("A", "new")] >= 500 and eV[("B", "new")] <= -500 and 2 * eV[("A", "new")] >= eV[("A", "train")] and 2 * eV[("B", "new")] <= eV[("B", "train")]
                st["V"][b] = "성공" if g else "사례"
            else:
                dA, dB = eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")]
                st["V"][b] = "부분" if (dA >= 1000 and dB <= -1000) else ("고착" if (abs(dA) < 500 and abs(dB) < 500) else "그밖")
            base = eR[("A", "train")] <= -1000 and eR[("B", "train")] >= 1000
            pre = eP[("C", "train")] >= 1000 and eP[("D", "train")] <= -1000
            st["K"][b] = (base, pre, (2 * eP[("A", "train")] <= eR[("A", "train")] and 2 * eP[("B", "train")] >= eR[("B", "train")]) if base and pre else None)
    except (OSError, KeyError, ValueError, IndexError, TypeError, AttributeError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    print("독립 F-GW: 범주 %s · 가장 약한 칸 %s" % (" ".join("b%d %s" % (b, cat[b]) for b in BR), " ".join("%+d" % m[b] for b in BR)))
    print("독립 앞 능력: 범주 규칙 %s | 반전 %s | 유지 %s" % (" ".join(st["R"][b] for b in BR), " ".join(st["V"][b] for b in BR),
                                                    " ".join(("유지" if k[2] else "간섭") if k[2] is not None else ("기준미달" if not k[0] else "전제미달") for k in (st["K"][b] for b in BR))))
    if bad:
        print("독립 조작검증 실패: " + "; ".join(bad))
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    cnt = lambda d, v: sum(x == v for x in d.values())  # noqa: E731
    sl4 = "보존" if cnt(cat, "conc") >= 4 else ("잃음" if cnt(cat, "fail") >= 4 else "부분")
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
    if sl4 == sr == sv == sp == "보존":
        print("독립 가설: H127 지지")
        print("독립 판정: 통합 관문 통과(개발 비교)")
        return 0
    if sp == "잃음":
        h = "H127-null(유지 안 섬)"
    elif sp == "보존":
        h = "F 가 L4 를 깸" if sl4 != "보존" else ("H127 부분(유지는 섬 — 범주 규칙 %s)" % sr if sr != "보존" else "맞교환(유지 ↔ 반전)")
    else:
        h = "H127 부분·보류"
    print("독립 가설: %s" % h)
    print("독립 판정: 통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (sl4, sr, sv, sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
