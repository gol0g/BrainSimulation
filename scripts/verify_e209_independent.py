#!/usr/bin/env python3
"""E209 독립 대조 — judge_e209(과 그것이 쓰는 judge_e193·e204·e205·e206·e208·report_e209_overlap)와 코드를 나누지 않는 다른 경로(정규식·정수 산술 직접)로 같은 판정을 다시 계산한다.
verify_e208_independent 와 같은 꼴 + 구획 되먹임 억제 점검을 따로 구현: 그 뇌의 초기 연결 스냅숏(traces/E176/snap/conn_b{B}.npz)에서 게이팅 무리(난수 19300 + genn_seed —
쪽 l → r 마다 KC 배정·X 마스크·Y 마스크 순서)와 억제 뉴런 무리(난수 19900 + genn_seed 순열 앞 200, l → r)를 모델 코드 없이 다시 만들어 같은 무리 안 연결 수를 세고,
모든 학습·평가 로그의 장치 확인 줄 연결 수가 그 값과 같은지(g 10.00·−30.00, 구성 줄 수 = 장치 줄 수, 학습 ≥ 2·평가 ≥ 1). 짝·같은 값은 E208 F-G0, 가설 줄은 H130 갈래.
실행: python3 scripts/verify_e209_independent.py (저장소 루트에서)"""
import os
import re
import sys

import numpy as np

EXP = "research/experiments"
BR = [16, 17, 18, 19, 20]
NKC, NIH, NCX, NCY, GFRAC, GP = 3000, 400, 200, 200, 0.5, 0.10
RX_EVAL = re.compile(r"^RCEVAL cat=([A-D]) set=(train|new) mod=([+-]\d+\.\d{4}) .*? trials=(\d+) feat_on=(\d+)(?: ctx=(\S+) ctx_on=(\d+))?")
SIGN = {("X", "A"): -1, ("X", "B"): 1, ("Y", "A"): 1, ("Y", "B"): -1}
NAMES = ("kc_l_to_inh_l", "kc_inh_l_to_kc_l", "kc_r_to_inh_r", "kc_inh_r_to_kc_r")


def i4(s):
    """'+0.1234' → 1234 (정수 1e-4) — 문자열에서 직접."""
    neg = s[0] == "-"
    w, f = s.lstrip("+-").split(".")
    v = int(w) * 10000 + int(f)
    return -v if neg else v


def expected_compart(b):
    """스냅숏 + 다시 만든 무리 마스크 → {집단: 같은 무리 안 연결 수}."""
    gs = 12345 + b
    rs = np.random.RandomState(19300 + gs)
    kx = {}
    for sd in "lr":
        kx[sd] = rs.random_sample(NKC) < GFRAC
        rs.random_sample((NCX, NKC)); rs.random_sample((NCY, NKC))   # X·Y 집단 억제 마스크 몫(소비 순서만 맞춤)
    ri = np.random.RandomState(19900 + gs)
    ix = {}
    for sd in "lr":
        m = np.zeros(NIH, dtype=bool)
        m[ri.permutation(NIH)[:NIH // 2]] = True
        ix[sd] = m
    S = np.load(os.path.join(EXP, "traces", "E176", "snap", "conn_b%d.npz" % b))
    out = {}
    for nm in NAMES:
        sd = nm[-1]   # 이름 끝 글자 = 쪽(kc_l_to_inh_l·kc_inh_l_to_kc_l)
        pre, post = S[nm + "__pre"].astype(np.int64), S[nm + "__post"].astype(np.int64)
        a, c = (kx[sd], ix[sd]) if "_to_inh_" in nm else (ix[sd], kx[sd])
        out[nm] = int(np.sum(a[pre] == c[post]))
    return out


RX_DEV = re.compile(r"^\s*\[E209 구획 되먹임 억제 확인\] 장치 — (.*)$", re.M)
RX_DEV1 = re.compile(r"(kc_\S+) (\d+)\(같음·무리 밖 0·g (\S+)\)")
RX_CB = re.compile(r"^\s*\[E209 구획 되먹임 억제\] KC 억제 뉴런 무리 X 좌 (\d+)·우 (\d+) / (\d+), 같은 무리 안 연결만 — .*, 가중치 KC→억제 (\S+)·억제→KC (\S+)\(2배\)$", re.M)


def compart_ok(t, exp, nmin):
    """로그 본문 t — 구성 줄·장치 줄 같은 수 ≥ nmin, 구성 줄 무리 X 200·200 / 400·가중치 '10.00'·'-30.00', 장치 줄마다 4집단 연결 수 = exp·g 문자열 맞음."""
    cb, dv = RX_CB.findall(t), RX_DEV.findall(t)
    if len(cb) < nmin or len(cb) != len(dv):
        return False
    if any(c != ("200", "200", "400", "10.00", "-30.00") for c in cb):
        return False
    for d in dv:
        g = {nm: (int(n), w) for nm, n, w in RX_DEV1.findall(d)}
        if set(g) != set(NAMES) or any(g[nm] != (exp[nm], "10.00" if "_to_inh_" in nm else "-30.00") for nm in NAMES):
            return False
    return True


def flines(t, b, tc):
    """로그 본문 t — F 적재 줄이 모두 그 뇌 파일·검증 일치이고 ≥ 1, 동작점 줄이 모두 T_C 이고 ≥ 1."""
    fl = re.findall(r"^\[E184 특징 가중치 적재\] (\S+)( 검증 일치)?", t, re.M)
    tn = re.findall(r"\[motor 동작점\] .*?기저값 (-?\d+)\.0\(", t)
    f_ok = bool(fl) and all(g == " 검증 일치" and p.replace("\\", "/").endswith("traces/E206/feat_formed_b%d.npz" % b) for p, g in fl)
    t_ok = bool(tn) and all(int(x) == tc for x in tn)
    return f_ok and t_ok


RX_CW = re.compile(r"^\s*\[맥락 입력(?: 2)?\] 맥락 전용 집단 (zz_ctx2?) \d+ → KC 좌·우, w=(\S+), p=(\S+), 연결 좌 (\d+)·우 (\d+)$", re.M)


def cwlines(t):
    d = {}
    for pop, w, p, nl, nr in RX_CW.findall(t):
        d.setdefault(pop, set()).add((w, p, int(nl), int(nr)))
    return d


def cw_ok(t, want):
    d = cwlines(t)
    return set(d) == {"zz_ctx", "zz_ctx2"} and all(v == {("0.00", "0.020") + want[k]} for k, v in d.items())


def evlog(path, b=None, tc=None):
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
        info["f"] = flines(t, b, tc)
    info["text"] = t
    return d, info


def trlog(path, b, tc):
    t = open(path, encoding="utf-8", errors="replace").read()
    g = re.search(r"^\[E204 맥락 게이팅 KC\] 시행 (\d+) · .*?비\(비활성/활성\) ([0-9.]+)$", t, re.M)
    c = re.search(r"^\[범주 규칙 맥락\] 시행 (\d+) 중 맥락 (X|Y) 켬 (\d+)$", t, re.M)
    w = re.search(r"^\[범주 과제\] catrule — .*?\(세계 시드 (\d+)\).*?배정 지역 난수 (\d+),", t, re.M)
    rv = re.search(r"^\[범주 반전\] 반전 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    p2 = re.search(r"^\[둘째 범주 쌍\] 둘째 쌍 시행 (\d+) \(시행 (\d+) 부터, 전체 (\d+)\)$", t, re.M)
    return {"gate": (int(g.group(1)), float(g.group(2))) if g else None, "cnt": (int(c.group(1)), c.group(2), int(c.group(3))) if c else None,
            "seed": (int(w.group(1)), int(w.group(2))) if w else None, "rev": rv.groups() if rv else None, "p2": p2.groups() if p2 else None,
            "f": flines(t, b, tc), "text": t}


def l4cat(a):
    tr = [a[k][0] for k in SIGN]; nw = [a[k][1] for k in SIGN]
    acq = sum(v >= 1000 for v in tr)
    if acq == 4:
        return ("conc" if all(n_ >= 500 and 2 * n_ >= t_ for t_, n_ in zip(tr, nw)) else "memo"), min(tr)
    return ("part" if acq >= 2 else "fail"), min(tr)


def main():
    L = os.path.join(EXP, "logs", "E209")
    L8 = os.path.join(EXP, "logs", "E208")
    L6 = os.path.join(EXP, "logs", "E206")
    bad = []
    cat, m = {}, {}
    st = {"R": {}, "V": {}, "K": {}}
    try:
        tc = int(re.search(r"T_C=(-?\d+)\b", open(os.path.join(L, "calib_tonic.txt"), encoding="utf-8").read()).group(1))
        for b in BR:
            c6 = cwlines(open(os.path.join(L6, "train_GW_b%d.log" % b), encoding="utf-8", errors="replace").read())
            if set(c6) != {"zz_ctx", "zz_ctx2"} or any(len(v) != 1 for v in c6.values()):
                raise ValueError("E206 b%d 맥락 연결 줄" % b)
            want = {k: next(iter(v))[2:] for k, v in c6.items()}
            exp = expected_compart(b)
            tl = trlog(os.path.join(L, "train_GC_b%d.log" % b), b, tc)
            if tl["seed"] != (38300, 38500 + b) or not (tl["gate"] and tl["gate"][0] == 3000 and tl["gate"][1] <= 0.01) or not tl["f"] or not cw_ok(tl["text"], want) or not compart_ok(tl["text"], exp, 2):
                bad.append("F-G0C b%d 학습(시드 %s·게이팅 %s·F/동작점 %s·맥락 무게 %s·구획 %s)" % (b, tl["seed"], tl["gate"], tl["f"], cw_ok(tl["text"], want), compart_ok(tl["text"], exp, 2)))
            a, same = {}, True
            for x in "XY":
                dl, il = evlog(os.path.join(L, "ev_GC_b%d_learn_%s.log" % (b, x)), b, tc)
                dn, inn = evlog(os.path.join(L, "ev_GC_b%d_none_%s.log" % (b, x)), b, tc)
                d8, _ = evlog(os.path.join(L8, "ev_GW_b%d_learn_%s.log" % (b, x)))
                same = same and dl == d8
                for nm, inf, pw in (("learn", il, 8), ("none", inn, 0)):
                    if (inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {x} or inf["ctxon"] != {"96"}
                            or inf["trials"] != {(96, 96)} or not inf["f"] or not cw_ok(inf["text"], want) or not compart_ok(inf["text"], exp, 1)):
                        bad.append("F-G0C b%d %s %s 평가 %s" % (b, nm, x, {k_: v_ for k_, v_ in inf.items() if k_ != "text"}))
                for c in "AB":
                    a[(x, c)] = tuple(SIGN[(x, c)] * (dl[(c, s)] - dn[(c, s)]) for s in ("train", "new"))
            if same:
                bad.append("F-G0C b%d learn = E208 F-G0 learn(같은 값)" % b)
            cat[b], m[b] = l4cat(a)
            for t, nt in (("R", 1500), ("V", 3000), ("P", 3000)):
                tl = trlog(os.path.join(L, "train_%s_b%d.log" % (t, b)), b, tc)
                if not (tl["gate"] and tl["gate"][0] == nt and tl["gate"][1] <= 0.01 and tl["cnt"] == (nt, "X", nt) and tl["seed"] == (38300, 38500 + b) and tl["f"]
                        and cw_ok(tl["text"], want) and compart_ok(tl["text"], exp, 2)):
                    bad.append("G0C-%s b%d 학습(게이팅 %s·켬 %s·시드 %s·F/동작점 %s·맥락 무게 %s·구획 %s)" % (t, b, tl["gate"], tl["cnt"], tl["seed"], tl["f"], cw_ok(tl["text"], want), compart_ok(tl["text"], exp, 2)))
                if (tl["rev"] is not None) != (t == "V") or (t == "V" and tl["rev"] != ("1500", "1500", "3000")):
                    bad.append("G0C-%s b%d 반전 줄 %s" % (t, b, tl["rev"]))
                if (tl["p2"] is not None) != (t == "P") or (t == "P" and tl["p2"] != ("1500", "1500", "3000")):
                    bad.append("G0C-%s b%d 둘째 쌍 줄 %s" % (t, b, tl["p2"]))
            D = {}
            for nm, pw in (("R_learn", 8), ("V_learn", 8), ("P_learn", 8), ("none", 0)):
                d, inf = evlog(os.path.join(L, "ev_b%d_%s.log" % (b, nm)), b, tc)
                if (inf["pushed"] != pw or inf["world"] != 38300 or inf["proto"] != (5, 14, 15, 19, 29) or inf["ctx"] != {"X"} or inf["ctxon"] != {"96"}
                        or inf["trials"] != {(96, 96)} or not inf["f"] or not cw_ok(inf["text"], want) or not compart_ok(inf["text"], exp, 1)):
                    bad.append("G0C b%d %s 평가 %s" % (b, nm, {k_: v_ for k_, v_ in inf.items() if k_ != "text"}))
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
    print("독립 F-G0C: 범주 %s · 가장 약한 칸 %s" % (" ".join("b%d %s" % (b, cat[b]) for b in BR), " ".join("%+d" % m[b] for b in BR)))
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
        print("독립 가설: H130 지지")
        print("독립 판정: 통합 관문 통과(개발 비교)")
        return 0
    if sp == "잃음":
        h = "H130-null(유지 안 섬)"
    elif sp == "보존":
        h = "H130-맞교환(유지 ↔ L4)" if sl4 != "보존" else ("H130 부분(유지는 섬 — 범주 규칙 %s)" % sr if sr != "보존" else "맞교환(유지 ↔ 반전)")
    else:
        h = "H130 부분·보류"
    print("독립 가설: %s" % h)
    print("독립 판정: 통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (sl4, sr, sv, sp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
