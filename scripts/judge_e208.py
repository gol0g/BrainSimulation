#!/usr/bin/env python3
"""E208 판정 — 맥락 흥분 없이 게이팅만(G-f):
E206 F-GW 고정 체계(형성 표현 F·게이팅·경쟁형 η 2.4·도파민 기준선·흔적 동결, 앞 능력 맥락 X 고정)에서 C2 맥락 → KC 흥분 무게만 20 → 0(연결·확률 그대로),
동작점은 결과 전 규칙으로 다시 맞춘 T_F0. 같은 뇌 16~20·세계 38300·같은 시행 시드, 표현 F = traces/E206/feat_formed_b{B}(형성 다시 안 함). 짝 = E206 F-GW(같은 뇌).
능력 정의는 E205·E206 과 같다(judge_e205 의 cat_rule·cat_rev·cat_keep·status_*, L4 범주 judge_e204 category·weakest).
입력: logs/E208/train_{GW|R|V|P}_b{B}.log, ev_GW_b{B}_{learn|none}_{X|Y}.log, ev_b{B}_{R_learn|V_learn|P_learn|none}.log, logs/E208/calib_tonic.txt('T_F=…'),
  traces/E208/w_{GW|R|V|P}_b{B}.npz; 짝 logs/E206·traces/E206(judge_e206 의 읽기·검사 함수 그대로).
조작검증(하나라도 어기면 보류): E206 과 같은 검사(F 적재·동작점 = T_F0·학습 줄·특징/맥락 점검·시드·맥락 고정·반전/둘째 쌍 줄·가중치·평가) +
  맥락 흥분 무게 0 — 모든 학습·평가 로그의 '[맥락 입력] … zz_ctx …'·'[맥락 입력 2] … zz_ctx2 …' 줄이 각각 ≥ 1, 모두 w=0.00·p=0.020, 연결 수(좌·우)가 E206 같은 뇌 F-GW 학습 로그와 같음;
  F-G0 L4 가중치는 E206 F-GW 가중치와 다름; 같은 값 — 어느 뇌든 F-G0 learn 8값이 E206 F-GW learn 과 모두 같으면 조작 무효 의심.
판정: L4 = F-G0 맥락 의존 범주 ≥ 4/5 보존·실패 ≥ 4/5 잃음·그 밖 부분, 앞 능력은 E205 와 같은 상태 함수. 종합: 조작검증 실패 → 보류 / 넷 다 보존 → 통합 관문 통과(개발 비교, H129 지지) /
  그 밖 → 통합 관문 미통과(능력별) — 가설 줄: 유지 잃음 H129-null / 유지 보존·L4 아님 H129-맞교환(유지 ↔ L4) / 유지·L4 보존·범주 규칙 아님 H129 부분 / 유지·L4·범주 규칙 보존·반전 아님 '맞교환(유지 ↔ 반전)' / 유지 부분·보류 'H129 부분·보류'.
실행: python3 scripts/judge_e208.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import read_eval, read_train  # noqa: E402
import judge_e204 as J4  # noqa: E402 — read_gate·category·weakest·cells
import judge_e205 as J5  # noqa: E402 — check_common·read_rule·read_eval_proto·motor_mean·weights_ok·cat_*·status_*
import judge_e206 as J6  # noqa: E402 — read_feat·read_tonic·read_calib·check_eval_f·status_l4, 짝(E206 F-GW) 읽기

EXP = "research/experiments"
BRAINS = J5.BRAINS
WORLD, TRIAL0, CTX0, PROTO_A = J5.WORLD, J5.TRIAL0, J5.CTX0, J5.PROTO_A
NL4 = J5.NL4
TASKS, BRANGE, EVALS = J5.TASKS, J5.BRANGE, J5.EVALS
CTXP = 0.02
RX_CW = re.compile(r"\[맥락 입력(?: 2)?\] 맥락 전용 집단 (zz_ctx2?) (\d+) → KC 좌·우, w=(-?[0-9.]+), p=([0-9.]+), 연결 좌 (\d+)·우 (\d+)")


def read_calib_signed(path):
    """calib_tonic.txt 'T_F=-1300 …' 또는 'T_F=800 …' → 정수(E208 격자는 양수 기저값을 포함 — judge_e206.read_calib 은 음수만 읽음)."""
    m = re.search(r"T_F=(-?\d+)\b", open(path, encoding="utf-8").read())
    if not m:
        raise ValueError("calib_tonic.txt 에 T_F 없음")
    return int(m.group(1))


def read_ctxw(path):
    """'[맥락 입력] 맥락 전용 집단 zz_ctx 200 → KC 좌·우, w=0.00, p=0.020, 연결 좌 N·우 M' → [(집단, n, w, p, 좌, 우), …]."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RX_CW.search(ln)
        if m:
            out.append((m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6))))
    return out


def counts6(b):
    """E206 같은 뇌 F-GW 학습 로그의 맥락 연결 수 {집단: (좌, 우)} — 집단마다 한 값이어야 한다."""
    cw = read_ctxw(os.path.join(EXP, "logs", "E206", "train_GW_b%d.log" % b))
    d = {}
    for pop, _, _, _, nl, nr in cw:
        d.setdefault(pop, set()).add((nl, nr))
    if set(d) != {"zz_ctx", "zz_ctx2"} or any(len(v) != 1 for v in d.values()):
        raise ValueError("E206 b%d 맥락 연결 줄 %s" % (b, d))
    return {k: next(iter(v)) for k, v in d.items()}


def check_ctxw(tag, path, want, why):
    """맥락 흥분 무게 0 — 두 집단 줄 ≥ 1, 모두 w 0·p 0.02, 연결 수 = E206."""
    cw = read_ctxw(path)
    pops = {c[0] for c in cw}
    good = (pops == {"zz_ctx", "zz_ctx2"} and all(c[2] == 0.0 and abs(c[3] - CTXP) < 1e-9 and (c[4], c[5]) == want[c[0]] for c in cw))
    if not good:
        why.append("%s 맥락 흥분 무게 줄 %s(연결 기대 %s)" % (tag, cw, want))
    return good


def check_eval_g0(tag, path, cats, ctx, pushed_want, b, tf, want, why):
    ok, rc = J6.check_eval_f(tag, path, cats, ctx, pushed_want, b, tf, why)
    return check_ctxw(tag, path, want, why) and ok, rc


def check_l4_g0(b, tf, want, why):
    L = os.path.join(EXP, "logs", "E208")
    tag = "F-G0 b%d" % b
    tp = os.path.join(L, "train_GW_b%d.log" % b)
    tr = read_train(tp)
    ok, st, gs, sd = J5.check_common(tag, tp, NL4, True, why)
    c, k = tr["ctx"], tr["cat"]
    if not (tr["catctx"] and tr["done"] and tr["kt"] >= 1 and c and k and c[0] == NL4 and c[1] + c[2] == NL4 and c[3] + c[4] == NL4
            and 1200 <= c[1] <= 1800 and 1200 <= c[3] <= 1800 and k[0] == NL4 and 1200 <= k[1] <= 1800):
        ok = False; why.append("%s 학습 줄(%s %s)" % (tag, c, k))
    if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
        ok = False; why.append("%s 특징 점검 %s" % (tag, tr["fchk"]))
    if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
        ok = False; why.append("%s 맥락 점검 %s" % (tag, tr["cchk"]))
    if not (sd["world"] == WORLD and sd["trial"] == TRIAL0 + b and sd["ctx"] == CTX0 + b):
        ok = False; why.append("%s 시드 줄(%s)" % (tag, sd))
    ok = J6.check_f_lines(tag, tp, b, tf, why) and ok
    ok = check_ctxw(tag, tp, want, why) and ok
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E208", "w_GW_b%d.npz" % b), b,
                             os.path.join(EXP, "traces", "E206", "w_GW_b%d.npz" % b), why)
    ok = ok and w_ok
    ev, good = {}, True
    for w in ("learn", "none"):
        for x in "XY":
            e_ok, rc = check_eval_g0("%s %s %s" % (tag, w, x), os.path.join(L, "ev_GW_b%d_%s_%s.log" % (b, w, x)), "AB", x,
                                     8 if w == "learn" else 0, b, tf, want, why)
            ok = ok and e_ok; good = good and e_ok
            ev[(w, x)] = (rc,)
    a = J4.cells(ev) if good else None
    mn = sum(J5.motor_mean(os.path.join(L, "ev_GW_b%d_none_%s.log" % (b, x))) for x in "XY") / 2
    learn8 = {(x, cs): ev[("learn", x)][0].get(cs) for x in "XY" for cs in [(c_, s_) for c_ in "AB" for s_ in ("train", "new")]}
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb, "motor": mn, "learn8": learn8}
    return ok, a, rep


def check_task_g0(t, b, tf, want, why):
    L = os.path.join(EXP, "logs", "E208")
    tag = "G0-%s b%d" % (t, b)
    tp = os.path.join(L, "train_%s_b%d.log" % (t, b))
    n = TASKS[t]
    tr = read_train(tp)
    rr = J5.read_rule(tp)
    ok, st, gs, sd = J5.check_common(tag, tp, n, True, why)
    k = tr["cat"]
    lo, hi = BRANGE[t]
    if not (not tr["catctx"] and tr["done"] and tr["kt"] >= 1 and k and k[0] == n and lo <= k[1] <= hi and k[1] + k[2] == n):
        ok = False; why.append("%s 학습 줄(catctx %s·%s)" % (tag, tr["catctx"], k))
    if len(tr["fchk"]) != 3 or not all(x > 0 and y == 0 for x, y in tr["fchk"]):
        ok = False; why.append("%s 특징 점검 %s" % (tag, tr["fchk"]))
    if len(tr["cchk"]) != 3 or not all(x > 0 and y == 0 and z == 0 for x, y, z in tr["cchk"]):
        ok = False; why.append("%s 맥락 점검 %s" % (tag, tr["cchk"]))
    if rr["ann"] != ["X"] or rr["cnt"] != (n, "X", n):
        ok = False; why.append("%s 맥락 고정(알림 %s·켬 %s)" % (tag, rr["ann"], rr["cnt"]))
    rev_ok = (rr["rev"] == [] if t != "V" else
              (len(rr["rev"]) == 2 and rr["rev"][0].startswith("[범주 반전] 시행 1500 부터")
               and rr["rev"][1] == "[범주 반전] 반전 시행 1500 (시행 1500 부터, 전체 3000)"))
    p2_ok = (rr["p2"] == [] if t != "P" else
             (len(rr["p2"]) == 2 and rr["p2"][0].startswith("[둘째 범주 쌍] 시행 1500 부터")
              and rr["p2"][1] == "[둘째 범주 쌍] 둘째 쌍 시행 1500 (시행 1500 부터, 전체 3000)"))
    if not (rev_ok and p2_ok):
        ok = False; why.append("%s 반전·둘째 쌍 줄(%s / %s)" % (tag, rr["rev"], rr["p2"]))
    if not (sd["world"] == WORLD and sd["trial"] == TRIAL0 + b and sd["ctx"] is None):
        ok = False; why.append("%s 시드 줄(%s)" % (tag, sd))
    ok = J6.check_f_lines(tag, tp, b, tf, why) and ok
    ok = check_ctxw(tag, tp, want, why) and ok
    other = os.path.join(EXP, "traces", "E208", "w_%s_b%d.npz" % ("V" if t == "R" else "R", b))
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E208", "w_%s_b%d.npz" % (t, b)), b, other, why)
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb}
    return ok and w_ok, rep


def task_evals_g0(b, tf, want, why):
    L = os.path.join(EXP, "logs", "E208")
    ok, rcs = True, {}
    for nm, _, cats, pw in EVALS:
        e_ok, rc = check_eval_g0("G0 b%d %s" % (b, nm), os.path.join(L, "ev_b%d_%s.log" % (b, nm)), cats, "X", pw, b, tf, want, why)
        ok = ok and e_ok
        rcs[nm] = rc
    if not ok:
        return False, None, float("nan")
    e = {}
    for nm, t, cats, _ in EVALS:
        if t is None:
            continue
        e[t] = {(c, s): rcs[nm][(c, s)] - rcs["none"][(c, s)] for c in cats for s in ("train", "new")}
    return True, e, J5.motor_mean(os.path.join(L, "ev_b%d_none.log" % b))


def decide(sl4, sr, sv, sp):
    """종합 판정 문자열과 H129 줄."""
    if sl4 == "보존" and sr == "보존" and sv == "보존" and sp == "보존":
        return "통합 관문 통과(개발 비교)", "H129 지지"
    v = "통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (sl4, sr, sv, sp)
    if sp == "잃음":
        h = "H129-null(유지 안 섬)"
    elif sp == "보존":
        if sl4 != "보존":
            h = "H129-맞교환(유지 ↔ L4)"
        elif sr != "보존":
            h = "H129 부분(유지는 섬 — 범주 규칙 %s)" % sr
        else:
            h = "맞교환(유지 ↔ 반전)"
    else:
        h = "H129 부분·보류"
    return v, h


def ratio(num, den):
    return num / den if den else float("nan")


def main():
    J5.EXP = EXP
    J6.EXP = EXP
    J6.J5.EXP = EXP
    ok, why = True, []
    A, RL, RT, E, MN = {}, {}, {t: {} for t in TASKS}, {}, {}
    G_a, G_e, G_l8, G_mn = {}, {}, {}, {}
    try:
        tf = read_calib_signed(os.path.join(EXP, "logs", "E208", "calib_tonic.txt"))
        tf6 = J6.read_calib(os.path.join(EXP, "logs", "E206", "calib_tonic.txt"))
        for b in BRAINS:
            want = counts6(b)
            ok_, a, rep = check_l4_g0(b, tf, want, why)
            ok = ok and ok_
            if a is not None:
                A[b] = a
            RL[b] = rep
            for t in TASKS:
                ok_, rep = check_task_g0(t, b, tf, want, why)
                ok = ok and ok_
                RT[t][b] = rep
            ok_, e, mn = task_evals_g0(b, tf, want, why)
            ok = ok and ok_
            if e is not None:
                E[b] = e
            MN[b] = mn
            # 짝(E206 F-GW, 같은 뇌) — 같은 값 점검·보고. E206 자체 검사는 E206 판정에서 통과(여기서는 값만)
            w6 = []
            _, ga, grep_ = J6.check_l4_f(b, tf6, w6)
            _, ge, gmn = J6.task_evals_f(b, tf6, w6)
            G_a[b], G_e[b], G_l8[b], G_mn[b] = ga, ge, grep_["learn8"], grep_["motor"]
    except (OSError, ValueError, IndexError, KeyError, TypeError, AttributeError) as ex:
        print("판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    same = [b for b in BRAINS if RL[b]["learn8"] == G_l8[b] and None not in G_l8[b].values()]
    if same:
        ok = False; why.append("같은 값(F-G0 learn 8값 = E206 F-GW) 뇌 %s — 조작 무효 의심" % same)
    if not ok or len(A) != len(BRAINS) or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    print("동작점 T_F0 %d (logs/E208/calib_tonic.txt) · 짝 E206 T_F %d" % (tf, tf6))
    cat = {b: J4.category(A[b]) for b in BRAINS}
    m = {b: J4.weakest(A[b]) for b in BRAINS}
    for b in BRAINS:
        r_ = RL[b]
        mg = J4.weakest(G_a[b]) if G_a[b] is not None else None
        print("F-G0 b%d 규칙 방향 효과(1e-4, 노출/새) %s | 범주 %s · 가장 약한 칸 %+d%s | 게이팅 비 %s · 평균 DA %+.4f · 경계 시냅스 %.3f · 무학습 motor 평균 %.4f(E206 F-GW %.4f)"
              % (b, J5.fmt(A[b]), cat[b], m[b], (" (E206 F-GW %+d · G0 − GW %+d)" % (mg, m[b] - mg)) if mg is not None else "",
                 ("%.4f" % r_["ratio"]) if r_["ratio"] is not None else "—", r_["mda"], r_["fb"], r_["motor"], G_mn[b]))
    CR, CV, K = {}, {}, {}
    for b in BRAINS:
        eR, eV, eP = E[b]["R"], E[b]["V"], E[b]["P"]
        CR[b], CV[b], K[b] = J5.cat_rule(eR), J5.cat_rev(eV, eR), J5.cat_keep(eP, eR)
        rA, rB = ratio(eP[("A", "train")], eR[("A", "train")]), ratio(eP[("B", "train")], eR[("B", "train")])
        ge = G_e[b]
        if ge is not None:
            grA, grB = ratio(ge["P"][("A", "train")], ge["R"][("A", "train")]), ratio(ge["P"][("B", "train")], ge["R"][("B", "train")])
            gtxt = (" | (보고) E206 F-GW: R A %+d B %+d · V A %+d B %+d · r_A %.2f r_B %.2f · R 의 C %+d·D %+d → G0 − GW: R A %+d B %+d · V A %+d B %+d · r_A %+.2f r_B %+.2f"
                    % (ge["R"][("A", "train")], ge["R"][("B", "train")], ge["V"][("A", "train")], ge["V"][("B", "train")], grA, grB,
                       ge["R"][("C", "train")], ge["R"][("D", "train")],
                       eR[("A", "train")] - ge["R"][("A", "train")], eR[("B", "train")] - ge["R"][("B", "train")],
                       eV[("A", "train")] - ge["V"][("A", "train")], eV[("B", "train")] - ge["V"][("B", "train")], rA - grA, rB - grB))
        else:
            gtxt = " | (보고) E206 F-GW 짝 없음"
        print("b%d 범주 규칙(R) A %+d/%+d B %+d/%+d → %s | 반전(V) A %+d/%+d B %+d/%+d · Δ A %+d B %+d → %s | 유지(P) A %+d B %+d · C %+d/%+d D %+d/%+d · r_A %.2f r_B %.2f → %s"
              " | (보고) R 의 C %+d·D %+d · 게이팅 비 R %.4f V %.4f P %.4f · 평균 DA R %+.4f V %+.4f P %+.4f · 무학습 motor 평균(맥락 X) %.4f%s"
              % (b, eR[("A", "train")], eR[("A", "new")], eR[("B", "train")], eR[("B", "new")], CR[b],
                 eV[("A", "train")], eV[("A", "new")], eV[("B", "train")], eV[("B", "new")], eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")], CV[b],
                 eP[("A", "train")], eP[("B", "train")], eP[("C", "train")], eP[("C", "new")], eP[("D", "train")], eP[("D", "new")], rA, rB,
                 K[b][2] if K[b][2] != "na" else ("기준 미달" if not K[b][0] else "전제 미달"),
                 eR[("C", "train")], eR[("D", "train")], RT["R"][b]["ratio"], RT["V"][b]["ratio"], RT["P"][b]["ratio"],
                 RT["R"][b]["mda"], RT["V"][b]["mda"], RT["P"][b]["mda"], MN[b], gtxt))
    print("F-G0: 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 실패 %d/5" % tuple(sum(cat[b] == c for b in BRAINS) for c in ("conc", "memo", "part", "fail")))
    sl4, sr, sv, sp = J6.status_l4(cat), J5.status_rule(CR), J5.status_rev(CV), J5.status_keep(K)
    print("L4: F-G0 맥락 의존 범주 %d/5 → %s" % (sum(cat[b] == "conc" for b in BRAINS), sl4))
    print("범주 규칙: 개념 수준 %d/5 · 사례 한정 %d/5 · 한 범주만 %d/5 · 획득 실패 %d/5 → %s" % (*(sum(v == c for v in CR.values()) for c in ("concept", "memo", "one", "fail")), sr))
    print("반전: 반전 성공 %d/5 · 사례 한정 %d/5 · 부분 반전 %d/5 · 고착 %d/5 · 그 밖 %d/5 → %s" % (*(sum(v == c for v in CV.values()) for c in ("success", "caseonly", "partial", "stuck", "other")), sv))
    print("유지: 기준 %d/5 · 전제 %d/5 · 유지 %d/5 · 간섭 %d/5 → %s" % (sum(v[0] for v in K.values()), sum(v[1] for v in K.values()),
                                                                    sum(v[2] == "keep" for v in K.values()), sum(v[2] == "interf" for v in K.values()), sp))
    v, h = decide(sl4, sr, sv, sp)
    print("가설: %s" % h)
    print("조작검증 통과")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
