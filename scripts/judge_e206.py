#!/usr/bin/env python3
"""E206 판정 — 형성 표현 F + 맥락 게이팅 통합: E205 고정 체계(게이팅·경쟁형 η 2.4·도파민 기준선·흔적 동결·C2, 앞 능력 맥락 X 고정)에서
표현만 U → F(세계 38300 네 범주 노출 형성 — E191 절차), 같은 뇌 16~20·세계 38300·E205 와 같은 시행 시드. 규칙 = research/experiments/logs/E206/criteria_fixed.txt.
능력 정의는 E205 와 같다(judge_e205 의 cat_rule·cat_rev·cat_keep·status_* — E185·E188·E190 원문), L4 범주는 E193~E205(judge_e204 category·weakest).
입력: logs/E206/form_b{B}.log, train_{GW|R|V|P}_b{B}.log, ev_GW_b{B}_{learn|none}_{X|Y}.log, ev_b{B}_{R_learn|V_learn|P_learn|none}.log, logs/E206/calib_tonic.txt,
  traces/E206/feat_formed_b{B}.npz·w_{GW|R|V|P}_b{B}.npz, 스냅숏 traces/E176/snap/conn_b{B}.npz; 짝(같은 값 점검·보고) = logs/E205 같은 뇌 GW·R·V·P(judge_e205 경로).
조작검증(하나라도 어기면 보류): 형성 — E191 form_ok(제시 816·노출 A,B,C,D·RCW 2·RCJ 12·RCCHK pre·pre2·post·RAWCAT 1) + 세계 줄 seed 38300·원형 A + AC·BD J_post ≤ 0.1000(두 쪽)·저장 파일;
  F 적재 — 모든 학습·평가 로그 '[E184 특징 가중치 적재] <그 뇌 feat_formed> 검증 일치' ≥ 1(경로가 그 뇌 파일); 동작점 — 모든 학습·평가 '[motor 동작점] … 기저값' = T_F(calib_tonic.txt), ≥ 1;
  E205 검사(check_common·학습 줄·특징/맥락 점검·시드·맥락 고정·반전/둘째 쌍 줄·가중치(F-GW 는 E205 U GW 와 다름)·평가 — 평가 특징 적재 ≥ 1);
  같은 값 — 어느 뇌든 F-GW learn 8값이 E205 GW learn 과 모두 같으면 조작 무효 의심.
판정: L4 = F-GW 맥락 의존 범주 ≥ 4/5 보존·실패 ≥ 4/5 잃음·그 밖 부분, 앞 능력은 E205 와 같은 상태 함수. 종합: 조작검증 실패 → 보류 / 넷 다 보존 → 통합 관문 통과(개발 비교) /
  그 밖 → 통합 관문 미통과(능력별) — 가설 줄: 유지 잃음 H127-null / 유지 보존·L4 아님 'F 가 L4 를 깸' / 유지·L4 보존·범주 규칙 아님 'H127 부분' / 유지·L4·범주 규칙 보존·반전 아님 '맞교환' / 유지 부분·보류 'H127 부분·보류'.
실행: python3 scripts/judge_e206.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import CELLS, read_eval, read_train  # noqa: E402
import judge_e191 as J1  # noqa: E402 — read_form·form_ok(형성 확인)
import judge_e204 as J4  # noqa: E402 — read_gate·category·weakest·cells
import judge_e205 as J5  # noqa: E402 — check_common·read_rule·read_eval_proto·motor_mean·weights_ok·cat_*·status_*·짝(E205) 계산

EXP = "research/experiments"
BRAINS = J5.BRAINS
WORLD, TRIAL0, CTX0, PROTO_A = J5.WORLD, J5.TRIAL0, J5.CTX0, J5.PROTO_A
NL4 = J5.NL4
TASKS, BRANGE, EVALS = J5.TASKS, J5.BRANGE, J5.EVALS
JMAX = 1000          # AC·BD J_post ≤ 0.1000(1e-4)


def read_feat(path):
    """'[E184 특징 가중치 적재] <경로> 검증 일치 …' → [경로]. '검증 일치' 없는 적재 줄은 None 으로."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[E184 특징 가중치 적재]"):
            m = re.match(r"\[E184 특징 가중치 적재\] (\S+) 검증 일치", ln)
            out.append(m.group(1) if m else None)
    return out


def read_tonic(path):
    """'[motor 동작점] motor 좌·우 Ioffset 기저값 -2800.0(긴장성 억제)' → [-2800.0, …]."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        if "[motor 동작점]" in ln:
            m = re.search(r"기저값 (-?[0-9]+(?:\.[0-9]+)?)\(", ln)
            out.append(float(m.group(1)) if m else float("nan"))
    return out


def read_calib(path):
    """calib_tonic.txt 'T_F=-2800 …' → -2800."""
    m = re.search(r"T_F=(-\d+)\b", open(path, encoding="utf-8").read())
    if not m:
        raise ValueError("calib_tonic.txt 에 T_F 없음")
    return int(m.group(1))


def read_form_extra(path):
    """형성 로그 — 세계 줄 (seed, 원형 A 앞 5개), RCJ 반응 KC 수 {(쪽, 쌍): (n_pre_a, n_pre_b, n_post_a, n_post_b)}."""
    world, nk = [], {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        if ln.startswith("[원시 특징 세계] "):
            m = re.search(r"seed (\d+) \| 원형 A ([0-9]+(?:,[0-9]+){4})", ln)
            world.append((int(m.group(1)), m.group(2)) if m else (-1, ""))
        elif ln.startswith("RCJ "):
            m = re.search(r"side=([lr]) pair=([A-D]{2}) .*n_pre=(\d+)/(\d+) n_post=(\d+)/(\d+)", ln)
            if m:
                nk[(m.group(1), m.group(2))] = tuple(int(m.group(i)) for i in range(3, 7))
    return world, nk


def ff_path(b):
    return os.path.join(EXP, "traces", "E206", "feat_formed_b%d.npz" % b)


def check_f_lines(tag, path, b, tf, why):
    """F 적재 줄(그 뇌 파일·검증 일치) ≥ 1, 동작점 줄 전부 T_F ≥ 1."""
    ok = True
    fl = read_feat(path)
    want = "traces/E206/feat_formed_b%d.npz" % b
    if not fl or not all(p is not None and p.replace("\\", "/").endswith(want) for p in fl):
        ok = False; why.append("%s F 적재 줄 %s" % (tag, fl))
    tl = read_tonic(path)
    if not tl or not all(t == float(tf) for t in tl):
        ok = False; why.append("%s 동작점 줄 %s(T_F %d)" % (tag, tl, tf))
    return ok


def check_form(b, why):
    L = os.path.join(EXP, "logs", "E206")
    tag = "형성 b%d" % b
    path = os.path.join(L, "form_b%d.log" % b)
    fm = J1.read_form(path)
    world, nk = read_form_extra(path)
    ok = True
    if not J1.form_ok(fm):
        ok = False; why.append("%s E191 형성 검사(제시 %s·범주 %s·RCW %d·RCJ %d·RAWCAT %d·RCCHK %d)"
                               % (tag, fm["expo"], fm["expo_cats"], len(fm["rcw"]), len(fm["rcj"]), fm["rawcat"], len(fm["chk"])))
    if world != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 세계 줄 %s" % (tag, world))
    jp = {(s, p): fm["rcj"].get((s, p), (None, None)) for s in "lr" for p in ("AC", "BD")}
    if not all(v[1] is not None and v[1] <= JMAX for v in jp.values()):
        ok = False; why.append("%s 겹침 J_post %s(문턱 %d)" % (tag, jp, JMAX))
    if not os.path.exists(ff_path(b)) or os.path.getsize(ff_path(b)) == 0:
        ok = False; why.append("%s 저장 파일 없음" % tag)
    return ok, {"J": jp, "n": {k: nk.get(k) for k in jp}}


def check_eval_f(tag, path, cats, ctx, pushed_want, b, tf, why):
    """평가 한 개(F) — E205 check_eval 과 같되 특징 적재 ≥ 1·F 경로·동작점. 반환 (ok, rc)."""
    ok = True
    rc, pushed, kt, fw, sat, bad = read_eval(path, ctx)
    want = {(c, s) for c in cats for s in ("train", "new")}
    if set(rc) != want or bad or sat or kt < 1 or fw < 1 or pushed != pushed_want:
        ok = False; why.append("%s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (tag, len(rc), bad, sat, kt, fw, pushed))
    pw = J5.read_eval_proto(path)
    if pw != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 평가 세계·원형 %s" % (tag, pw))
    if J4.read_gate(path)["build"] < 1:
        ok = False; why.append("%s 평가 게이팅 구성 줄 없음" % tag)
    ok = check_f_lines(tag, path, b, tf, why) and ok
    return ok, rc


def check_l4_f(b, tf, why):
    L = os.path.join(EXP, "logs", "E206")
    tag = "F-GW b%d" % b
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
    ok = check_f_lines(tag, tp, b, tf, why) and ok
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E206", "w_GW_b%d.npz" % b), b,
                             os.path.join(EXP, "traces", "E205", "w_GW_b%d.npz" % b), why)
    ok = ok and w_ok
    ev, good = {}, True
    for w in ("learn", "none"):
        for x in "XY":
            e_ok, rc = check_eval_f("%s %s %s" % (tag, w, x), os.path.join(L, "ev_GW_b%d_%s_%s.log" % (b, w, x)), "AB", x,
                                    8 if w == "learn" else 0, b, tf, why)
            ok = ok and e_ok; good = good and e_ok
            ev[(w, x)] = (rc,)
    a = J4.cells(ev) if good else None
    mn = sum(J5.motor_mean(os.path.join(L, "ev_GW_b%d_none_%s.log" % (b, x))) for x in "XY") / 2
    learn8 = {(x, cs): ev[("learn", x)][0].get(cs) for x in "XY" for cs in [(c_, s_) for c_ in "AB" for s_ in ("train", "new")]}
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb, "motor": mn, "learn8": learn8}
    return ok, a, rep


def check_task_f(t, b, tf, why):
    L = os.path.join(EXP, "logs", "E206")
    tag = "F-%s b%d" % (t, b)
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
    ok = check_f_lines(tag, tp, b, tf, why) and ok
    other = os.path.join(EXP, "traces", "E206", "w_%s_b%d.npz" % ("V" if t == "R" else "R", b))
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E206", "w_%s_b%d.npz" % (t, b)), b, other, why)
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb}
    return ok and w_ok, rep


def task_evals_f(b, tf, why):
    L = os.path.join(EXP, "logs", "E206")
    ok, rcs = True, {}
    for nm, _, cats, pw in EVALS:
        e_ok, rc = check_eval_f("F b%d %s" % (b, nm), os.path.join(L, "ev_b%d_%s.log" % (b, nm)), cats, "X", pw, b, tf, why)
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


def status_l4(cat):
    n = lambda c: sum(v == c for v in cat.values())  # noqa: E731
    return "보존" if n("conc") >= 4 else ("잃음" if n("fail") >= 4 else "부분")


def decide(sl4, sr, sv, sp):
    """종합 판정 문자열과 H127 줄."""
    if sl4 == "보존" and sr == "보존" and sv == "보존" and sp == "보존":
        return "통합 관문 통과(개발 비교)", "H127 지지"
    v = "통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (sl4, sr, sv, sp)
    if sp == "잃음":
        h = "H127-null(유지 안 섬)"
    elif sp == "보존":
        if sl4 != "보존":
            h = "F 가 L4 를 깸"
        elif sr != "보존":
            h = "H127 부분(유지는 섬 — 범주 규칙 %s)" % sr
        else:
            h = "맞교환(유지 ↔ 반전)"
    else:
        h = "H127 부분·보류"
    return v, h


def ratio(num, den):
    return num / den if den else float("nan")


def main():
    J5.EXP = EXP
    J1.EXP = EXP
    ok, why = True, []
    A, RL, FM, RT, E, MN = {}, {}, {}, {t: {} for t in TASKS}, {}, {}
    U_a, U_e, U_l8 = {}, {}, {}
    try:
        tf = read_calib(os.path.join(EXP, "logs", "E206", "calib_tonic.txt"))
        for b in BRAINS:
            f_ok, FM[b] = check_form(b, why)
            ok = ok and f_ok
            ok_, a, rep = check_l4_f(b, tf, why)
            ok = ok and ok_
            if a is not None:
                A[b] = a
            RL[b] = rep
            for t in TASKS:
                ok_, rep = check_task_f(t, b, tf, why)
                ok = ok and ok_
                RT[t][b] = rep
            ok_, e, mn = task_evals_f(b, tf, why)
            ok = ok and ok_
            if e is not None:
                E[b] = e
            MN[b] = mn
            # 짝(E205 U, 같은 뇌) — 같은 값 점검·보고. E205 자체 검사는 E205 판정에서 통과(여기서는 값만)
            w5 = []
            _, ua, urep = J5.check_l4("GW", b, w5)
            _, ue, _ = J5.task_evals(b, w5)
            U_a[b], U_e[b] = ua, ue
            U_l8[b] = {(x, cs): read_eval(os.path.join(EXP, "logs", "E205", "ev_GW_b%d_learn_%s.log" % (b, x)), x)[0].get(cs)
                       for x in "XY" for cs in [(c_, s_) for c_ in "AB" for s_ in ("train", "new")]}
    except (OSError, ValueError, IndexError, KeyError, TypeError, AttributeError) as ex:
        print("판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    same = [b for b in BRAINS if RL[b]["learn8"] == U_l8[b] and None not in U_l8[b].values()]
    if same:
        ok = False; why.append("같은 값(F-GW learn 8값 = E205 GW) 뇌 %s — 조작 무효 의심" % same)
    if not ok or len(A) != len(BRAINS) or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    print("동작점 T_F %d (logs/E206/calib_tonic.txt)" % tf)
    cat = {b: J4.category(A[b]) for b in BRAINS}
    m = {b: J4.weakest(A[b]) for b in BRAINS}
    for b in BRAINS:
        r_ = RL[b]; jp = FM[b]["J"]; nk = FM[b]["n"]
        mu = J4.weakest(U_a[b]) if U_a[b] is not None else None
        print("F-GW b%d 규칙 방향 효과(1e-4, 노출/새) %s | 범주 %s · 가장 약한 칸 %+d%s | 게이팅 비 %s · 평균 DA %+.4f · 경계 시냅스 %.3f · 무학습 motor 평균 %.4f"
              " | 형성 J_pre/J_post AC 좌 %d/%d 우 %d/%d · BD 좌 %d/%d 우 %d/%d · 반응 KC(AC 좌 n_pre %s → n_post %s)"
              % (b, J5.fmt(A[b]), cat[b], m[b], (" (E205 U %+d · F − U %+d)" % (mu, m[b] - mu)) if mu is not None else "",
                 ("%.4f" % r_["ratio"]) if r_["ratio"] is not None else "—", r_["mda"], r_["fb"], r_["motor"],
                 jp[("l", "AC")][0], jp[("l", "AC")][1], jp[("r", "AC")][0], jp[("r", "AC")][1],
                 jp[("l", "BD")][0], jp[("l", "BD")][1], jp[("r", "BD")][0], jp[("r", "BD")][1],
                 "%d/%d" % nk[("l", "AC")][:2] if nk[("l", "AC")] else "?", "%d/%d" % nk[("l", "AC")][2:] if nk[("l", "AC")] else "?"))
    CR, CV, K = {}, {}, {}
    for b in BRAINS:
        eR, eV, eP = E[b]["R"], E[b]["V"], E[b]["P"]
        CR[b], CV[b], K[b] = J5.cat_rule(eR), J5.cat_rev(eV, eR), J5.cat_keep(eP, eR)
        rA, rB = ratio(eP[("A", "train")], eR[("A", "train")]), ratio(eP[("B", "train")], eR[("B", "train")])
        ue = U_e[b]
        if ue is not None:
            urA, urB = ratio(ue["P"][("A", "train")], ue["R"][("A", "train")]), ratio(ue["P"][("B", "train")], ue["R"][("B", "train")])
            utxt = (" | (보고) E205 U: R A %+d B %+d · V A %+d B %+d · r_A %.2f r_B %.2f → F − U: R A %+d B %+d · V A %+d B %+d · r_A %+.2f r_B %+.2f"
                    % (ue["R"][("A", "train")], ue["R"][("B", "train")], ue["V"][("A", "train")], ue["V"][("B", "train")], urA, urB,
                       eR[("A", "train")] - ue["R"][("A", "train")], eR[("B", "train")] - ue["R"][("B", "train")],
                       eV[("A", "train")] - ue["V"][("A", "train")], eV[("B", "train")] - ue["V"][("B", "train")], rA - urA, rB - urB))
        else:
            utxt = " | (보고) E205 U 짝 없음"
        print("b%d 범주 규칙(R) A %+d/%+d B %+d/%+d → %s | 반전(V) A %+d/%+d B %+d/%+d · Δ A %+d B %+d → %s | 유지(P) A %+d B %+d · C %+d/%+d D %+d/%+d · r_A %.2f r_B %.2f → %s"
              " | (보고) R 의 C %+d·D %+d · 게이팅 비 R %.4f V %.4f P %.4f · 평균 DA R %+.4f V %+.4f P %+.4f · 무학습 motor 평균(맥락 X) %.4f%s"
              % (b, eR[("A", "train")], eR[("A", "new")], eR[("B", "train")], eR[("B", "new")], CR[b],
                 eV[("A", "train")], eV[("A", "new")], eV[("B", "train")], eV[("B", "new")], eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")], CV[b],
                 eP[("A", "train")], eP[("B", "train")], eP[("C", "train")], eP[("C", "new")], eP[("D", "train")], eP[("D", "new")], rA, rB,
                 K[b][2] if K[b][2] != "na" else ("기준 미달" if not K[b][0] else "전제 미달"),
                 eR[("C", "train")], eR[("D", "train")], RT["R"][b]["ratio"], RT["V"][b]["ratio"], RT["P"][b]["ratio"],
                 RT["R"][b]["mda"], RT["V"][b]["mda"], RT["P"][b]["mda"], MN[b], utxt))
    print("F-GW: 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 실패 %d/5" % tuple(sum(cat[b] == c for b in BRAINS) for c in ("conc", "memo", "part", "fail")))
    sl4, sr, sv, sp = status_l4(cat), J5.status_rule(CR), J5.status_rev(CV), J5.status_keep(K)
    print("L4: F-GW 맥락 의존 범주 %d/5 → %s" % (sum(cat[b] == "conc" for b in BRAINS), sl4))
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
