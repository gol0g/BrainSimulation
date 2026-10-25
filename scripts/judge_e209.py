#!/usr/bin/env python3
"""E209 판정 — 구획 되먹임 억제 재배선(F-G0R):
E208 F-G0 고정 체계(형성 표현 F·흥분 0·게이팅·경쟁형 η 2.4·도파민 기준선·흔적 동결, 앞 능력 맥락 X 고정)에서 KC 되먹임 억제 고리만 쪽마다 하나 → 게이팅 무리마다 하나로
(--kc-inh-rewire — KC 억제 뉴런을 무리마다 반반, 무리 밖 연결의 앞 뉴런을 같은 무리 안으로 — 뒤 뉴런마다 받는 연결 수·가중치 그대로), 동작점은 결과 전 규칙으로 다시 맞춘 T_C.
표현 F_R = 재배선 구조에서 E206 형성 절차로 다시 형성(traces/E209/feat_formed_b{B} — 개발 진단 5·6: 공유 고리에서 형성한 F 는 재배선 구조에서 범주 선택성이 깨짐).
같은 뇌 16~20·세계 38300·같은 시행 시드, 표현 F = traces/E206/feat_formed_b{B}(형성 다시 안 함). 짝 = E208 F-G0(같은 뇌 — judge_e208 의 읽기·검사 함수 그대로).
능력 정의는 E205~E208 과 같다(judge_e205 의 cat_rule·cat_rev·cat_keep·status_*, L4 범주 judge_e204 category·weakest).
입력: logs/E209/form_b{B}.log(F_R 형성 — 구획 기준 연결 수도 여기서), train_{GC|R|V|P}_b{B}.log, ev_GC_b{B}_{learn|none}_{X|Y}.log, ev_b{B}_{R_learn|V_learn|P_learn|none}.log,
  logs/E209/calib_tonic.txt('T_C=…'), traces/E209/w_{GC|R|V|P}_b{B}.npz; 짝 logs/E208·traces/E208.
조작검증(하나라도 어기면 보류): 형성 — E191 form_ok(제시 816·노출 A,B,C,D·RCW 2·RCJ 12·RCCHK pre·pre2·post·RAWCAT 1) + 세계 줄 38300·원형 A + AC·BD J_post ≤ 0.1000 + 저장 파일 + 재배선 줄;
  E208 과 같은 검사(F_R 적재(그 뇌 traces/E209/feat_formed)·동작점 = T_C·학습 줄·특징/맥락 점검·시드·맥락 고정·반전/둘째 쌍 줄·가중치·평가·맥락 흥분 무게 0·연결 수 = E206) +
  구획 되먹임 억제(재배선) — 모든 학습·평가 로그에 '[E209 구획 되먹임 억제 — 재배선]' 구성 줄과 '[E209 구획 되먹임 억제 확인] 장치' 줄이 같은 수(학습 ≥ 2·평가 ≥ 1)로 있고,
  구성 줄마다 억제 뉴런 무리 X 좌·우 200 / 400·4집단(옮김 0 < k < 전체)·가중치 그대로 5.00·−15.00, 장치 줄 연결 수 = 구성 줄의 전체 수·g 같음·재배선 꼬리, 모든 줄이 그 뇌의 형성 로그 값과 같음;
  F-G0R L4 가중치는 E208 F-G0 가중치와 다름; 같은 값 — 어느 뇌든 F-G0R learn 8값이 E208 F-G0 learn 과 모두 같으면 조작 무효 의심.
  게이팅 문턱(criteria_fixed.txt — 결과 전): 학습 로그 게이팅 요약 비 ≤ 0.10(E205~E208 은 0.01 — 구획은 Y 무리를 X 활동이 억제하던 길을 끊어 새는 몫이 설계상 늘어남).
판정: L4 = F-G0R 맥락 의존 범주 ≥ 4/5 보존·실패 ≥ 4/5 잃음·그 밖 부분, 앞 능력은 E205 와 같은 상태 함수. 종합: 조작검증 실패 → 보류 / 넷 다 보존 → 통합 관문 통과(개발 비교, H130 지지) /
  그 밖 → 통합 관문 미통과(능력별) — 가설 줄: 유지 잃음 H130-null / 유지 보존·L4 아님 H130-맞교환(유지 ↔ L4) / 유지·L4 보존·범주 규칙 아님 H130 부분 / 유지·L4·범주 규칙 보존·반전 아님 '맞교환(유지 ↔ 반전)' / 유지 부분·보류 'H130 부분·보류'.
실행: python3 scripts/judge_e209.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge_e193 import read_train  # noqa: E402
import judge_e204 as J4  # noqa: E402 — category·weakest·cells
import judge_e205 as J5  # noqa: E402 — check_common·read_rule·motor_mean·weights_ok·cat_*·status_*
import judge_e206 as J6  # noqa: E402 — check_f_lines·status_l4
import judge_e208 as J8  # noqa: E402 — check_ctxw·counts6·check_eval_g0, 짝(E208 F-G0) 읽기
import report_e209_overlap as R9  # noqa: E402 — read_compart(장치 줄)·GROUPS
import judge_e191 as J1  # noqa: E402 — read_form·form_ok(형성 확인)
from judge_e193 import read_eval  # noqa: E402

EXP = "research/experiments"
BRAINS = J5.BRAINS
WORLD, TRIAL0, CTX0 = J5.WORLD, J5.TRIAL0, J5.CTX0
NL4 = J5.NL4
TASKS, BRANGE, EVALS = J5.TASKS, J5.BRANGE, J5.EVALS
NIH, NIHX = 400, 200
PROTO_A = J5.PROTO_A
JMAX = 1000          # 형성 AC·BD J_post ≤ 0.1000(1e-4) — E206 과 같음
FF9 = "traces/E209/feat_formed_b%d.npz"
W_KI, W_IK = 5.0, -15.0   # 재배선 판 — 가중치 그대로
RMAX9 = 0.10             # 게이팅 요약 비 문턱(criteria_fixed.txt)
RX_CB = re.compile(r"\[E209 구획 되먹임 억제 — 재배선\] KC 억제 뉴런 무리 X 좌 (\d+)·우 (\d+) / (\d+), 무리 밖 연결의 앞 뉴런을 같은 무리 안으로\(뒤 뉴런마다 받는 연결 수 그대로\) — (.*), 가중치 그대로 KC→억제 (-?[0-9.]+)·억제→KC (-?[0-9.]+)$")
RX_CB1 = re.compile(r"(kc_(?:inh_)?[lr]_to_(?:inh|kc)_[lr]) 옮김 (\d+)/(\d+)")


def read_calib_c(path):
    """calib_tonic.txt 'T_C=1100 …' 또는 'T_C=-400 …' → 정수."""
    m = re.search(r"T_C=(-?\d+)\b", open(path, encoding="utf-8").read())
    if not m:
        raise ValueError("calib_tonic.txt 에 T_C 없음")
    return int(m.group(1))


def read_cbuild(path):
    """구성 줄 → [(무리 X 좌, 우, 억제 뉴런 수, {집단: (옮긴 수, 전체 수)}, w_ki, w_ik), …]."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RX_CB.search(ln.rstrip("\n"))
        if m:
            g = {x.group(1): (int(x.group(2)), int(x.group(3))) for x in RX_CB1.finditer(m.group(4))}
            out.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), g, float(m.group(5)), float(m.group(6))))
    return out


def compart_of(path):
    """한 로그의 구획 연결 {집단: (옮긴 수, 전체 수)}(구성·장치 줄이 모두 같을 때) 또는 None, 그리고 줄 수."""
    cb, dv = read_cbuild(path), R9.read_compart(path)["dev"]
    if not cb or len(cb) != len(dv):
        return None, len(cb), len(dv)
    keep = {g: cb[0][3][g] for g in R9.GROUPS} if set(cb[0][3]) == set(R9.GROUPS) else None
    good = keep is not None
    for c in cb:
        good = good and c[0] == NIHX and c[1] == NIHX and c[2] == NIH and c[4] == W_KI and c[5] == W_IK and c[3] == cb[0][3]
        good = good and all(0 < c[3][g][0] < c[3][g][1] for g in R9.GROUPS)
    for d in dv:
        good = good and set(d) == set(R9.GROUPS) | {"_tail"} and d["_tail"] and all(d[g] == (keep[g][1], W_KI if "_to_inh_" in g else W_IK) for g in R9.GROUPS)
    return (keep if good else None), len(cb), len(dv)


def check_compart(tag, path, nmin, want, why):
    keep, nc, nd = compart_of(path)
    good = keep is not None and nc >= nmin and keep == want
    if not good:
        why.append("%s 구획 억제 줄(구성 %d·장치 %d·연결 %s·기준 %s)" % (tag, nc, nd, keep, want))
    return good


def check_f_lines_r(tag, path, b, tc, why):
    """F_R 적재 줄(그 뇌 traces/E209/feat_formed·검증 일치) ≥ 1, 동작점 줄 전부 T_C ≥ 1."""
    ok = True
    fl = J6.read_feat(path)
    want = FF9 % b
    if not fl or not all(p is not None and p.replace("\\", "/").endswith(want) for p in fl):
        ok = False; why.append("%s F_R 적재 줄 %s" % (tag, fl))
    tl = J6.read_tonic(path)
    if not tl or not all(t == float(tc) for t in tl):
        ok = False; why.append("%s 동작점 줄 %s(T_C %d)" % (tag, tl, tc))
    return ok


def check_eval_r(tag, path, cats, ctx, pushed_want, b, tc, want, cw, why):
    """평가 한 개 — judge_e206.check_eval_f 와 같되 F_R 경로·맥락 흥분 무게 0·재배선 줄. 반환 (ok, rc)."""
    ok = True
    rc, pushed, kt, fw, sat, bad = read_eval(path, ctx)
    wantc = {(c, s) for c in cats for s in ("train", "new")}
    if set(rc) != wantc or bad or sat or kt < 1 or fw < 1 or pushed != pushed_want:
        ok = False; why.append("%s 평가(줄 %d·나쁨 %s·포화 %s·종류 %d·특징 %d·pushed %s)" % (tag, len(rc), bad, sat, kt, fw, pushed))
    pw = J5.read_eval_proto(path)
    if pw != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 평가 세계·원형 %s" % (tag, pw))
    if J4.read_gate(path)["build"] < 1:
        ok = False; why.append("%s 평가 게이팅 구성 줄 없음" % tag)
    ok = check_f_lines_r(tag, path, b, tc, why) and ok
    ok = J8.check_ctxw(tag, path, want, why) and ok
    ok = check_compart(tag, path, 1, cw, why) and ok
    return ok, rc


def check_form_r(b, cw, why):
    """F_R 형성 로그 — E191 form_ok·세계·AC·BD J_post ≤ 0.1000·저장 파일·재배선 줄. 반환 (ok, {J_post, J_pre})."""
    path = os.path.join(EXP, "logs", "E209", "form_b%d.log" % b)
    tag = "형성 b%d" % b
    fm = J1.read_form(path)
    world, _ = J6.read_form_extra(path)
    ok = True
    if not J1.form_ok(fm):
        ok = False; why.append("%s E191 형성 검사(제시 %s·범주 %s·RCW %d·RCJ %d·RAWCAT %d·RCCHK %d)"
                               % (tag, fm["expo"], fm["expo_cats"], len(fm["rcw"]), len(fm["rcj"]), fm["rawcat"], len(fm["chk"])))
    if world != [(WORLD, PROTO_A)]:
        ok = False; why.append("%s 세계 줄 %s" % (tag, world))
    jp = {(s, p): fm["rcj"].get((s, p), (None, None)) for s in "lr" for p in ("AC", "BD")}
    if not all(v[1] is not None and v[1] <= JMAX for v in jp.values()):
        ok = False; why.append("%s 겹침 J_post %s(문턱 %d)" % (tag, jp, JMAX))
    fp = os.path.join(EXP, FF9 % b)
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        ok = False; why.append("%s 저장 파일 없음" % tag)
    ok = check_compart(tag, path, 1, cw, why) and ok
    return ok, jp


def check_l4_c(b, tc, want, cw, why):
    L = os.path.join(EXP, "logs", "E209")
    tag = "F-G0R b%d" % b
    tp = os.path.join(L, "train_GC_b%d.log" % b)
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
    ok = check_f_lines_r(tag, tp, b, tc, why) and ok
    ok = J8.check_ctxw(tag, tp, want, why) and ok
    ok = check_compart(tag, tp, 2, cw, why) and ok
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E209", "w_GC_b%d.npz" % b), b,
                             os.path.join(EXP, "traces", "E208", "w_GW_b%d.npz" % b), why)
    ok = ok and w_ok
    ev, good = {}, True
    for w in ("learn", "none"):
        for x in "XY":
            ep = os.path.join(L, "ev_GC_b%d_%s_%s.log" % (b, w, x))
            e_ok, rc = check_eval_r("%s %s %s" % (tag, w, x), ep, "AB", x, 8 if w == "learn" else 0, b, tc, want, cw, why)
            ok = ok and e_ok; good = good and e_ok
            ev[(w, x)] = (rc,)
    a = J4.cells(ev) if good else None
    mn = sum(J5.motor_mean(os.path.join(L, "ev_GC_b%d_none_%s.log" % (b, x))) for x in "XY") / 2
    learn8 = {(x, cs): ev[("learn", x)][0].get(cs) for x in "XY" for cs in [(c_, s_) for c_ in "AB" for s_ in ("train", "new")]}
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb, "motor": mn, "learn8": learn8}
    return ok, a, rep


def check_task_c(t, b, tc, want, cw, why):
    L = os.path.join(EXP, "logs", "E209")
    tag = "G0R-%s b%d" % (t, b)
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
    ok = check_f_lines_r(tag, tp, b, tc, why) and ok
    ok = J8.check_ctxw(tag, tp, want, why) and ok
    ok = check_compart(tag, tp, 2, cw, why) and ok
    other = os.path.join(EXP, "traces", "E209", "w_%s_b%d.npz" % ("V" if t == "R" else "R", b))
    w_ok, fb = J5.weights_ok(tag, os.path.join(EXP, "traces", "E209", "w_%s_b%d.npz" % (t, b)), b, other, why)
    rep = {"ratio": (gs[3] if gs and len(gs) == 4 else None), "mda": (st[1] if st and len(st) == 4 else float("nan")), "fb": fb}
    return ok and w_ok, rep


def task_evals_c(b, tc, want, cw, why):
    L = os.path.join(EXP, "logs", "E209")
    ok, rcs = True, {}
    for nm, _, cats, pw in EVALS:
        ep = os.path.join(L, "ev_b%d_%s.log" % (b, nm))
        e_ok, rc = check_eval_r("G0R b%d %s" % (b, nm), ep, cats, "X", pw, b, tc, want, cw, why)
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
    """종합 판정 문자열과 H130 줄."""
    if sl4 == "보존" and sr == "보존" and sv == "보존" and sp == "보존":
        return "통합 관문 통과(개발 비교)", "H130 지지"
    v = "통합 관문 미통과 — L4 %s · 범주 규칙 %s · 반전 %s · 유지 %s" % (sl4, sr, sv, sp)
    if sp == "잃음":
        h = "H130-null(유지 안 섬)"
    elif sp == "보존":
        if sl4 != "보존":
            h = "H130-맞교환(유지 ↔ L4)"
        elif sr != "보존":
            h = "H130 부분(유지는 섬 — 범주 규칙 %s)" % sr
        else:
            h = "맞교환(유지 ↔ 반전)"
    else:
        h = "H130 부분·보류"
    return v, h


def ratio(num, den):
    return num / den if den else float("nan")


def main():
    J5.EXP = EXP
    J6.EXP = EXP
    J6.J5.EXP = EXP
    J8.EXP = EXP
    J5.RMAX = RMAX9   # 게이팅 요약 비 문턱(criteria_fixed.txt — E209 만)
    ok, why = True, []
    A, RL, RT, E, MN, FJ = {}, {}, {t: {} for t in TASKS}, {}, {}, {}
    G_a, G_e, G_l8, G_mn = {}, {}, {}, {}
    try:
        tc = read_calib_c(os.path.join(EXP, "logs", "E209", "calib_tonic.txt"))
        t8 = J8.read_calib_signed(os.path.join(EXP, "logs", "E208", "calib_tonic.txt"))
        for b in BRAINS:
            want = J8.counts6(b)
            cw, _, _ = compart_of(os.path.join(EXP, "logs", "E209", "form_b%d.log" % b))
            if cw is None:
                ok = False; why.append("b%d 구획 기준(형성 로그) 읽기 실패" % b)
            f_ok, FJ[b] = check_form_r(b, cw, why)
            ok = ok and f_ok
            ok_, a, rep = check_l4_c(b, tc, want, cw, why)
            ok = ok and ok_
            if a is not None:
                A[b] = a
            RL[b] = rep
            for t in TASKS:
                ok_, rep = check_task_c(t, b, tc, want, cw, why)
                ok = ok and ok_
                RT[t][b] = rep
            ok_, e, mn = task_evals_c(b, tc, want, cw, why)
            ok = ok and ok_
            if e is not None:
                E[b] = e
            MN[b] = mn
            # 짝(E208 F-G0, 같은 뇌) — 같은 값 점검·보고. E208 자체 검사는 E208 판정에서 통과(여기서는 값만)
            w8 = []
            _, ga, grep_ = J8.check_l4_g0(b, t8, want, w8)
            _, ge, gmn = J8.task_evals_g0(b, t8, want, w8)
            G_a[b], G_e[b], G_l8[b], G_mn[b] = ga, ge, grep_["learn8"], grep_["motor"]
    except (OSError, ValueError, IndexError, KeyError, TypeError, AttributeError) as ex:
        print("판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    same = [b for b in BRAINS if RL[b]["learn8"] == G_l8[b] and None not in G_l8[b].values()]
    if same:
        ok = False; why.append("같은 값(F-G0R learn 8값 = E208 F-G0) 뇌 %s — 조작 무효 의심" % same)
    if not ok or len(A) != len(BRAINS) or len(E) != len(BRAINS):
        print("조작검증 실패: " + "; ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    print("동작점 T_C %+d (logs/E209/calib_tonic.txt) · 짝 E208 T_F0 %+d" % (tc, t8))
    for b in BRAINS:
        print("형성 b%d F_R 겹침 J_pre/J_post(1e-4) AC 좌 %d/%d 우 %d/%d · BD 좌 %d/%d 우 %d/%d" % (b, *FJ[b][("l", "AC")], *FJ[b][("r", "AC")], *FJ[b][("l", "BD")], *FJ[b][("r", "BD")]))
    cat = {b: J4.category(A[b]) for b in BRAINS}
    m = {b: J4.weakest(A[b]) for b in BRAINS}
    for b in BRAINS:
        r_ = RL[b]
        mg = J4.weakest(G_a[b]) if G_a[b] is not None else None
        print("F-G0R b%d 규칙 방향 효과(1e-4, 노출/새) %s | 범주 %s · 가장 약한 칸 %+d%s | 게이팅 비 %s · 평균 DA %+.4f · 경계 시냅스 %.3f · 무학습 motor 평균 %.4f(E208 F-G0 %.4f)"
              % (b, J5.fmt(A[b]), cat[b], m[b], (" (E208 F-G0 %+d · G0R − G0 %+d)" % (mg, m[b] - mg)) if mg is not None else "",
                 ("%.4f" % r_["ratio"]) if r_["ratio"] is not None else "—", r_["mda"], r_["fb"], r_["motor"], G_mn[b]))
    CR, CV, K = {}, {}, {}
    for b in BRAINS:
        eR, eV, eP = E[b]["R"], E[b]["V"], E[b]["P"]
        CR[b], CV[b], K[b] = J5.cat_rule(eR), J5.cat_rev(eV, eR), J5.cat_keep(eP, eR)
        rA, rB = ratio(eP[("A", "train")], eR[("A", "train")]), ratio(eP[("B", "train")], eR[("B", "train")])
        ge = G_e[b]
        if ge is not None:
            grA, grB = ratio(ge["P"][("A", "train")], ge["R"][("A", "train")]), ratio(ge["P"][("B", "train")], ge["R"][("B", "train")])
            gtxt = (" | (보고) E208 F-G0: R A %+d B %+d · V A %+d B %+d · r_A %.2f r_B %.2f · R 의 C %+d·D %+d → G0R − G0: R A %+d B %+d · V A %+d B %+d · r_A %+.2f r_B %+.2f"
                    % (ge["R"][("A", "train")], ge["R"][("B", "train")], ge["V"][("A", "train")], ge["V"][("B", "train")], grA, grB,
                       ge["R"][("C", "train")], ge["R"][("D", "train")],
                       eR[("A", "train")] - ge["R"][("A", "train")], eR[("B", "train")] - ge["R"][("B", "train")],
                       eV[("A", "train")] - ge["V"][("A", "train")], eV[("B", "train")] - ge["V"][("B", "train")], rA - grA, rB - grB))
        else:
            gtxt = " | (보고) E208 F-G0 짝 없음"
        print("b%d 범주 규칙(R) A %+d/%+d B %+d/%+d → %s | 반전(V) A %+d/%+d B %+d/%+d · Δ A %+d B %+d → %s | 유지(P) A %+d B %+d · C %+d/%+d D %+d/%+d · r_A %.2f r_B %.2f → %s"
              " | (보고) R 의 C %+d·D %+d · 게이팅 비 R %.4f V %.4f P %.4f · 평균 DA R %+.4f V %+.4f P %+.4f · 무학습 motor 평균(맥락 X) %.4f%s"
              % (b, eR[("A", "train")], eR[("A", "new")], eR[("B", "train")], eR[("B", "new")], CR[b],
                 eV[("A", "train")], eV[("A", "new")], eV[("B", "train")], eV[("B", "new")], eV[("A", "train")] - eR[("A", "train")], eV[("B", "train")] - eR[("B", "train")], CV[b],
                 eP[("A", "train")], eP[("B", "train")], eP[("C", "train")], eP[("C", "new")], eP[("D", "train")], eP[("D", "new")], rA, rB,
                 K[b][2] if K[b][2] != "na" else ("기준 미달" if not K[b][0] else "전제 미달"),
                 eR[("C", "train")], eR[("D", "train")], RT["R"][b]["ratio"], RT["V"][b]["ratio"], RT["P"][b]["ratio"],
                 RT["R"][b]["mda"], RT["V"][b]["mda"], RT["P"][b]["mda"], MN[b], gtxt))
    print("F-G0R: 맥락 의존 범주 %d/5 · 사례 한정 %d/5 · 부분 %d/5 · 실패 %d/5" % tuple(sum(cat[b] == c for b in BRAINS) for c in ("conc", "memo", "part", "fail")))
    sl4, sr, sv, sp = J6.status_l4(cat), J5.status_rule(CR), J5.status_rev(CV), J5.status_keep(K)
    print("L4: F-G0R 맥락 의존 범주 %d/5 → %s" % (sum(cat[b] == "conc" for b in BRAINS), sl4))
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
