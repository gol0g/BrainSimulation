#!/usr/bin/env python3
"""E195 판정 — KC 희소화와 XOR 결합 순도(측정). 규칙은 research/experiments/logs/E195/criteria_fixed.txt(정정 1 — 조작은 KC 되먹임 억제 배율).
입력: logs/E195/meas_b{B}_{S1,SP,SR}.log(rawcat --rc-measure-ctx, 맥락 연결 C2, --kc-inh-scale 1·1.5·2 — 주 1.5·보조 2 는 경로 검사 2 의 사전 규칙), 뇌 10~14.
지표(쪽 평균): 순도 p = 반응 KC(칸 평균 ≥ 0.25)의 활동 가중 평균 XOR 순도(rcxor.purity_rows), 제시당 활성 KC n, 해독 XOR·범주·맥락(rcxor.decode_log).
조작검증: 로그마다 RCKXCHK(npres 288·켠 맥락 > 0·끈 0·비활성 특징 0·g 불변·맥락 w 20·p 0.020·kc_inh_scale = 수준·cfg_kc_inh_w = −15 × 수준·장치 억제 g 평균 inh_g_l·r = −15 × 수준·cfg_kc_target 0.05),
  RCKX 쪽마다 72/72, 맥락 연결 줄(w 20·p 0.020·연결 12,000 ± 15%), 특징 줄(w 24·p 0.10), 종류 검증, 수준 ≠ 1 은 '[E195 KC 되먹임 억제] 배율' 줄; 희소화가 들음 = n(SP) ≤ 0.9·n(S1) — 4/5 미만이면 보류(조작 무효).
뇌 단위(주 = SP 대 S1, 경계는 1e-4 반올림 정수): Δp = p(SP) − p(S1); 유효 = 범주·맥락 해독(SP) ≥ 0.9.
  상승 = 유효·Δp ≥ +0.05·XOR(SP) ≥ 0.75; 결합 소실 = 유효·Δp ≥ +0.05·XOR(SP) < 0.75; 변화 없음 = 유효·|Δp| < 0.02.
종합: 조작검증 실패 → 보류 / 희소화 안 들음 → 보류(조작 무효) / 상승 ≥ 4/5 → 순도 상승(H116) / 결합 소실 ≥ 4/5 / 변화 없음 ≥ 4/5 → 변화 없음(H116-null) / 그 밖 보류.
(보고) SR 같은 지표, S1 과 E192 C2 의 일치(회귀).
실행: python3 scripts/judge_e195.py (저장소 루트에서)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcxor  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
LEVELS = (("S1", 1.0), ("SP", 1.5), ("SR", 2.0))   # 주 1.5·보조 2 — 경로 검사 2(뇌 15)의 사전 규칙 결과(logs/E195/path_check2.out)
N_KC = 3000


def check_log(path, scale):
    why = []
    chk, rckx, ctxl, feat, kt, line195 = None, {"l": {"train": 0, "new": 0}, "r": {"train": 0, "new": 0}}, {}, False, 0, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            s = ln.strip()
            if s.startswith("RCKXCHK "):
                chk = dict(x.split("=", 1) for x in s.split()[1:])
            elif s.startswith("RCKX "):
                kv = dict(x.split("=", 1) for x in s.split(" ids=")[0].split()[1:])
                rckx[kv["side"]][kv["set"]] += 1
            elif s.startswith("[맥락 입력 2] ") or s.startswith("[맥락 입력] "):
                key = "ctx2" if s.startswith("[맥락 입력 2]") else "ctx"
                ctxl[key] = (float(s.split("w=")[1].split(",")[0]), float(s.split("p=")[1].split(",")[0]),
                             int(s.split("연결 좌 ")[1].split("·")[0]), int(s.split("·우 ")[1].split()[0]))
            elif s.startswith("[원시 특징] ") and "w=24.00" in s and "p=0.100" in s:
                feat = True
            elif s.startswith("[E153 종류 입력 적재]") and "검증 일치" in s:
                kt += 1
            elif s.startswith("[E195 KC 되먹임 억제] 배율 "):
                line195 = abs(float(s.split("배율 ")[1].split()[0]) - scale) < 1e-9
    if chk is None:
        return ["RCKXCHK 없음"]
    if (int(chk["npres"]) != 288 or int(chk["ctx_on_spikes"]) <= 0 or int(chk["ctx_off_spikes"]) != 0 or int(chk["feat_inactive_spikes"]) != 0
            or int(chk["g_same"]) != 1):
        why.append("점검 %s" % {k: chk[k] for k in ("npres", "ctx_on_spikes", "ctx_off_spikes", "feat_inactive_spikes", "g_same")})
    if any(abs(float(chk[k]) - 20.0) > 1e-6 for k in ("ctx_w", "ctx2_w")) or any(abs(float(chk[k]) - 0.02) > 1e-9 for k in ("ctx_p", "ctx2_p")):
        why.append("맥락 설정 %s" % {k: chk[k] for k in ("ctx_w", "ctx_p", "ctx2_w", "ctx2_p")})
    w_exp = -15.0 * scale
    if ("kc_inh_scale" not in chk or abs(float(chk["kc_inh_scale"]) - scale) > 1e-9 or abs(float(chk["cfg_kc_inh_w"]) - w_exp) > 1e-6
            or abs(float(chk["inh_g_l"]) - w_exp) > 1e-3 or abs(float(chk["inh_g_r"]) - w_exp) > 1e-3 or abs(float(chk.get("cfg_kc_target", "nan")) - 0.05) > 1e-9):
        why.append("억제 배율 %s" % {k: chk.get(k) for k in ("kc_inh_scale", "cfg_kc_inh_w", "inh_g_l", "inh_g_r", "cfg_kc_target")})
    if rckx != {"l": {"train": 72, "new": 72}, "r": {"train": 72, "new": 72}}:
        why.append("RCKX 수 %s" % rckx)
    exp_n = 200 * N_KC * 0.02
    for key in ("ctx", "ctx2"):
        if key not in ctxl:
            why.append("%s 연결 줄 없음" % key); continue
        ww, pp, nl, nr = ctxl[key]
        if abs(ww - 20.0) > 1e-6 or abs(pp - 0.02) > 1e-9 or not (0.85 * exp_n <= nl <= 1.15 * exp_n and 0.85 * exp_n <= nr <= 1.15 * exp_n):
            why.append("%s 연결 %s" % (key, ctxl[key]))
    if not feat:
        why.append("특징 연결 줄 없음")
    if kt < 1:
        why.append("종류 검증 없음")
    if abs(scale - 1.0) > 1e-9 and not line195:
        why.append("E195 배율 줄 없음")
    return why


def measure(path):
    pr = rcxor.purity_log(path, N_KC)
    d = rcxor.decode_log(path, N_KC)
    if set(pr) != {"l", "r"} or set(d) != {"l", "r"}:
        raise ValueError("쪽 결측")
    return {"p": (pr["l"][0] + pr["r"][0]) / 2.0, "n": (pr["l"][3] + pr["r"][3]) / 2.0, "hi": pr["l"][2] + pr["r"][2],
            "xor": (d["l"]["xor"] + d["r"]["xor"]) / 2.0, "cat": (d["l"]["cat"] + d["r"]["cat"]) / 2.0, "ctx": (d["l"]["ctx"] + d["r"]["ctx"]) / 2.0}


def main():
    ok, why, M = True, [], {}
    for b in BRAINS:
        for lv, t in LEVELS:
            path = os.path.join(EXP, "logs", "E195", "meas_b%d_%s.log" % (b, lv))
            try:
                wy = check_log(path, t)
                M[(b, lv)] = measure(path)
            except (OSError, ValueError, IndexError, KeyError) as ex:
                print("판정: 보류(결측 — b%d %s %s)" % (b, lv, type(ex).__name__))
                return 0
            if wy:
                ok = False; why.append("b%d %s: %s" % (b, lv, "; ".join(wy)))
    if not ok:
        print("조작검증 실패: " + " | ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    q4 = lambda x: int(round(x * 10000))     # 경계 비교는 1e-4 정수로(부동소수 오차 — 합성 시험 0.15 − 0.10 = 0.0499…)
    sparse = sum(q4(M[(b, "SP")]["n"] * 10) <= q4(M[(b, "S1")]["n"] * 9) for b in BRAINS)
    cnt = {"rise": 0, "loss": 0, "flat": 0}
    for b in BRAINS:
        m5, m3, m2 = M[(b, "S1")], M[(b, "SR")], M[(b, "SP")]
        dp = m2["p"] - m5["p"]
        dq = q4(m2["p"]) - q4(m5["p"])
        valid = q4(m2["cat"]) >= 9000 and q4(m2["ctx"]) >= 9000
        cnt["rise"] += valid and dq >= 500 and q4(m2["xor"]) >= 7500
        cnt["loss"] += valid and dq >= 500 and q4(m2["xor"]) < 7500
        cnt["flat"] += valid and abs(dq) < 200
        reg = ""
        e192 = os.path.join(EXP, "logs", "E192", "meas_b%d_C2.log" % b)
        if os.path.exists(e192):
            r = measure(e192)
            reg = " | (보고) E192 C2 대비 S1 순도 차 %+.4f·XOR 차 %+.3f" % (m5["p"] - r["p"], m5["xor"] - r["xor"])
        print("b%d 순도 S1 %.3f · SR(2) %.3f · SP(1.5) %.3f (Δ %+.3f) | 활성 KC %.0f · %.0f · %.0f | 순도 ≥ 0.5 KC %d · %d · %d | XOR %.3f · %.3f · %.3f | 범주·맥락(SP) %.3f·%.3f%s"
              % (b, m5["p"], m3["p"], m2["p"], dp, m5["n"], m3["n"], m2["n"], m5["hi"], m3["hi"], m2["hi"], m5["xor"], m3["xor"], m2["xor"], m2["cat"], m2["ctx"], reg))
    print("조작검증 통과 | 희소화 들음 %d/5 | 상승 %d/5 · 결합 소실 %d/5 · 변화 없음 %d/5" % (sparse, cnt["rise"], cnt["loss"], cnt["flat"]))
    if sparse < 4:
        print("판정: 보류(조작 무효 — 희소화 %d/5)" % sparse)
        return 0
    v = ("순도 상승(H116)" if cnt["rise"] >= 4 else "결합 소실" if cnt["loss"] >= 4 else "변화 없음(H116-null)" if cnt["flat"] >= 4 else "보류")
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
