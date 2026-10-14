#!/usr/bin/env python3
"""E192 판정 — 범주 × 맥락 결합 부호(측정). 규칙은 research/experiments/logs/E192/criteria_fixed.txt.
입력: logs/E192/meas_b{B}_{C0,C1,C2}.log(rawcat --rc-measure-ctx, 맥락 연결 C0 p 0.10·w 4 / C1 p 0.05·w 8 / C2 p 0.02·w 20), 뇌 10~14.
해독: scripts/rcxor.py(능형 회귀, 노출 사례로 맞춤·새 사례로 시험, 쪽 평균) — XOR·범주·맥락 정답률.
뇌·설정 단위: 측정 유효 = 범주 ≥ 0.9 이면서 맥락 ≥ 0.9; XOR 충분 = 유효 이면서 XOR ≥ 0.75; XOR 부족 = 유효 이면서 XOR ≤ 0.60.
종합(우선순위): 조작검증 실패 → 보류 / C0 충분 ≥ 4/5 → 표현 충분(H113-null) / C0 부족 ≥ 4/5 이면서 C1·C2 중 충분 ≥ 4/5 → 설정 의존(H113 — 충분 설정 중 평균 XOR 이 높은 것)
  / 세 설정 모두 부족 ≥ 4/5 → 결합 부호 부족(설정 무관) / 그 밖 보류.
실행: python3 scripts/judge_e192.py (저장소 루트에서)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rcxor  # noqa: E402

EXP = "research/experiments"
BRAINS = (10, 11, 12, 13, 14)
CONFIGS = (("C0", 0.10, 4.0), ("C1", 0.05, 8.0), ("C2", 0.02, 20.0))
NPRES = 288
N_KC = 3000


def check_log(path, p, w):
    why = []
    chk, rckx, ctxl, feat, kt = None, {"l": {"train": 0, "new": 0}, "r": {"train": 0, "new": 0}}, {}, False, 0
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
                ww = float(s.split("w=")[1].split(",")[0]); pp = float(s.split("p=")[1].split(",")[0])
                nl = int(s.split("연결 좌 ")[1].split("·")[0]); nr = int(s.split("·우 ")[1].split()[0])
                ctxl[key] = (ww, pp, nl, nr)
            elif s.startswith("[원시 특징] ") and "w=24.00" in s and "p=0.100" in s:
                feat = True
            elif s.startswith("[E153 종류 입력 적재]") and "검증 일치" in s:
                kt += 1
    if chk is None:
        return ["RCKXCHK 없음"]
    if int(chk["npres"]) != NPRES or int(chk["ctx_on_spikes"]) <= 0 or int(chk["ctx_off_spikes"]) != 0 or int(chk["feat_inactive_spikes"]) != 0 or int(chk["g_same"]) != 1:
        why.append("점검 %s" % {k: chk[k] for k in ("npres", "ctx_on_spikes", "ctx_off_spikes", "feat_inactive_spikes", "g_same")})
    for k in ("ctx_w", "ctx2_w"):
        if abs(float(chk[k]) - w) > 1e-6:
            why.append("%s %s" % (k, chk[k]))
    for k in ("ctx_p", "ctx2_p"):
        if abs(float(chk[k]) - p) > 1e-9:
            why.append("%s %s" % (k, chk[k]))
    if rckx != {"l": {"train": 72, "new": 72}, "r": {"train": 72, "new": 72}}:
        why.append("RCKX 수 %s" % rckx)
    exp_n = 200 * N_KC * p
    for key in ("ctx", "ctx2"):
        if key not in ctxl:
            why.append("%s 연결 줄 없음" % key); continue
        ww, pp, nl, nr = ctxl[key]
        if abs(ww - w) > 1e-6 or abs(pp - p) > 1e-9 or not (0.85 * exp_n <= nl <= 1.15 * exp_n and 0.85 * exp_n <= nr <= 1.15 * exp_n):
            why.append("%s 연결 %s" % (key, ctxl[key]))
    if not feat:
        why.append("특징 연결 줄(w 24·p 0.10) 없음")
    if kt < 1:
        why.append("종류 검증 없음")
    return why


def main():
    ok, why, res = True, [], {}
    for b in BRAINS:
        for c, p, w in CONFIGS:
            path = os.path.join(EXP, "logs", "E192", "meas_b%d_%s.log" % (b, c))
            try:
                wy = check_log(path, p, w)
                d = rcxor.decode_log(path, N_KC)
            except (OSError, ValueError, IndexError, KeyError) as ex:
                print("판정: 보류(결측 — b%d %s %s)" % (b, c, type(ex).__name__))
                return 0
            if wy or set(d) != {"l", "r"}:
                ok = False; why.append("b%d %s: %s" % (b, c, "; ".join(wy) if wy else "쪽 %s" % sorted(d)))
                continue
            m = {k: (d["l"][k] + d["r"][k]) / 2.0 for k in ("xor", "cat", "ctx")}
            m["prev"] = (d["l"]["prev"] + d["r"]["prev"]) / 2.0 if "prev" in d["l"] and "prev" in d["r"] else float("nan")
            m["conj"] = d["l"]["conj"] + d["r"]["conj"]
            res[(b, c)] = m
    if not ok:
        print("조작검증 실패: " + " | ".join(why))
        print("판정: 보류(조작검증 실패)")
        return 0
    cnt = {}
    for c, _, _ in CONFIGS:
        n_valid = n_suf = n_ins = 0
        for b in BRAINS:
            m = res[(b, c)]
            valid = m["cat"] >= 0.9 and m["ctx"] >= 0.9
            n_valid += valid
            n_suf += valid and m["xor"] >= 0.75
            n_ins += valid and m["xor"] <= 0.60
            print("b%d %s XOR %.3f 범주 %.3f 맥락 %.3f 결합 KC %d · (보고) 직전 범주 %.3f%s" % (b, c, m["xor"], m["cat"], m["ctx"], m["conj"], m["prev"], "" if valid else " (측정 무효)"))
        cnt[c] = (n_valid, n_suf, n_ins, sum(res[(b, c)]["xor"] for b in BRAINS) / len(BRAINS))
    print("조작검증 통과 | " + " | ".join("%s 유효 %d·충분 %d·부족 %d·평균 XOR %.3f" % (c, v[0], v[1], v[2], v[3]) for c, v in cnt.items()))
    if cnt["C0"][1] >= 4:
        v = "표현 충분(H113-null)"
    elif cnt["C0"][2] >= 4 and any(cnt[c][1] >= 4 for c in ("C1", "C2")):
        best = max((c for c in ("C1", "C2") if cnt[c][1] >= 4), key=lambda c: cnt[c][3])
        v = "설정 의존(H113) — %s" % best
    elif all(cnt[c][2] >= 4 for c in ("C0", "C1", "C2")):
        v = "결합 부호 부족(설정 무관)"
    else:
        v = "보류"
    print("판정: %s" % v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
