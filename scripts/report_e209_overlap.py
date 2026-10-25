#!/usr/bin/env python3
"""E209 보고(판정 밖 — 사전등록 4절 '보고'): 구획 되먹임 억제 구조(F-G0C)에서 잰 형성 표현 F 의 범주 KC 겹침 — E208 보고 측정 커맨드에 --kc-inh-compart 만 더함.
뇌마다 맥락 X 켬(meas_C_X)·맥락 없음(meas_C_none) 두 측정. 주 지표 = 기능적 겹침 코사인(posthoc_e208_graded.brain_stats — 둘째 측정 post, AC·BD 두 쪽 네 값 평균; 통합 병목 정리 8-1절과 같은 정의).
같은 뇌의 짝: E207 F·맥락 없음(공유 억제 — cos_none), E208 F·흥분 0·맥락 X(공유 억제 — cos_G0).
남은 몫 q = (cos_C − cos_none) / (cos_G0 − cos_none) 를 1e-4 정수로 반올림해 읽기 규칙(결과 전): q ≤ 3333 '겹침 사라짐(맥락 없음 쪽)', q ≥ 6667 '겹침 남음', 그 밖 '가운데'. 분모 ≤ 0 이면 '판단 불가'.
함께 적음: 집합 S 첫/둘째(J7.score), 사례당 반응 KC(post·train 48사례 평균 — C·X / C·none 비), 억제 뉴런 무리별 발화.
점검(어기면 그 뇌 '측정 무효'): C·X — '[E207 측정 맥락]' pre·post(X·제시 288·켠 > 0·끈 0·게이팅 있음·활성 > 0·비활성 × 100 ≤ 같은 뇌 E207 F·X 같은 tag 활성 — E208 정정 1),
  '[E209 구획 억제 측정]' pre·post(X·288·활성 > 0·비활성 × 100 ≤ 활성), RCJ 12·노출 0·맥락 무게 0, 구획 장치 확인 줄 ≥ 1(4집단);
  C·none — RCJ 12·노출 0·'[E207 측정 맥락]' 없음·구획 장치 확인 줄 ≥ 1; 두 측정의 구획 연결 수가 같음; 자기 점검(집합 J 의 '%.4f' 문자열 = RCJ J_post 문자열 — self_check) 네 로그 모두.
입력: logs/E209/meas_C_{X|none}_b{B}.log, logs/E207/meas_F_{none|X}_b{B}.log, logs/E208/meas_F_X_b{B}.log. 실행: python3 scripts/report_e209_overlap.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e207 as J7  # noqa: E402 — read_meas·score
import posthoc_e208_graded as P  # noqa: E402 — brain_stats(합성 14/14)

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
Q_LO, Q_HI = 3333, 6667
RX_W = re.compile(r"\[맥락 입력(?: 2)?\] 맥락 전용 집단 (zz_ctx2?) \d+ → KC 좌·우, w=(-?[0-9.]+),")
RX_DEV = re.compile(r"\[E209 구획 되먹임 억제 확인\] 장치 — (.*)$")
RX_DEV1 = re.compile(r"(kc_(?:inh_)?[lr]_to_(?:inh|kc)_[lr]) (\d+)\(같음·무리 밖 0·g (-?[0-9.]+)\)")
RX_IM = re.compile(r"\[E209 구획 억제 측정\] tag=(\w+) 맥락 (\w) 켬 — 제시 (\d+) · KC 억제 뉴런 발화 활성 무리 (\d+) · 비활성 무리 (\d+)")
GROUPS = ("kc_l_to_inh_l", "kc_inh_l_to_kc_l", "kc_r_to_inh_r", "kc_inh_r_to_kc_r")


def ctx_weights(path):
    return [(m.group(1), float(m.group(2))) for m in (RX_W.search(ln) for ln in open(path, encoding="utf-8", errors="replace")) if m]


def read_compart(path):
    """→ {"dev": [{집단: (연결 수, g)}, …](확인 줄마다), "im": {tag: (맥락, 제시, 활성, 비활성)}}."""
    dev, im = [], {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RX_DEV.search(ln)
        if m:
            dev.append({g.group(1): (int(g.group(2)), float(g.group(3))) for g in RX_DEV1.finditer(m.group(1))})
            continue
        m = RX_IM.search(ln)
        if m:
            im[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    return {"dev": dev, "im": im}


def dev_ok(c):
    """구획 장치 확인 줄 ≥ 1, 줄마다 4집단·g = 10/−30, 모든 줄의 연결 수가 같음 → (통과, 연결 수 사전)."""
    if not c["dev"]:
        return False, None
    first = c["dev"][0]
    good = all(set(d) == set(GROUPS) for d in c["dev"]) and all(d == first for d in c["dev"])
    good = good and all(first[g][1] == (10.0 if "_to_inh_" in g else -30.0) and first[g][0] > 0 for g in GROUPS)
    return good, ({g: first[g][0] for g in GROUPS} if good else None)


def valid_x(path, r, ref, c):
    cx, rc = r["ctx"], ref["ctx"]
    w = ctx_weights(path)
    ok = (set(cx) == {"pre", "post"} and set(rc) == {"pre", "post"}
          and all(v and v[0] == "X" and v[1] == 288 and v[2] > 0 and v[3] == 0 and v[6] == "있음" and v[4] > 0 and rc[t] and 100 * v[5] <= rc[t][4]
                  for t, v in cx.items())
          and len(r["rcj"]) == 12 and r["expo"] == 0 and {p for p, _ in w} == {"zz_ctx", "zz_ctx2"} and all(x == 0.0 for _, x in w))
    im = c["im"]
    ok = ok and set(im) == {"pre", "post"} and all(v[0] == "X" and v[1] == 288 and v[2] > 0 and 100 * v[3] <= v[2] for v in im.values())
    return ok and dev_ok(c)[0]


def valid_none(path, r, c):
    w = ctx_weights(path)
    return (len(r["rcj"]) == 12 and r["expo"] == 0 and not r["ctx"] and not c["im"]
            and {p for p, _ in w} == {"zz_ctx", "zz_ctx2"} and all(x == 0.0 for _, x in w) and dev_ok(c)[0])


def q4(cc, cn, cg):
    """남은 몫 q 를 1e-4 정수로(분모 ≤ 0 이면 None)."""
    den = cg - cn
    if den <= 0:
        return None
    return int(round((cc - cn) / den * 1e4))


def classify(q):
    if q is None:
        return "판단 불가"
    if q <= Q_LO:
        return "겹침 사라짐"
    if q >= Q_HI:
        return "겹침 남음"
    return "가운데"


RX_JS = re.compile(r"^RCJ side=([lr]) pair=([A-D]{2}) J_pre=[0-9.]+ J_post=([0-9.]+)")


def self_check(path):
    """집합 J(p ≥ 0.5 — posthoc_e208_graded 와 같은 정의)를 과제 출력과 같은 '%.4f' 문자열로 만들어 로그 RCJ J_post 문자열과 비교(AC·BD·AB, 두 쪽).
    posthoc_e208_graded.brain_stats 의 self_check 는 1e-4 반올림 정수 비교라 정확히 .5 경계(J = 0.04375 등)에서 거짓 불일치를 낸다(E209 개발 진단 3 — 뇌 15 맥락 없음 l AC)."""
    cases, _ = P.read(path)
    pv = {k: P.pvec(v) for k, v in cases.items()}
    js = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = RX_JS.match(ln)
        if m:
            js[(m.group(1), m.group(2))] = m.group(3)
    return all(js.get((sd, pr)) == "%.4f" % P.metrics(pv[(sd, pr[0])], pv[(sd, pr[1])])[0] for sd in "lr" for pr in ("AC", "BD", "AB"))


def per_case(path):
    """RCKC tag=post set=train 사례당 반응 KC 수 평균(48사례 — 쪽 2 × 범주 4 × 사례 6)."""
    n = [len(s) for sd in P.read(path)[0].values() for s in sd]
    if len(n) != 48:
        raise ValueError("사례 수 %d" % len(n))
    return sum(n) / len(n)


def main():
    rows, cls = [], {}
    for b in BRAINS:
        pcx = os.path.join(EXP, "logs", "E209", "meas_C_X_b%d.log" % b)
        pcn = os.path.join(EXP, "logs", "E209", "meas_C_none_b%d.log" % b)
        p7n = os.path.join(EXP, "logs", "E207", "meas_F_none_b%d.log" % b)
        p7x = os.path.join(EXP, "logs", "E207", "meas_F_X_b%d.log" % b)
        p8x = os.path.join(EXP, "logs", "E208", "meas_F_X_b%d.log" % b)
        try:
            rcx, rcn, r7x = J7.read_meas(pcx), J7.read_meas(pcn), J7.read_meas(p7x)
            ccx, ccn = read_compart(pcx), read_compart(pcn)
            same_conn = dev_ok(ccx)[1] is not None and dev_ok(ccx)[1] == dev_ok(ccn)[1]
            if not (valid_x(pcx, rcx, r7x, ccx) and valid_none(pcn, rcn, ccn) and same_conn):
                cls[b] = "측정 무효"
                rows.append("b%d: 측정 무효(점검 — C·X 맥락 줄 %s · 억제 측정 %s · 장치 %s · C·none 맥락 줄 %s · 장치 %s · 연결 수 같음 %s · RCJ %d/%d · 노출 %s/%s)"
                            % (b, rcx["ctx"], ccx["im"], dev_ok(ccx), rcn["ctx"], dev_ok(ccn), same_conn, len(rcx["rcj"]), len(rcn["rcj"]), rcx["expo"], rcn["expo"]))
                continue
            st = {k: P.brain_stats(p) for k, p in (("cx", pcx), ("cn", pcn), ("n7", p7n), ("g8", p8x))}
            sc = {k: self_check(p) for k, p in (("cx", pcx), ("cn", pcn), ("n7", p7n), ("g8", p8x))}
            if not all(sc.values()):
                cls[b] = "측정 무효"
                rows.append("b%d: 측정 무효(자기 점검 %s)" % (b, sc))
                continue
            q = q4(st["cx"]["cos"], st["n7"]["cos"], st["g8"]["cos"])
            cls[b] = classify(q)
            kc = {k: per_case(p) for k, p in (("cx", pcx), ("cn", pcn), ("n7", p7n), ("g8", p8x))}
            im = ccx["im"]
            rows.append("b%d: 구획·X 코사인 %.3f → q %s → %s | 짝 E207 맥락 없음 %.3f · E208 흥분 0·X %.3f · 구획·맥락 없음 %.3f | 집합 S 첫/둘째 구획·X %d/%d · 구획·없음 %d/%d · E208 %d/%d"
                        " | 사례당 반응 KC 구획·X %.1f · 구획·없음 %.1f(비 %.2f) · E208 X %.1f · E207 없음 %.1f | 억제 뉴런 발화(X) 활성 %d/%d · 비활성 %d/%d"
                        % (b, st["cx"]["cos"], ("%.4f" % (q / 1e4)) if q is not None else "—", cls[b], st["n7"]["cos"], st["g8"]["cos"], st["cn"]["cos"],
                           J7.score(rcx, 0), J7.score(rcx, 1), J7.score(rcn, 0), J7.score(rcn, 1), J7.score(J7.read_meas(p8x), 0), J7.score(J7.read_meas(p8x), 1),
                           kc["cx"], kc["cn"], kc["cx"] / kc["cn"], kc["g8"], kc["n7"], im["pre"][2], im["post"][2], im["pre"][3], im["post"][3]))
        except (OSError, KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as ex:
            cls[b] = "결측"
            rows.append("b%d: 결측(%s)" % (b, type(ex).__name__))
    print("E209 보고(판정 밖): 구획 되먹임 억제 구조의 기능적 겹침(코사인, AC·BD 두 쪽 평균, 둘째 측정) — 남은 몫 q = (구획·X − E207 맥락 없음) / (E208 흥분 0·X − E207 맥락 없음), "
          "읽기 규칙 q ≤ %.4f 겹침 사라짐 · ≥ %.4f 겹침 남음" % (Q_LO / 1e4, Q_HI / 1e4))
    for ln in rows:
        print(ln)
    print("요약: 겹침 사라짐 %d/5 · 겹침 남음 %d/5 · 가운데 %d/5 · 판단 불가 %d/5 · 측정 무효 %d/5 · 결측 %d/5"
          % tuple(sum(v == k for v in cls.values()) for k in ("겹침 사라짐", "겹침 남음", "가운데", "판단 불가", "측정 무효", "결측")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
