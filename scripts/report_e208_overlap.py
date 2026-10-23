#!/usr/bin/env python3
"""E208 보고(판정 밖 — 사전등록 4절 '보고'): 흥분 0 구조(F-G0)에서 맥락 X 를 켜고 잰 형성 표현 F 의 범주 KC 겹침 — E207 과 같은 측정 커맨드에서 흥분 무게만 0.
뇌마다 S(두 쪽 J(A,C)+J(B,D), 1e-4, 첫/둘째 측정)를 같은 뇌 E207 F·맥락 없음·F·맥락 X(흥분 20)와 나란히 적고, 사전 읽기 규칙으로 분류한다:
  공유 사라짐 = 두 측정 모두 S_G0 ≤ 3200(E207 '되살리지 않음' 문턱), 공유 남음 = 두 측정 모두 S_G0 ≥ 6000(E207 '되살림' 문턱의 S 부분), 그 밖 가운데.
점검(어기면 그 뇌 '측정 무효'): '[E207 측정 맥락]' tag pre·post — 맥락 X·제시 288·켠 맥락 발화 > 0·끈 0·게이팅 있음·비활성 무리 발화 ≤ 같은 뇌 E207 F·X(흥분 20) 같은 tag 활성 무리 발화의 1%(정정 1), RCJ 12줄, 노출 0,
  설정 줄 '[맥락 입력]'·'[맥락 입력 2]' w=0.00.
입력: logs/E208/meas_F_X_b{B}.log, logs/E207/meas_F_{none|X}_b{B}.log. 실행: python3 scripts/report_e208_overlap.py (저장소 루트에서)"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge_e207 as J7  # noqa: E402 — read_meas·score

EXP = "research/experiments"
BRAINS = (16, 17, 18, 19, 20)
LO, HI = 3200, 6000
RX_W = re.compile(r"\[맥락 입력(?: 2)?\] 맥락 전용 집단 (zz_ctx2?) \d+ → KC 좌·우, w=(-?[0-9.]+),")


def ctx_weights(path):
    return [(m.group(1), float(m.group(2))) for m in (RX_W.search(ln) for ln in open(path, encoding="utf-8", errors="replace")) if m]


def valid(path, r, ref):
    """ref = 같은 뇌 E207 F·X(흥분 20) read_meas — 게이팅 점검의 기준 활성 발화(정정 1: 흥분 0 은 활성 무리 발화를 줄여 '비' 의 분모가 조작으로 바뀜)."""
    cx, rc = r["ctx"], ref["ctx"]
    w = ctx_weights(path)
    return (set(cx) == {"pre", "post"} and set(rc) == {"pre", "post"}
            and all(v and v[0] == "X" and v[1] == 288 and v[2] > 0 and v[3] == 0 and v[6] == "있음" and v[4] > 0 and rc[t] and 100 * v[5] <= rc[t][4]
                    for t, v in cx.items())
            and len(r["rcj"]) == 12 and r["expo"] == 0 and {p for p, _ in w} == {"zz_ctx", "zz_ctx2"} and all(x == 0.0 for _, x in w))


def classify(s0, s1):
    if s0 <= LO and s1 <= LO:
        return "공유 사라짐"
    if s0 >= HI and s1 >= HI:
        return "공유 남음"
    return "가운데"


def main():
    rows, cls = [], {}
    for b in BRAINS:
        p8 = os.path.join(EXP, "logs", "E208", "meas_F_X_b%d.log" % b)
        try:
            r8 = J7.read_meas(p8)
            rn = J7.read_meas(os.path.join(EXP, "logs", "E207", "meas_F_none_b%d.log" % b))
            rx = J7.read_meas(os.path.join(EXP, "logs", "E207", "meas_F_X_b%d.log" % b))
            if not valid(p8, r8, rx):
                cls[b] = "측정 무효"
                rows.append("b%d: 측정 무효(점검 — 맥락 줄 %s · RCJ %d · 노출 %s · 맥락 무게 %s)" % (b, r8["ctx"], len(r8["rcj"]), r8["expo"], ctx_weights(p8)))
                continue
            s = [J7.score(r8, k) for k in (0, 1)]
            cls[b] = classify(*s)
            act = [r8["ctx"][t][4] for t in ("pre", "post")]; ina = [r8["ctx"][t][5] for t in ("pre", "post")]
            rows.append("b%d: 흥분 0 맥락 X S 첫/둘째 %d/%d → %s | 대조 E207 맥락 없음 %d/%d · 흥분 20 맥락 X %d/%d | J(A,B) 좌/우 첫 %d/%d · E207 흥분 20 %d/%d | KC 발화 활성 무리 %d/%d · 비활성 %d/%d"
                        % (b, s[0], s[1], cls[b], J7.score(rn, 0), J7.score(rn, 1), J7.score(rx, 0), J7.score(rx, 1),
                           r8["rcj"][("l", "AB")][0], r8["rcj"][("r", "AB")][0], rx["rcj"][("l", "AB")][0], rx["rcj"][("r", "AB")][0], act[0], act[1], ina[0], ina[1]))
        except (OSError, KeyError, ValueError, TypeError, IndexError) as ex:
            cls[b] = "결측"
            rows.append("b%d: 결측(%s)" % (b, type(ex).__name__))
    print("E208 보고(판정 밖): 흥분 0 구조의 맥락 X 범주 KC 겹침 S(두 쪽 J(A,C)+J(B,D), 1e-4) — 읽기 규칙 S ≤ %d 두 측정 = 공유 사라짐, ≥ %d = 공유 남음" % (LO, HI))
    for ln in rows:
        print(ln)
    print("요약: 공유 사라짐 %d/5 · 공유 남음 %d/5 · 가운데 %d/5 · 측정 무효 %d/5 · 결측 %d/5" % tuple(sum(v == k for v in cls.values()) for k in ("공유 사라짐", "공유 남음", "가운데", "측정 무효", "결측")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
