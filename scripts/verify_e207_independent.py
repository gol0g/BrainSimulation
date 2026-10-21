#!/usr/bin/env python3
"""E207 독립 대조 — judge_e207 과 코드를 나누지 않는 다른 경로(정규식·정수 산술 직접)로 같은 판정을 다시 계산한다.
RCJ 줄의 J_pre·J_post 문자열을 정수로 바꿔 S = J(A,C)_좌 + J(B,D)_좌 + J(A,C)_우 + J(B,D)_우 를 두 측정 각각 만들고 criteria_fixed.txt 규칙(정정 1 포함)을 따로 구현한다.
정정 1: 첫 측정(J_pre)·둘째 측정(J_post) 분류가 같을 때만 그 분류, 다르면 가운데. J_pre = J_post 는 조작검증이 아니다.
조작검증 중 핵심만: 맥락 X 측정의 '[E207 측정 맥락]' pre·post(켠 > 0·끈 0·비활성 ≤ 활성/100)·맥락 없음 측정에 그 줄 없음·노출 0·RCJ 12·같은 값(J_pre·J_post 각각).
실행: python3 scripts/verify_e207_independent.py (저장소 루트에서)"""
import os
import re
import sys

EXP = "research/experiments"
BR = [16, 17, 18, 19, 20]
RX_J = re.compile(r"^RCJ side=([lr]) pair=([A-D]{2}) J_pre=(\d+)\.(\d{4}) J_post=(\d+)\.(\d{4})")
RX_C = re.compile(r"^\[E207 측정 맥락\] tag=(pre|post) 맥락 X 켬 — 제시 \d+ · 켠 맥락 발화 (\d+) · 끈 맥락 발화 (\d+) · KC 발화 활성 무리 (\d+) · 비활성 무리 (\d+) · 게이팅 있음$")


def parse(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    J, ctx, any_ctx = ({}, {}), {}, "[E207 측정 맥락]" in t
    for ln in t.splitlines():
        m = RX_J.match(ln)
        if m:
            J[0][(m.group(1), m.group(2))] = int(m.group(3)) * 10000 + int(m.group(4))
            J[1][(m.group(1), m.group(2))] = int(m.group(5)) * 10000 + int(m.group(6))
        m = RX_C.match(ln)
        if m:
            ctx[m.group(1)] = tuple(int(m.group(i)) for i in range(2, 6))
    e = re.search(r"^\[원시 특징 노출\] 제시 (\d+)\(", t, re.M)
    return J, ctx, any_ctx, (int(e.group(1)) if e else None)


def main():
    L = os.path.join(EXP, "logs", "E207")
    bad, S, Jall = [], {}, {}
    try:
        for b in BR:
            for rep in ("U", "F"):
                for c in ("none", "X"):
                    J, ctx, any_ctx, expo = parse(os.path.join(L, "meas_%s_%s_b%d.log" % (rep, c, b)))
                    if len(J[0]) != 12 or len(J[1]) != 12 or expo != 0:
                        bad.append("%s %s b%d RCJ %d/%d·노출 %s" % (rep, c, b, len(J[0]), len(J[1]), expo))
                    if c == "X" and not (set(ctx) == {"pre", "post"} and all(v[0] > 0 and v[1] == 0 and 100 * v[3] <= v[2] and v[2] > 0 for v in ctx.values())):
                        bad.append("%s X b%d 측정 맥락 %s" % (rep, b, ctx))
                    if c == "none" and any_ctx:
                        bad.append("%s none b%d 맥락 줄 있음" % (rep, b))
                    for k in (0, 1):
                        S[(b, rep, c, k)] = J[k][("l", "AC")] + J[k][("l", "BD")] + J[k][("r", "AC")] + J[k][("r", "BD")]
                        Jall[(b, rep, c, k)] = [J[k][(s, p)] for s in "lr" for p in ("AB", "AC", "AD", "BC", "BD", "CD")]
            for rep in ("U", "F"):
                for k in (0, 1):
                    if Jall[(b, rep, "X", k)] == Jall[(b, rep, "none", k)]:
                        bad.append("같은 값 %s b%d %s" % (rep, b, "J_pre" if k == 0 else "J_post"))
    except (OSError, KeyError, ValueError, IndexError) as ex:
        print("독립 판정: 보류(결측 — %s)" % type(ex).__name__)
        return 0
    cls, both = {}, {}
    for b in BR:
        one = []
        for k in (0, 1):
            sx, sn = S[(b, "F", "X", k)], S[(b, "F", "none", k)]
            one.append("되살림" if (sx >= 6000 and sx - sn >= 4000) else ("되살리지 않음" if sx <= 3200 else "가운데"))
        both[b] = one
        cls[b] = one[0] if one[0] == one[1] else "가운데"
    print("독립 F: " + " · ".join("b%d 첫 S_none %d S_X %d %s · 둘째 S_none %d S_X %d %s ⇒ %s" % (b, S[(b, "F", "none", 0)], S[(b, "F", "X", 0)], both[b][0],
                                                                                         S[(b, "F", "none", 1)], S[(b, "F", "X", 1)], both[b][1], cls[b]) for b in BR))
    print("독립 U: " + " · ".join("b%d 첫 S_none %d S_X %d · 둘째 S_none %d S_X %d" % (b, S[(b, "U", "none", 0)], S[(b, "U", "X", 0)], S[(b, "U", "none", 1)], S[(b, "U", "X", 1)]) for b in BR))
    if bad:
        print("독립 조작검증 실패: " + "; ".join(bad))
        print("독립 판정: 보류(조작검증 실패)")
        return 0
    nr = sum(v == "되살림" for v in cls.values()); nn = sum(v == "되살리지 않음" for v in cls.values())
    if nr >= 4:
        print("독립 판정: H128 지지 — 맥락 X 가 형성 표현의 범주 KC 겹침을 되살림")
    elif nn >= 4:
        print("독립 판정: H128-null — 맥락 X 에서도 형성 표현의 범주 KC 겹침 낮음")
    else:
        print("독립 판정: 보류 — 되살림 %d/5·되살리지 않음 %d/5" % (nr, nn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
