#!/usr/bin/env python3
"""E209 동작점 규칙(결과 전 — logs/E209/criteria_fixed.txt 와 같은 문장). e208_mpick.py 와 같은 방식, 기준점과 격자만 E209 짝(E208 F-G0, 기저값 +1100)에 맞춤.
경로 검사(뇌 15, 표본 밖)의 무학습 평가 motor 평균으로 기저값 T_C 를 고른다.
m(V) = 맥락 X·Y × A·B × train·new × 두 쪽 × L/R = 32값의 평균(로그 m_{V}_{X|Y}_b15.log 의 RCEVAL 'motor L/R a/b').
m_ref = 짝(E208 F-G0 구조 — 흥분 0·게이팅·공유 되먹임 억제·기저값 +1100) 평균. m_C(T) = 같은 구조 + 구획 되먹임 억제·기저값 T.
  first    : |m_C(+1100) − m_ref| ≤ 0.0170 → T_C = +1100. 아니면 m_C(+1100) < m_ref 이면 NEED_UP, 크면 NEED_DOWN(구동이 줄면 기저값을 올린다).
  up       : 격자 +1100 … +4100(300 간격)에서 |차| 최소(같으면 +1100 에 가까운 쪽). 최소가 위 끝 +4100 이면 NEED_UPWIDE. 아니면 |차| ≤ 0.0170 일 때만 T_C, 넘으면 실패.
  upwide   : 격자 + 4400 … 5600 에서 같은 선택, |차| ≤ 0.0170 일 때만 T_C, 넘으면 실패.
  down     : 격자 +1100, +800 … +200, −100 … −1900(300 간격, 0 을 지나지 않음 — motor_tonic 0 은 '이전 모델' 경로)에서 같은 선택.
             최소가 아래 끝 −1900 이면 NEED_DOWNWIDE. 아니면 |차| ≤ 0.0170 일 때만, 넘으면 실패.
  downwide : 격자 + −2200 … −3400, |차| ≤ 0.0170 일 때만, 넘으면 실패.
T_C 가 정해지면 <cal> 에 'T_C=… m_ref=… m_C=… diff=… (규칙: …)' 한 줄. 실패면 <cal> 을 쓰지 않는다(본실험은 T_C 없이는 시작하지 않음).
사용: python3 e209_mpick.py <로그 디렉터리> <cal 경로> <first|up|upwide|down|downwide>"""
import os
import re
import sys

TH = 170   # 1e-4 단위 정수 — 경계 비교는 로그 정밀도(소수 4자리) 정수로(|차| 0.0170 이면 안)
T0 = 1100
UP = [T0] + list(range(1400, 4101, 300))
UPW = list(range(4400, 5601, 300))
DOWN = [T0] + list(range(800, 199, -300)) + list(range(-100, -1901, -300))
DOWNW = list(range(-2200, -3401, -300))
RX = re.compile(r"motor L/R ([0-9.]+)/([0-9.]+)")


def nm(T):
    return "C_m%d" % -T if T < 0 else "C_p%d" % T


def mean_of(out, V):
    v = []
    for X in "XY":
        for ln in open(os.path.join(out, "m_%s_%s_b15.log" % (V, X)), encoding="utf-8", errors="replace"):
            if ln.startswith("RCEVAL "):
                for a, b in RX.findall(ln):
                    v += [float(a), float(b)]
    if len(v) != 32:
        raise SystemExit("[실패] %s motor 값 수 %d(32 이어야)" % (V, len(v)))
    return sum(v) / len(v)


def d4(m, mr):
    """|m − m_ref| 를 1e-4 단위 정수로(반올림)."""
    return int(round(abs(m - mr) * 10000))


def pick(out, grid, mr):
    cand = []
    for T in grid:
        m = mean_of(out, nm(T))
        cand.append((d4(m, mr), abs(T - T0), T, m))
        print("  후보 %+d: m_C %.4f · |m_C − m_ref| %.4f" % (T, m, d4(m, mr) / 1e4))
    d, _, T, m = min(cand)
    return d, T, m


def main(argv):
    out, cal, mode = argv[1:4]
    mr, m0 = mean_of(out, "ref"), mean_of(out, nm(T0))
    d0 = d4(m0, mr)
    print("  m_ref(공유 억제·%+d) %.4f · m_C(구획 억제·%+d) %.4f · |차| %.4f(규칙 문턱 %.4f)" % (T0, mr, T0, m0, d0 / 1e4, TH / 1e4))
    if mode == "first":
        if d0 <= TH:
            line = "T_C=%d m_ref=%.4f m_C=%.4f diff=%.4f (규칙: |m_C − m_ref| ≤ %.4f → %+d 그대로)" % (T0, mr, m0, d0 / 1e4, TH / 1e4, T0)
            open(cal, "w", encoding="utf-8").write(line + "\n")
            print("[동작점 규칙] " + line)
        else:
            print("NEED_UP" if m0 < mr else "NEED_DOWN")
        return 0
    grids = {"up": UP, "upwide": UP + UPW, "down": DOWN, "downwide": DOWN + DOWNW}
    if mode not in grids:
        raise SystemExit("[실패] 모드 %s" % mode)
    d, T, m = pick(out, grids[mode], mr)
    if mode == "up" and T == UP[-1]:
        print("NEED_UPWIDE"); return 0
    if mode == "down" and T == DOWN[-1]:
        print("NEED_DOWNWIDE"); return 0
    if d > TH:
        print("[동작점 규칙] 실패 — 격자(%s) 최소 |차| %.4f > %.4f(T %+d) — T_C 없음(구현 불가 — 미해결)" % (mode, d / 1e4, TH / 1e4, T))
        return 0
    line = "T_C=%d m_ref=%.4f m_C=%.4f diff=%.4f (규칙: %+d 에서 |차| %.4f > %.4f → 격자 %s 최소)" % (T, mr, m, d / 1e4, T0, d0 / 1e4, TH / 1e4, mode)
    open(cal, "w", encoding="utf-8").write(line + "\n")
    print("[동작점 규칙] " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
