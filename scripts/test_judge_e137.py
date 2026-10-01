#!/usr/bin/env python3
"""judge_e137.py 합성 시험(조건 1): 지지·천장·평탄·전이 실패·경계(14/16 vs 13/16)·비단조·늦은 하락·조작검증 실패 7종·동점·결측·줄 파싱.
실행: python3 scripts/test_judge_e137.py  (저장소 루트에서)"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_e137 as J

W = J.WIRES


def build(nfun, ffun=lambda w, d: 50.0, tr_gap=0.0, mm=0.82, mx=0.89, mut=None):
    """nfun(i, d, s) → 학습 새 항목 정답률. 발달·연결·블록·보상·|Δg| 는 정상값으로 채운다."""
    D = {w: {"seed": w, "env": "corr", "fm": 0.03, "fx": 0.30, "mm": mm, "mx": mx} for w in W}
    T, RW = {}, {}
    for i, w in enumerate(W):
        base = ["1" if (j * 7 + w) % 3 else "0" for j in range(800)]
        for d in J.DOSES:
            for s in J.TSEEDS:
                nl = nfun(i, d, s)
                T[("learn", w, d, s)] = {"file": "/x/traces/E137/dev_corr_w%d.npz" % w, "lm": round(mm * 400), "nm": 400, "lx": round(mx * 400),
                                         "nx": 400, "mode": "learn", "seed": w, "ts": s, "tl": nl + tr_gap, "nl": nl, "blocks": d // 100,
                                         "rw": str(d), "dl": 0.01 * d / 100, "dr": 0.011 * d / 100}
                RW[(w, d, s)] = base[:d]
        for d in J.FDOSES:
            T[("frozen", w, d, 600)] = {"file": "/x/traces/E137/dev_corr_w%d.npz" % w, "lm": round(mm * 400), "nm": 400, "lx": round(mx * 400),
                                        "nx": 400, "mode": "frozen", "seed": w, "ts": 600, "tl": 50.0, "nl": ffun(w, d), "blocks": d // 100,
                                        "rw": "-", "dl": 0.0, "dr": 0.0}
    if mut:
        mut(D, T, RW)
    return D, T, RW


def verdict(D, T, RW):
    c, r = J.judge(D, T, RW)
    return (r["verdict"] if r else None), c, r


CURVE = {100: 60.0, 200: 70.0, 400: 80.0, 800: 90.0}
cases = []
# 1 지지: 단조 증가, 무학습 50
cases.append(("지지", build(lambda i, d, s: CURVE[d] + (i % 3) - 1), "지지(H060)"))
# 2 천장: 100에서 이미 85, 800 88(D +3)
cases.append(("천장", build(lambda i, d, s: {100: 85, 200: 86, 400: 87, 800: 88}[d] + (i % 2)), "보류(천장"))
# 3 평탄: 70 → 72, 무학습 50
cases.append(("평탄", build(lambda i, d, s: {100: 70, 200: 71, 400: 71, 800: 72}[d] + (i % 3) - 1), "용량 무관"))
# 4 전이 재현 실패: 800 도 55, 무학습 50 → S3 불충족
cases.append(("전이실패", build(lambda i, d, s: {100: 50, 200: 52, 400: 53, 800: 55}[d]), "보류(전이 재현 실패"))
# 5 경계 S1: D 평균 정확히 +10(−2×2, 11.5×12, 13×2 → 160/16), 양수 14/16 → p=0.0042 → 지지
D14 = [-2.0, -2.0] + [11.5] * 12 + [13.0] * 2
cases.append(("경계14", build(lambda i, d, s: 60.0 + D14[i] if d == 800 else {100: 60.0, 200: 66.0, 400: 68.0}[d]), "지지(H060)"))
# 6 경계 S1 실패: 양수 13/16(−2×3, 12.5×12, 16×1 → 평균 10.0) → p=0.0213 → S1 불충족, 평균 ≥5 → 보류(혼재)
D13 = [-2.0, -2.0, -2.0] + [12.5] * 12 + [16.0]
cases.append(("경계13", build(lambda i, d, s: 60.0 + D13[i] if d == 800 else {100: 60.0, 200: 66.0, 400: 68.0}[d]), "보류(혼재)"))
# 7 비단조: N200 평균이 N100 보다 낮음 → S2 불충족 → 보류(혼재)
cases.append(("비단조", build(lambda i, d, s: {100: 60, 200: 58, 400: 80, 800: 90}[d] + (i % 3) - 1), "보류(혼재)"))
# 8 늦은 하락: N400 92 → N800 84(−8) → S2 불충족
cases.append(("늦은하락", build(lambda i, d, s: {100: 60, 200: 75, 400: 92, 800: 84}[d] + (i % 3) - 1), "보류(혼재)"))
# 9~15 조작검증 실패
def m_dev(D, T, RW):
    for w in W:
        D[w]["mm"] = 0.30
cases.append(("M1 일치형 형성 실패", build(lambda i, d, s: CURVE[d], mut=m_dev), "보류(조작검증 실패)"))
def m_conn(D, T, RW):
    T[("learn", 97, 400, 601)]["lm"] += 3
cases.append(("M2 연결 불일치", build(lambda i, d, s: CURVE[d], mut=m_conn), "보류(조작검증 실패)"))
def m_block(D, T, RW):
    T[("learn", 100, 200, 600)]["blocks"] = 1
cases.append(("M3 블록 수", build(lambda i, d, s: CURVE[d], mut=m_block), "보류(조작검증 실패)"))
def m_rw(D, T, RW):
    T[("learn", 101, 800, 600)]["rw"] = "799"
cases.append(("M3 보상 줄", build(lambda i, d, s: CURVE[d], mut=m_rw), "보류(조작검증 실패)"))
def m_dg(D, T, RW):
    for w in W:
        for s in J.TSEEDS:
            T[("learn", w, 400, s)]["dl"] = T[("learn", w, 800, s)]["dl"]; T[("learn", w, 400, s)]["dr"] = T[("learn", w, 800, s)]["dr"]
cases.append(("M4 |Δg| 비증가", build(lambda i, d, s: CURVE[d], mut=m_dg), "보류(조작검증 실패)"))
def m_fz(D, T, RW):
    T[("frozen", 105, 800, 600)]["dl"] = 0.00001
cases.append(("M4 무학습 |Δg|≠0", build(lambda i, d, s: CURVE[d], mut=m_fz), "보류(조작검증 실패)"))
cases.append(("M5 무학습 T100≠T800", build(lambda i, d, s: CURVE[d], ffun=lambda w, d: 50.0 + (6.0 if (w == 99 and d == 800) else 0.0)), "보류(조작검증 실패)"))
# 16 동점: D=0 두 배선 + 양수 14(12×10, 12.5×4 → 합 170) → nz 14 → 지지
DT = [0.0, 0.0] + [12.0] * 10 + [12.5] * 4
cases.append(("동점", build(lambda i, d, s: 60.0 + DT[i] if d == 800 else {100: 60.0, 200: 66.0, 400: 68.0}[d]), "지지(H060)"))

ok_all = True
for name, (D, T, RW), want in cases:
    v, c, r = verdict(D, T, RW)
    good = v is not None and v.startswith(want)
    ok_all &= good
    print("%-18s 기대 %-22s → %s %s" % (name, want, v, "✓" if good else "✗"))

# 결측: 런 하나 빠지면 수치 미출력
D, T, RW = build(lambda i, d, s: CURVE[d]); del T[("learn", 94, 100, 600)]
c, r = J.judge(D, T, RW)
good = r is None and "결측 1/176" in c[0]
ok_all &= good
print("%-18s 기대 결측·수치 미출력 → %s %s" % ("결측", c[0][:40], "✓" if good else "✗"))

# 중첩 보고: 한 런의 앞 100 이 다르면 95/96
D, T, RW = build(lambda i, d, s: CURVE[d]); RW[(94, 800, 600)] = ["9"] + RW[(94, 800, 600)][1:]
c, r = J.judge(D, T, RW)
good = any("95/96" in x for x in c) and r["verdict"].startswith("지지")
ok_all &= good
print("%-18s 기대 중첩 95/96(판정 불변) → %s %s" % ("중첩 보고", [x for x in c if "중첩" in x][0][-6:], "✓" if good else "✗"))

# lag 부지표: 훈련이 새 항목보다 12%p 높다가 800 에서 3%p
D, T, RW = build(lambda i, d, s: CURVE[d])
for k in T:
    if k[0] == "learn":
        T[k]["tl"] = T[k]["nl"] + (12.0 if k[2] in (100, 200) else 3.0)
c, r = J.judge(D, T, RW)
good = r["lag"] is True and r["verdict"].startswith("지지")
ok_all &= good
print("%-18s 기대 lag True → %s %s" % ("lag 부지표", r["lag"], "✓" if good else "✗"))

# 줄 파싱: 러너가 찍는 형식 그대로
dev = ("  e137 dev w94: => DEVHEBB seed=94 env=corr exposures=400 eta=1.00 w_fix=4.00 wc_total=4.00 wi_total=8.00 | 발화율(KC·노출당) 일치형 0.031 불일치형 0.300 | "
       "가지치기 후 같은 위치: 일치형 0.825 불일치형 0.888 | 같은 위치 후보 가중치 몫 평균: 일치형 0.028 불일치형 0.058 (균등=0.020) → /mnt/c/x/traces/E137/dev_corr_w94.npz\n")
lrn = ("  e137 learn w94 T100 t600: => [KC불러옴] /mnt/c/x/traces/E137/dev_corr_w94.npz | 일치형 같은 위치 330/400 | 불일치형 같은 위치 355/400 || "
       "=> SDLAB diff=cyclic rule=samediff mode=learn seed=94 trialseed=600 train_accL=80.0 train_accR=70.0 train_lbal=75.0 novel_accL=70.0 novel_accR=60.0 novel_lbal=65.0"
       " || 블록 1 || 보상 100 || dl 0.01000 dr 0.01100\n")
frz = ("  e137 frozen w94 T800 t600: => [KC불러옴] /mnt/c/x/traces/E137/dev_corr_w94.npz | 일치형 같은 위치 330/400 | 불일치형 같은 위치 355/400 || "
       "=> SDLAB diff=cyclic rule=samediff mode=frozen seed=94 trialseed=600 train_accL=40.0 train_accR=60.0 train_lbal=50.0 novel_accL=45.0 novel_accR=55.0 novel_lbal=50.0"
       " || 블록 8 || 보상 - || dl 0.00000 dr 0.00000\n")
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "E137.log"), "w", encoding="utf-8").write(dev + lrn + frz)
    J.EXP = td; J.TRACE = os.path.join(td, "traces")
    Dp, Tp, RWp = J.load()
good = (Dp == {94: {"seed": 94, "env": "corr", "fm": 0.031, "fx": 0.300, "mm": 0.825, "mx": 0.888}}
        and Tp[("learn", 94, 100, 600)]["nl"] == 65.0 and Tp[("learn", 94, 100, 600)]["blocks"] == 1 and Tp[("learn", 94, 100, 600)]["rw"] == "100"
        and Tp[("learn", 94, 100, 600)]["dr"] == 0.011 and Tp[("frozen", 94, 800, 600)]["rw"] == "-" and Tp[("frozen", 94, 800, 600)]["blocks"] == 8
        and Tp[("frozen", 94, 800, 600)]["mode"] == "frozen" and len(Tp) == 2)
ok_all &= good
print("%-18s 기대 발달 1·과제 2 줄 파싱 → %s" % ("줄 파싱", "✓" if good else "✗ %s %s" % (Dp, Tp)))
print("전체: %s" % ("통과" if ok_all else "실패"))
sys.exit(0 if ok_all else 1)
