#!/usr/bin/env python3
"""FlyWire 실측 통계 vs 우리 뇌의 추측값.

우리가 추측으로 정한 것들:
  - 반사 good_food_eye->motor: 가중치 25, sparsity 0.15 (n≈15000)
  - 학습경로 food_eye->D1: 가중치 1, sparsity 0.08 (n≈3200)
  - E/I: 억제를 -100~-400으로 "튜닝"했으나 근거는 포화 해소뿐
실측에서 이 비율들이 어떤지 본다.
"""
import pandas as pd, numpy as np
df = pd.read_parquet("data/flywire/shiu_model/Connectivity_783.parquet")
n_pre = df.Presynaptic_Index.nunique(); n_post = df.Postsynaptic_Index.nunique()
n_neurons = max(df.Presynaptic_Index.max(), df.Postsynaptic_Index.max()) + 1

print("=== 규모 ===")
print("뉴런 %s / 연결(쌍) %s / 시냅스 총합 %s"
      % (f"{n_neurons:,}", f"{len(df):,}", f"{df.Connectivity.sum():,}"))
print("연결 밀도 = %.2e (우리 sparsity 0.08~0.15와 비교)"
      % (len(df) / (n_neurons * n_neurons)))

print("\n=== 시냅스 수(연결 강도) 분포 ===")
c = df.Connectivity
for q in (.5, .75, .9, .99, .999):
    print("  p%-5s %d" % (int(q*1000)/10, c.quantile(q)))
print("  max %d   mean %.2f" % (c.max(), c.mean()))
print("  → 대부분의 연결이 시냅스 1~2개. 강한 연결은 극소수(롱테일)")

print("\n=== 흥분/억제 비율 ===")
ex = df.Excitatory
print("  흥분 연결 %.1f%% / 억제 %.1f%%" % ((ex == 1).mean()*100, (ex != 1).mean()*100))
w_ex = df.loc[ex == 1, "Connectivity"]; w_in = df.loc[ex != 1, "Connectivity"]
print("  흥분 시냅스 총합 %s / 억제 %s  → 억제/흥분 = %.3f"
      % (f"{w_ex.sum():,}", f"{w_in.sum():,}", w_in.sum()/w_ex.sum()))
print("  평균 강도: 흥분 %.2f / 억제 %.2f" % (w_ex.mean(), w_in.mean()))

print("\n=== 뉴런당 출력/입력 ===")
out_deg = df.groupby("Presynaptic_Index").size()
in_deg = df.groupby("Postsynaptic_Index").size()
print("  출력 연결수: p50 %d / p90 %d / max %d" % (out_deg.quantile(.5), out_deg.quantile(.9), out_deg.max()))
print("  입력 연결수: p50 %d / p90 %d / max %d" % (in_deg.quantile(.5), in_deg.quantile(.9), in_deg.max()))
out_syn = df.groupby("Presynaptic_Index").Connectivity.sum()
print("  출력 시냅스합: p50 %d / p90 %d / p99 %d / max %d"
      % (out_syn.quantile(.5), out_syn.quantile(.9), out_syn.quantile(.99), out_syn.max()))

print("\n=== 우리 뇌와 비교 ===")
print("  우리 반사:   1개 경로에 시냅스 15,000  (한 뉴런 출력합 p99=%d 와 비교)" % out_syn.quantile(.99))
print("  우리 학습:   1개 경로에 시냅스  3,200")
print("  우리 반사/학습 시냅스비 = %.1f" % (15000/3200))
