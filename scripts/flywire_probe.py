#!/usr/bin/env python3
"""FlyWire connectome 1차 분석 — 우리 뇌의 추측값을 실측과 비교한다.

왜: 우리 뇌의 배선(반사 g=25·15000시냅스, 학습경로 sparsity 0.08 등)은 전부 내 추측이다.
    "학습이 반사를 못 이긴다"가 설계 실수인지 원리적 한계인지 구분할 근거가 없었다.
    FlyWire에는 실측값이 있다.
"""
import pandas as pd, numpy as np, sys
p = "data/flywire/shiu_model/Connectivity_783.parquet"
df = pd.read_parquet(p)
print("=== 스키마 ===")
print(df.dtypes.to_string())
print("\n행 수: %s" % f"{len(df):,}")
print("\n=== 상위 5행 ===")
print(df.head().to_string())
print("\n=== 가중치 분포 ===")
wcols = [c for c in df.columns if df[c].dtype.kind in "if"]
for c in wcols[:4]:
    s = df[c]
    print("%-14s min=%.3g  p50=%.3g  p90=%.3g  p99=%.3g  max=%.3g  mean=%.3g"
          % (c, s.min(), s.quantile(.5), s.quantile(.9), s.quantile(.99), s.max(), s.mean()))
