#!/usr/bin/env python3
"""FlyWire 버섯체(MB) 실측 — 우리 KC 층이 학습에 참여하지 않는 문제와 직결."""
import pandas as pd, numpy as np
cls = pd.read_csv("data/flywire/classification.csv.gz")
conn = pd.read_parquet("data/flywire/shiu_model/Connectivity_783.parquet")

def find(pat, col="hemibrain_type"):
    s = cls[col].fillna("")
    return cls[s.str.contains(pat, case=False, regex=True, na=False)]

kc = cls[cls["class"].fillna("") == "Kenyon_Cell"]
mbon = find(r"^MBON")
dan = find(r"^(PAM|PPL1)")   # 도파민 뉴런
print("=== MB 구성 뉴런 수 (실측) ===")
print("  KC   %5d" % len(kc))
print("  MBON %5d" % len(mbon))
print("  DAN  %5d  (PAM+PPL1)" % len(dan))

# 연결 분석: root_id -> index 매핑이 필요
idmap = conn[["Presynaptic_ID","Presynaptic_Index"]].drop_duplicates().set_index("Presynaptic_ID")["Presynaptic_Index"]
idmap2 = conn[["Postsynaptic_ID","Postsynaptic_Index"]].drop_duplicates().set_index("Postsynaptic_ID")["Postsynaptic_Index"]
allmap = pd.concat([idmap, idmap2]).groupby(level=0).first()

def ids(df): return set(allmap.reindex(df.root_id.values).dropna().astype(int))
KC, MBON, DAN = ids(kc), ids(mbon), ids(dan)
print("  (연결 데이터에 존재: KC %d / MBON %d / DAN %d)" % (len(KC), len(MBON), len(DAN)))

def edges(src, dst):
    m = conn.Presynaptic_Index.isin(src) & conn.Postsynaptic_Index.isin(dst)
    return conn[m]

print("\n=== KC 회로 실측 ===")
for nm, src, dst in [("KC→MBON", KC, MBON), ("KC→KC", KC, KC),
                     ("DAN→KC", DAN, KC), ("DAN→MBON", DAN, MBON),
                     ("MBON→MBON", MBON, MBON)]:
    e = edges(src, dst)
    if len(e) == 0:
        print("  %-10s 연결 없음" % nm); continue
    print("  %-10s 연결 %6d | 시냅스합 %8d | 연결당 중앙값 %2d p99 %3d | 억제비율 %.0f%%"
          % (nm, len(e), e.Connectivity.sum(), e.Connectivity.median(),
             e.Connectivity.quantile(.99), (e.Excitatory != 1).mean()*100))

# KC 하나가 몇 개 MBON에 연결되는가 (수렴도)
e = edges(KC, MBON)
if len(e):
    per_kc = e.groupby("Presynaptic_Index").size()
    per_mbon = e.groupby("Postsynaptic_Index").size()
    print("\n=== 수렴 구조 (학습이 일어나는 지점) ===")
    print("  KC 1개 → MBON %d개 (중앙값), p90 %d" % (per_kc.median(), per_kc.quantile(.9)))
    print("  MBON 1개 ← KC %d개 (중앙값), p90 %d" % (per_mbon.median(), per_mbon.quantile(.9)))
    print("  → 우리 뇌: KC(2000~4000) → D1 좌/우 2개. 수렴비가 실측과 맞는가?")
