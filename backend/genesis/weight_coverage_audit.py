#!/usr/bin/env python3
"""가중치 커버리지 감사 (C24) — 프로브 런간 불안정의 후보 원인.

증상: 같은 뇌·같은 코드로 프로브를 반복하면 결과가 두 체제로 갈림(ZoneA 100/ZoneB 44 vs ZoneA 50/ZoneB 0).
시행간 리셋을 넣어도 남음 = 불안정이 **런 수준**.

가설: 체크포인트가 뇌의 시냅스 집단 중 일부만 담고 있어(로그: "loaded (54 synapses)"),
**나머지는 매 런 무작위 초기화** → 런마다 다른 뇌. 사실이면 이 세션의 모든 프로브 수치에 영향
(±6~7 노이즈 바닥의 근본원인 후보).
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

ckpt = sys.argv[1] if len(sys.argv) > 1 else "../../checkpoints/brain_concepts_550ep.npz"

b = ForagerBrain(ForagerBrainConfig())
# 뇌가 가진 시냅스 집단 전수
syn = []
for name in dir(b):
    o = getattr(b, name, None)
    if o is not None and hasattr(o, "vars") and hasattr(o, "src_name" if hasattr(o, "src_name") else "name"):
        if type(o).__name__ in ("SynapseGroup",):
            syn.append(name)
if not syn:  # 폴백: 타입명으로 못 잡으면 g 변수 보유로 판정
    for name in dir(b):
        o = getattr(b, name, None)
        try:
            if o is not None and hasattr(o, "vars") and "g" in o.vars:
                syn.append(name)
        except Exception:
            pass

z = np.load(ckpt, allow_pickle=True)
keys = set(z.files)
saved = [s for s in syn if any(k == s or k.startswith(s + "_") or k.endswith("_" + s) for k in keys)]
missing = [s for s in syn if s not in saved]

print("뇌 시냅스 집단: %d개" % len(syn))
print("체크포인트 키: %d개" % len(keys))
print("커버됨: %d | 미커버(매 런 무작위 초기화): %d" % (len(saved), len(missing)))
print("커버율: %.1f%%" % (100.0 * len(saved) / max(len(syn), 1)))
if missing:
    print("\n[미커버 시냅스 일부]")
    for m in missing[:40]:
        print("  -", m)
