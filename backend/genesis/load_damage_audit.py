#!/usr/bin/env python3
"""로드 손상 감사 (C24) — 체크포인트 로드가 학습 구조를 얼마나 파괴하는가.

발견: `_load_sparse_weights`(forager_brain.py:5996)는 SPARSE 연결 개수가 저장본과 다르면
저장 가중치를 **평균 스칼라 하나로 브로드캐스트**한다(6004~6006). SPARSE 연결은 런마다 무작위
생성되므로 개수가 어긋나고 → **학습된 시냅스별 구조가 상수로 치환**된다.
로그는 "Weights loaded"라고 말하지만 실제로는 학습 내용이 지워진 뇌일 수 있다.

여기서 정량화: 로드 시 (a)정상 복원 (b)평균 브로드캐스트(=구조 파괴) (c)미포함 이 각각 몇 개인가.
파괴 비율이 크면 이 세션의 모든 "학습된 뇌" 측정이 무효 — 노이즈 바닥 ±6~7의 정체이기도 하다.
"""
import sys, os, io, contextlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

ckpt = sys.argv[1]
b = ForagerBrain(ForagerBrainConfig())

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    b.load_all_weights(ckpt)
out = buf.getvalue()

mismatch = re.findall(r"Shape mismatch \((\(\d+,\))→(\(\d+,\))\), broadcast mean=([\d.]+)", out)
loaded = re.search(r"loaded from .*? \((\d+) synapses\)", out)
n_loaded = int(loaded.group(1)) if loaded else -1

print("=" * 62)
print("체크포인트: %s" % os.path.basename(ckpt))
print("로더가 '로드했다'고 보고한 시냅스: %d" % n_loaded)
print("그중 평균-브로드캐스트로 **구조 파괴**된 것: %d" % len(mismatch))
print("정상 복원(시냅스별 구조 유지): %d" % (n_loaded - len(mismatch)))
if n_loaded > 0:
    print("파괴 비율: %.1f%%" % (100.0 * len(mismatch) / n_loaded))
print("=" * 62)
for a, c, m in mismatch[:20]:
    print("  %s → %s  (평균 %s 로 치환)" % (a, c, m))
