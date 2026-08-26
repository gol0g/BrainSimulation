#!/usr/bin/env python3
"""C24 수리 검증 — 시드 고정이 실제로 SPARSE 연결을 재현하는가 (저장→재로드 왕복).

기존 체크포인트로는 판정 불가: 그건 무작위 연결 시절에 저장돼 애초에 맞출 수 없다.
따라서 (1) 시드 적용된 뇌를 만들어 저장 → (2) 새 뇌를 만들어 그걸 로드 → 파괴 0 이면 수리 성공.
동시에 model.seed 속성이 PyGeNN에 실제로 존재하는지도 확인(없으면 조용히 무시됐다는 뜻).
"""
import sys, os, io, re, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from pygenn import GeNNModel

# (0) seed 속성 실재 확인
m = GeNNModel("float", "seedcheck")
has_attr = hasattr(type(m), "seed") or "seed" in dir(m)
print("PyGeNN GeNNModel에 seed 속성 존재: %s" % has_attr)
try:
    m.seed = 999
    print("  seed 대입 성공, 재조회 값 = %r" % getattr(m, "seed", None))
except Exception as e:
    print("  seed 대입 실패: %s" % e)

tmp = "/tmp/seed_roundtrip.npz"

# (1) 저장
b1 = ForagerBrain(ForagerBrainConfig())
b1.save_all_weights(tmp)
print("\n[1] 시드 적용 뇌 저장 완료: %s" % tmp)

# (2) 새 뇌에 로드 — 연결이 재현되면 shape이 전부 맞아야 함
del b1
b2 = ForagerBrain(ForagerBrainConfig())
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    b2.load_all_weights(tmp)
out = buf.getvalue()
mis = re.findall(r"Shape mismatch", out)
mm = re.search(r"loaded from .*? \((\d+) synapses\)", out)
n = int(mm.group(1)) if mm else -1
print("[2] 재로드: 보고 %d 시냅스 | 구조 파괴 %d | 정상 %d" % (n, len(mis), n - len(mis)))
print("판정: %s" % ("수리 성공 (연결 재현됨)" if len(mis) == 0 else "수리 실패 (연결 여전히 비결정)"))
