#!/usr/bin/env python3
"""무학습 대조군 체크포인트 생성 (C24 이후 필수 대조).

C24 이후 핵심 질문은 "성능이 있느냐"가 아니라 **"학습이 기여하느냐"**다.
따라서 모든 개념 측정에는 학습을 전혀 하지 않은 뇌가 **같은 로드 경로를 거쳐** 대조로 들어가야 한다.
(평가 스크립트가 --load-weights를 필수로 요구하므로 무학습 뇌도 파일로 저장해 쓴다.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/brain_untrained.npz"
b = ForagerBrain(ForagerBrainConfig())
b.save_all_weights(out)
print("[untrained] 무학습 뇌 저장: %s" % out)
