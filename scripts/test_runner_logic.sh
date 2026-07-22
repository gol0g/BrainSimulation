#!/bin/bash
# 러너 argparse/가드 로직 검증 (GPU 불필요 — import 전 단계까지만)
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
echo "=== [1] py_compile ==="
python -m py_compile "$R" && echo "compile OK"
echo "=== [2] --help ==="
python "$R" --help >/dev/null 2>&1 && echo "help OK"
echo "=== [3] v3-klino 차단 확인 (exit 2 기대) ==="
python "$R" --task place_pref --zone-circle --appetitive-place --start-far --v3-klino --episodes 1 2>&1 | grep -E "재건 미완|--v3-klino"
echo "  exit=${PIPESTATUS[0]}"
echo "=== [4] zone-only 플래그는 가드 통과? (import까지 진행 → pygenn 뜨면 OK) ==="
python "$R" --task place_pref --zone-circle --appetitive-place --start-far --zone-cx 0.3 --zone-cy 0.3 --episodes 0 2>&1 | grep -iE "place_pref\]|Building Forager|재건 미완|Traceback" | head -3
