#!/bin/bash
# 러너 argparse/가드 로직 검증 (GPU 불필요 — import/argparse 단계까지)
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
echo "=== [1] py_compile ==="
python -m py_compile "$R" && echo "compile OK"
echo "=== [2] --help ==="
python "$R" --help >/dev/null 2>&1 && echo "help OK"
echo "=== [3] v3-klino 차단 유지 (exit 2 기대) ==="
python "$R" --task place_pref --biletaxis --v3-klino --episodes 1 2>&1 | grep -E "재건 미완|--v3-klino"
echo "  exit=${PIPESTATUS[0]}"
echo "=== [4] biletaxis+sparse-reward 가드 통과 → 뇌 빌드 진입? ==="
python "$R" --task place_pref --zone-circle --appetitive-place --start-far --sparse-reward --biletaxis --biletaxis-gain 0.5 --n-food 0 --episodes 1 2>&1 | grep -iE "place_pref\]|biletaxis\]|재건 미완|Building Forager" | head -4
