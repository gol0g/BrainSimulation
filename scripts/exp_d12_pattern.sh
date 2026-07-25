#!/bin/bash
# D12 결정판: WM 패턴 상관으로 PBWM 래치 확정 판정. pattern_corr>0.5=A패턴 유지=진짜 래치.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## PBWM 래치 패턴 판정 (seq-wm ON), 20ep ##########"
python -u $R --task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 0 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 20 --seed 0 --output "$D/d12_pattern.json" 2>&1 | grep -iE "WM pattern_corr|last_5_pattern|WM latch|최종순서율|Traceback|Error" | tail -6
echo "########## DONE ##########"
