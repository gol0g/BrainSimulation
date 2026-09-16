#!/bin/bash
# E079 조작검증: KC를 시냅스별 R-STDP로 바꾸면 신용 할당이 생기는가.
# 사전기준: std(후) > 0.1 AND 변화율 < 95%  (현재는 std 0.0000 / 변화율 100%)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
echo "########## 기본 (집단 스칼라) ##########"
timeout 2400 python kc_learning_probe.py 2>&1 | grep -E "kc_to_d1|food_to_d1|시냅스 " 
echo "########## --kc-rstdp (시냅스별) ##########"
rm -rf forager_brain_CODE CODE
timeout 2400 python kc_learning_probe.py --kc-rstdp 2>&1 | grep -E "E079|kc_to_d1|food_to_d1|시냅스 "
