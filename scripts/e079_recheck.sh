#!/bin/bash
# E079 (a) 재측정 — INV-A4/A5 적용판.  2026-09-16
# 1차는 kc_learning_probe.py가 d1_inhibition을 지정하지 않아 **D1 포화 상태**에서 돌았다(규약 P15).
# R-STDP 자격흔적은 pre/post 발화에 의존하므로, 포화 상태의 std 4.33 / 변화율 29.8%는 무효다.
# 사전 기준(원본 유지): std > 0.1 AND 변화율 < 95% → 신용할당 발생.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "### 기본 (집단 스칼라 갱신) — d1_inhib -400"
timeout 2400 python kc_learning_probe.py --d1-inhib -400 --direct-inhib -100 > /root/kc_run/e079r_base.log 2>&1
rc=$?; out=$(grep -E "kc_to_d1|food_to_d1|시냅스 " /root/kc_run/e079r_base.log)
[ -z "$out" ] && { echo "[실패 rc=$rc]"; tail -4 /root/kc_run/e079r_base.log; } || echo "$out"

echo
echo "### --kc-rstdp (시냅스별 자격흔적) — d1_inhib -400"
timeout 2400 python kc_learning_probe.py --kc-rstdp --d1-inhib -400 --direct-inhib -100 > /root/kc_run/e079r_kc.log 2>&1
rc=$?; out=$(grep -E "E079|kc_to_d1|food_to_d1|시냅스 " /root/kc_run/e079r_kc.log)
[ -z "$out" ] && { echo "[실패 rc=$rc]"; tail -4 /root/kc_run/e079r_kc.log; } || echo "$out"
