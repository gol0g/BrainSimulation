#!/bin/bash
# 회귀 검증 — 최소 회로의 확립된 성취가 유지되는가.
#
# 확립(E095, 2026-09-25): tau_e=12, act_drive=18, eta=0.001, epsilon=0.6, gap 600, 400시행에서
#   세 배선(w0·w3·w4, 기준선 0%/47%/54%) × 8 난수열 = **24런 전부 평가 100%**.
# 앞으로 학습 코드·회로·파라미터를 바꾸면 **이 스크립트를 먼저 돌린다.**
# 24/24 가 깨지면 그 변경이 능력을 파괴한 것이다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12"
ok=0; tot=0; fail=""
for S in 0 3 4; do
  for T in 100 101 102 103 104 105 106 107; do
    tot=$((tot+1))
    f=/root/minc_run/reg_w${S}t${T}.log
    timeout 3600 python minimal_circuit.py --mode learn --seed "$S" --trial-seed "$T" $CFG > "$f" 2>&1
    ev=$(grep -oE '\*\*eval=[0-9.]+\*\*' "$f" | grep -oE '[0-9.]+' | head -1)
    if [ "$ev" = "100.0" ]; then ok=$((ok+1)); else fail="$fail w${S}t${T}=${ev:-실패}"; fi
  done
done
echo "[회귀] 평가 100%: $ok/$tot"
if [ "$ok" -eq "$tot" ]; then
  echo "[회귀] **통과** — E095의 성취가 유지된다"
else
  echo "[회귀] **실패** — 깨진 런:$fail"
  exit 1
fi
