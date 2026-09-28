#!/bin/bash
# E119 경로 검사 P5: --judge exec 새 경로. 뇌 15(표본 밖). 반사 25는 불일치 0이라 judge v(P3)와 소수점까지 같아야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E119"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e119_run && cd /root/e119_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for RW in 0 25; do
  f="$OUT/path_exec_rw${RW}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $RW --episodes 5 --steps 100 --transplant-eval --brain-seed 15 > "$f" 2>&1; rc=$?
  if grep -q "변조폭 변화" "$f"; then echo "  exec_rw$RW: $(grep -E '^\[학습\]|^\[판정경로\]|^\[사후\]|^=> ' "$f" | tr '\n' ' ')"; else echo "  exec_rw$RW: [실패 rc=$rc]"; tail -3 "$f"; fi
done
echo "[E119 exec 검사] 종료"
