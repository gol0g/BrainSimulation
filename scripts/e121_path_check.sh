#!/bin/bash
# E121 경로 검사(측정 확인): 이식 평가가 학습을 다 담는가. 뇌 15(표본 밖), 반사 0, 9 에피소드(짧은 진단). E119 BASE·ACT(judge exec).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E121"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e121_run && cd /root/e121_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --episodes 9 --steps 100 --brain-seed 15"
run() {  # 이름, 추가 인자
  f="$OUT/$1.log"
  timeout 7200 python reflex_override_task.py $BASE $ACT $2 > "$f" 2>&1; rc=$?
  if grep -q "변조폭 변화" "$f"; then echo "  $1: $(grep -E '^\[사전\]|^\[학습\]|^\[사후\]|^=> |^\[전체시냅스\] 집단|^\[추적\]' "$f" | tr '\n' ' ')"; else echo "  $1: [실패 rc=$rc]"; tail -3 "$f"; fi
}
run C0_learn_tp "--transplant-eval"
run C1_learn_tp_snap "--transplant-eval --snap-all-syn"
run C2_learn_tp_trace "--transplant-eval --trace-kc-motor $OUT/C2_trace.csv"
run C3_learn_direct ""
run C4_nolearn_direct_snap "--no-reward --snap-all-syn"
echo "[E121 경로 검사] 종료"
