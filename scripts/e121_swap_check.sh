#!/bin/bash
# E121 경로 검사(요소 맞바꾸기 --kc-bilateral-scale): 뇌 15(표본 밖), 반사 0. 짧은 진단(≤9 에피소드).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E121"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e121_run && cd /root/e121_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --steps 100 --brain-seed 15"
run() {  # 이름, 기대 줄 패턴, 인자...
  local n="$1" pat="$2"; shift 2; local f="$OUT/$n.log"
  timeout 7200 python reflex_override_task.py "$@" > "$f" 2>&1; local rc=$?
  if grep -qE "$pat" "$f"; then echo "  $n: $(grep -E '^\[KC공통입력\]|^\[사전\]|^\[학습\]|^\[사후\]|^=> |^\[전체시냅스\] 집단' "$f" | sed -E 's/(\[KC공통입력\] scale=[0-9.]+ 집단 [0-9]+개: [^;]*;[^;]*).*/\1 …/' | tr '\n' ' ')"; else echo "  $n: [실패 rc=$rc]"; tail -3 "$f"; fi
}
run S1_regress_scale1_learn9 '^=> 정답률' $BASE $ACT --episodes 9 --transplant-eval --kc-bilateral-scale 1
run S2_base_scale0 '^=> 정답률' $BASE $ACT --episodes 0 --transplant-eval --kc-bilateral-scale 0
run S3_rev_scale0 '^=> CALIBKM1' $BASE --reflex-w 0 --episodes 0 --brain-seed 15 --calib-kc-motor-set rev --kc-bilateral-scale 0
run S4_kcsets_scale1 '^=> KCSETS' $BASE --reflex-w 0 --episodes 0 --brain-seed 15 --decomp-weights "$R/research/experiments/traces/E119/path_rw0_b15.npz" --decomp-mode kcsets --kc-bilateral-scale 1
run S5_kcsets_scale0 '^=> KCSETS' $BASE --reflex-w 0 --episodes 0 --brain-seed 15 --decomp-weights "$R/research/experiments/traces/E119/path_rw0_b15.npz" --decomp-mode kcsets --kc-bilateral-scale 0
run S6_learn9_scale0_snap '^=> 정답률' $BASE $ACT --episodes 9 --transplant-eval --kc-bilateral-scale 0 --snap-all-syn
echo "[E121 요소 검사] 종료"
