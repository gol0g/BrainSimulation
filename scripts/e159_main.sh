#!/bin/bash
# E159 본실험 — 기준 logs/E159/criteria_fixed.txt(2026-10-09 14:35:05, 수정 1 14:35:46). 뇌 10~14, 이득 맞춘 형성 표현(×0.70), 학습 없음.
# 분해 평가(E112 경로): none(무학습)·W0(E157 Fk 반사 0 500시행)·W25(E158 F500 반사 25 500시행) × 평가 반사 0·25 = 6칸 × 5뇌. 재개 가능(P13).
# 요약 줄: "  e159 W0_R25 b10: => mod +0.1234 pushed 8 || 적재 2 배율 2"  (판정은 원 로그를 직접 읽는다)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E159.log"
RAW="$R/research/experiments/logs/E159"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e159_main_run && cd /root/e159_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'
for B in 10 11 12 13 14; do
  KT="--kc-type-weights $R/research/experiments/traces/E153/kctype_b$B.npz --kc-type-scale 0.70"
  for C in none_R0 none_R25 W0_R0 W0_R25 W25_R0 W25_R25; do
    tag="e159 $C b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $C in
      none_*) WF="$R/research/experiments/traces/E157/w_Fk_b$B.npz"; MODE=none ;;
      W0_*) WF="$R/research/experiments/traces/E157/w_Fk_b$B.npz"; MODE=all ;;
      W25_*) WF="$R/research/experiments/traces/E158/w_F500_b$B.npz"; MODE=all ;;
    esac
    case $C in *_R0) RW=0 ;; *_R25) RW=25 ;; esac
    f="$RAW/${C}_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $RW --steps 100 --brain-seed $B $KT --decomp-weights $WF --decomp-mode $MODE > "$f" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$f"; then
      echo "=> mod $(grep '^=> DECOMP' "$f" | grep -oE 'mod=[-+0-9.]+' | cut -d= -f2) pushed $(grep '^=> DECOMP' "$f" | grep -oE 'pushed=[0-9]+' | cut -d= -f2) || 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E159] 전체 루프 종료"
