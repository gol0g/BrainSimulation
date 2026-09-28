#!/bin/bash
# E119: 반사 0/25 × 학습/무학습 × 뇌 10~14 (E118 조건, 초기 150·행동 창 3스텝 ±5000). 재개 가능(P13).
# 요약 줄 형식은 judge_e119.py 의 TR 과 맞춘다: "  rw0 learn b10: => <사전> <사후> | ... 변조폭 변화 X ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E119.log"
RAW="$R/research/experiments/logs/E119"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E119"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e119_main_run && cd /root/e119_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
JUDGE="exec"   # 경로 검사 P3(불일치 12.4%) → P5(exec 경로 통과) 후 확정(E119.md 7절)
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge $JUDGE"
for B in 10 11 12 13 14; do for RW in 0 25; do for C in learn nolearn; do
  tag="rw$RW $C b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/rw${RW}_${C}_b$B.log"; EXTRA="--save-weights $WD/w_rw${RW}_b$B.npz"; [ "$C" = "nolearn" ] && EXTRA="--no-reward"
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w $RW --episodes 5 --steps 100 --transplant-eval --brain-seed $B $EXTRA > "$f" 2>&1; rc=$?
  if grep -q "변조폭 변화" "$f"; then echo "=> $(grep -E '^\[사전\]|^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/' | tr '\n' ' ')| $(grep '변조폭 변화' "$f" | tail -1 | sed 's/^=> //')"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done; done
echo "[E119] 전체 루프 종료"
