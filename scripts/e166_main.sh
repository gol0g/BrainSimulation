#!/bin/bash
# E166 본실험 — 기준 logs/E166/criteria_fixed.txt. 보상 창 흔적 동결(호스트 가소성 제어) 제거의 능력 보존(외부 검토 권고 2, 전체 모델).
# 뇌 16~20, 반사 0, 500시행, E141 인자에서 --rw-apm-scale 0 만 뺌. 팔 FNF = + 같은 뇌 E161 망 안 형성 가중치, 팔 DNF = 기본 표현.
# 기준 = 같은 뇌 E161 F(형성 + 동결)·D(기본 + 동결) 원 로그. 추적 rate 파일 = 같은 뇌 E161. 요약 줄: "  e166 {FNF|DNF} b16: => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E166.log"
RAW="$R/research/experiments/logs/E166"; WD="$R/research/experiments/traces/E166"; mkdir -p "$RAW" "$WD"
W161="$R/research/experiments/traces/E161"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e166_main_run && cd /root/e166_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 16 17 18 19 20; do
  for ARM in FNF DNF; do
    tag="e166 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ "$ARM" = "FNF" ] && XA="--kc-type-weights $W161/kctype_oja_b$B.npz" || XA=""
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed $B $XA \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $W161/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E166] 전체 루프 종료"
