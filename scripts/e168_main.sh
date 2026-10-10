#!/bin/bash
# E168 본실험 — 기준 logs/E168/criteria_fixed.txt. 보정 선택 W*(logs/E168/pick.txt — e168_calib.sh·e168_pick.py)가 'none' 이면 실행하지 않는다.
# 뇌 16~20, 형성 표현(E161 가중치), 반사 0, 500시행, 보상 창 동결 없음, --rw-da-reset, --kc-rw-diag(읽기 전용).
# 팔 FI = 도파민 뉴런 → KC 억제 뉴런 연결 W*(확률 0.2), FR = 연결 없음. 기준 = 같은 뇌 E161 F(동결). 요약 줄 "  e168 {FI|FR} b16: => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E168.log"
RAW="$R/research/experiments/logs/E168"; WD="$R/research/experiments/traces/E168"; mkdir -p "$RAW" "$WD"
W161="$R/research/experiments/traces/E161"
WSTAR=$(grep -oE '^W=[0-9a-z.]+' "$RAW/pick.txt" 2>/dev/null | cut -d= -f2)
if [ -z "$WSTAR" ] || [ "$WSTAR" = "none" ]; then echo "[E168] 보정 선택 없음(W*=${WSTAR:-결측}) — 본실험 실행 안 함"; echo "[E168] 전체 루프 종료"; exit 0; fi
echo "[E168] W* = $WSTAR"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e168_main_run && cd /root/e168_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0"
for B in 16 17 18 19 20; do
  for ARM in FI FR; do
    tag="e168 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ "$ARM" = "FI" ] && W=$WSTAR || W=0
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $W161/kctype_oja_b$B.npz \
      --rw-da-reset --kc-rw-diag --da-kc-inh $W --da-kc-inh-p 0.2 \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $W161/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f" && grep -q "^\[E168 KC 발화\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^\[E168 KC 발화\]' "$f" | grep -oE '평균 [0-9.]+' | tr '\n' ' ')|| 연결 $(grep -c '^  \[E168 도파민→KC억제\]' "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E168] 전체 루프 종료"
