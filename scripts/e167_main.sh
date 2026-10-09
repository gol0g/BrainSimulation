#!/bin/bash
# E167 본실험 — 기준 logs/E167/criteria_fixed.txt. 최소 회로 관계 전이에서 호스트 승자 선택·가중치 재부여 제거(외부 검토 권고 2).
# 배선 78~93(E136 과 같음): corr 발달(E136 설정 그대로, 후보 전체 저장) → 발달이 끝난 망 그대로(ojafull) 과제 learn·frozen × 난수열 600·601.
# 호스트 경로(H)는 같은 배선 E136 corr learn·frozen 원 로그. 요약 줄: "  e167 dev w78: => ..." / "  e167 N learn w78 t600: => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E167.log"
RAW="$R/research/experiments/logs/E167"; WD="$R/research/experiments/traces/E167"; mkdir -p "$RAW" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e167_main_run && cd /root/e167_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-exposures 400 --dev-mode oja --dev-oja-eta 0.005 --dev-oja-beta 5 --dev-oja-mmax-c 2.0 --dev-oja-mmax-i 4.0"
for W in 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93; do
  F="$WD/dev_corr_w$W.npz"
  tag="e167 dev w$W"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    f="$RAW/dev_corr_w$W.log"; printf "  %s: " "$tag"
    timeout 3600 python minimal_circuit.py --mode frozen --seed $W --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env corr --dev-hebb-save "$F" > "$f" 2>&1; rc=$?
    if grep -q "^=> DEVHEBB" "$f" && [ -f "$F" ]; then echo "=> $(grep -oE '가지치기 후 같은 위치: 일치형 [0-9.]+ 불일치형 [0-9.]+' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; continue; fi
  fi
  for TS in 600 601; do
    for MODE in learn frozen; do
      tag="e167 N $MODE w$W t$TS"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      f="$RAW/N_${MODE}_w${W}_t$TS.log"; printf "  %s: " "$tag"
      timeout 3600 python minimal_circuit.py --mode $MODE --seed $W --trial-seed $TS $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring ojafull --kc-wiring-file "$F" > "$f" 2>&1; rc=$?
      if grep -q "^\[KC망안\]" "$f" && grep -q "^=> SDLAB" "$f"; then
        echo "=> $(grep '^\[KC망안\]' "$f" | grep -oE '같은 위치 몫 [0-9.]+|최대 \|차\| [0-9.e+-]+' | tr '\n' ' ')|| $(grep '^=> SDLAB' "$f" | grep -oE 'train_lbal=[0-9.]+|novel_lbal=[0-9.]+' | tr '\n' ' ')"
      else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    done
  done
done
echo "[E167] 전체 루프 종료"
