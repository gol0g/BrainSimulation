#!/bin/bash
# E133: 망 스파이크 헤브 발달(corr/indep) → 가지치기 연결로 같음/다름 과제. 새 배선 46~61. 재개 가능(P13).
# 요약 줄: "  hb dev corr w46: => DEVHEBB ..." / "  hb corr learn w46 t600: => [KC불러옴] ... || => SDLAB ..."  (judge_e133.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E133.log"
RAW="$R/research/experiments/logs/E133"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E133"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e133_main_run && cd /root/e133_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-eta 1 --dev-hebb-exposures 400"
dev() {  # 환경 배선
  local tag="hb dev $1 w$2"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/dev_$1_w$2.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $2 --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env $1 --dev-hebb-save "$WD/dev_$1_w$2.npz" > "$f" 2>&1; local rc=$?
  if grep -q "^=> DEVHEBB" "$f"; then echo "$(grep '^=> DEVHEBB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
task() {  # 환경 모드 배선 난수열
  local tag="hb $1 $2 w$3 t$4"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/$1_$2_w$3_t$4.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $2 --seed $3 --trial-seed $4 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/dev_$1_w$3.npz" > "$f" 2>&1; local rc=$?
  if grep -q "^\[KC불러옴\]" "$f" && grep -q "^=> SDLAB" "$f"; then echo "=> $(grep '^\[KC불러옴\]' "$f") || $(grep '^=> SDLAB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61; do
  dev corr $S; dev indep $S
  task corr learn $S 600; task corr learn $S 601; task indep learn $S 600; task indep learn $S 601; task corr frozen $S 600; task corr frozen $S 601
done
echo "[E133] 전체 루프 종료"
