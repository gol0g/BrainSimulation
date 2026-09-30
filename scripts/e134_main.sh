#!/bin/bash
# E134: 이중 해리 — 발달 {corr(identity), shift k=7} × 과제 {tid(sd-shift 0), tsh(sd-shift 7)}, 새 배선 62~77. 재개 가능(P13).
# 요약 줄: "  ds dev corr w62: => DEVHEBB ... || => DEVSHIFT ..." / "  ds corr tsh w62 t600: => [KC불러옴] ... || [KC불러옴이동] ... || => SDLAB ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E134.log"
RAW="$R/research/experiments/logs/E134"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E134"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e134_main_run && cd /root/e134_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-eta 1 --dev-hebb-exposures 400 --dev-shift 7"
dev() {  # 발달환경 배선
  local tag="ds dev $1 w$2"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/dev_$1_w$2.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $2 --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env $1 --dev-hebb-save "$WD/dev_$1_w$2.npz" > "$f" 2>&1; local rc=$?
  if grep -q "^=> DEVSHIFT" "$f"; then echo "$(grep '^=> DEVHEBB' "$f" | sed 's/ → .*//') || $(grep '^=> DEVSHIFT' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
task() {  # 발달환경 과제(tid|tsh) 배선 난수열
  local tag="ds $1 $2 w$3 t$4"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local SH=0; [ "$2" = "tsh" ] && SH=7
  local f="$RAW/$1_$2_w$3_t$4.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode learn --seed $3 --trial-seed $4 $K50 --trials 800 $SDO --sd-diff cyclic --sd-shift $SH --dev-shift 7 --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/dev_$1_w$3.npz" > "$f" 2>&1; local rc=$?
  if grep -q "^\[KC불러옴이동\]" "$f" && grep -q "^=> SDLAB" "$f"; then echo "=> $(grep '^\[KC불러옴\]' "$f") || $(grep '^\[KC불러옴이동\]' "$f") || $(grep '^=> SDLAB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77; do
  dev corr $S; dev shift $S
  for D in corr shift; do for TK in tid tsh; do for T in 600 601; do task $D $TK $S $T; done; done; done
done
echo "[E134] 전체 루프 종료"
