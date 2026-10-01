#!/bin/bash
# E137: 관계 전이의 용량-반응(헌장 개념 조건 4). Oja 발달(E136 과 같은 인자, corr) → 가지치기 연결(loaded) → 학습 100·200·400·800시행 × 난수열 600·601,
# 무학습 100·800시행(난수열 600). 새 배선 94~109. 재개 가능(P13). E136 대비 바뀐 것: 시행 수, --block 100(출력 전용), --dump-rewards(기록 전용).
# 요약 줄: "  e137 dev w94: => DEVHEBB ..." / "  e137 learn w94 T100 t600: => [KC불러옴] ... || => SDLAB ... || 블록 1 || 보상 100 || dl ... dr ..."  (judge_e137.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E137.log"
RAW="$R/research/experiments/logs/E137"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E137"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e137_main_run && cd /root/e137_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 100 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-exposures 400 --dev-mode oja --dev-oja-eta 0.005 --dev-oja-beta 5 --dev-oja-mmax-c 2.0 --dev-oja-mmax-i 4.0"
dev() {  # 배선
  local tag="e137 dev w$1"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/dev_corr_w$1.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $1 --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env corr --dev-hebb-save "$WD/dev_corr_w$1.npz" > "$f" 2>&1; local rc=$?
  if grep -q "^=> DEVHEBB" "$f"; then echo "$(grep '^=> DEVHEBB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
task() {  # 모드 배선 시행수 난수열
  local tag="e137 $1 w$2 T$3 t$4"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/$1_w$2_T$3_t$4.log"; local rw="-"; local rwopt=""
  if [ "$1" = "learn" ]; then rw="$WD/rw_w$2_T$3_t$4.txt"; rwopt="--dump-rewards $rw"; fi
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $1 --seed $2 --trial-seed $4 $K50 --trials $3 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/dev_corr_w$2.npz" $rwopt > "$f" 2>&1; local rc=$?
  if grep -q "^\[KC불러옴\]" "$f" && grep -q "^=> SDLAB" "$f"; then
    local nb; nb=$(grep -c '^  시행 ' "$f")
    local nr="-"; if [ "$rw" != "-" ]; then nr=$(wc -l < "$rw" 2>/dev/null || echo NA); fi
    local dl; dl=$(sed -nE 's/^  kc_out_l: n=[0-9]+ [|]Δ[|]평균 ([0-9.]+).*/\1/p' "$f")
    local dr; dr=$(sed -nE 's/^  kc_out_r: n=[0-9]+ [|]Δ[|]평균 ([0-9.]+).*/\1/p' "$f")
    echo "=> $(grep '^\[KC불러옴\]' "$f") || $(grep '^=> SDLAB' "$f") || 블록 $nb || 보상 $nr || dl $dl dr $dr"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109; do
  dev $S
  for T in 100 200 400 800; do task learn $S $T 600; task learn $S $T 601; done
  task frozen $S 100 600; task frozen $S 800 600
done
echo "[E137] 전체 루프 종료"
