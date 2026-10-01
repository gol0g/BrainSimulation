#!/bin/bash
# E137 경로 검사(조건 2) — 표본 밖 배선 18(E136 보정에서 발달만 본 배선). 판정 기준 고정(criteria_fixed.txt) 뒤에 돈다.
# (1) K50 회귀(코드 불변 확인) (2) Oja 발달 회귀(E136 보정 w18 corr: 가지치기 0.810/0.920)
# (3) 학습 T100·T200: 블록 줄 수 1·2, 보상 계열 100·200줄, 앞 100시행 같음(중첩), |Δg| T100 < T200
# (4) 무학습 T100·T200: 새 항목 정답률 같음, |Δg| 0
# (5)(6) 2026-10-02 수정 회귀: main() 지역 변수 io → ino(모듈 io 가림 → --dump-rewards UnboundLocalError, 1차 경로 검사에서 발견).
#        이름을 바꾼 두 검사 줄(comparator·developed)이 E129 w10·E130 w10 원 로그와 같은 [KC배선]·[KC발달]·SDLAB 을 내는가
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E137/pathcheck"; WD="$R/research/experiments/traces/E137/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e137_run && cd /root/e137_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50C="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-exposures 400 --dev-mode oja --dev-oja-eta 0.005 --dev-oja-beta 5 --dev-oja-mmax-c 2.0 --dev-oja-mmax-i 4.0"
dsum() {  # 로그 → 요약(블록 줄 수, |Δg| l·r, SDLAB 새 항목·훈련)
  local f="$1"
  echo "블록 $(grep -c '^  시행 ' "$f") | dl $(sed -nE 's/^  kc_out_l: n=[0-9]+ [|]Δ[|]평균 ([0-9.]+).*/\1/p' "$f") dr $(sed -nE 's/^  kc_out_r: n=[0-9]+ [|]Δ[|]평균 ([0-9.]+).*/\1/p' "$f") | $(grep -oE 'train_lbal=[0-9.]+|novel_lbal=[0-9.]+' "$f" | tr '\n' ' ')"
}
echo "[1 회귀 K50 w0 t100 trials 400 block 400 — 기대 first=60.5 reward=60.5 eval=100.0]"
f="$OUT/regress_k50.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 0 --trial-seed 100 $K50C --block 400 --trials 400 > "$f" 2>&1; grep '^=> MINCIRC' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[2 Oja 발달 회귀 w18 corr — 기대(E136 보정) 가지치기 일치형 0.810 불일치형 0.920]"
f="$OUT/dev_corr_w18.log"; timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50C --block 100 --trials 1 $SDO $DEV --dev-env corr --dev-hebb-save "$WD/dev_corr_w18.npz" > "$f" 2>&1
grep -oE '가지치기 후 같은 위치: 일치형 [0-9.]+ 불일치형 [0-9.]+' "$f" || { echo "[실패]"; tail -3 "$f"; }
for MODE in learn frozen; do for T in 100 200; do
  f="$OUT/${MODE}_w18_T${T}_t600.log"; rw="$WD/rw_${MODE}_w18_T${T}_t600.txt"
  timeout 3600 python minimal_circuit.py --mode $MODE --seed 18 --trial-seed 600 $K50C --block 100 --trials $T $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/dev_corr_w18.npz" --dump-rewards "$rw" > "$f" 2>&1; rc=$?
  if grep -q '^=> SDLAB' "$f"; then echo "[3/4 $MODE T$T] $(dsum "$f")| 보상 계열 $(wc -l < "$rw")줄 | $(grep '^\[KC불러옴\]' "$f" | grep -oE '일치형 같은 위치 [0-9/]+ [|] 불일치형 같은 위치 [0-9/]+')"; else echo "[3/4 $MODE T$T 실패 rc=$rc]"; tail -3 "$f"; fi
done; done
a="$WD/rw_learn_w18_T100_t600.txt"; b="$WD/rw_learn_w18_T200_t600.txt"
if [ "$(sed -n '1,100p' "$b" | md5sum)" = "$(md5sum < "$a")" ]; then echo "[3 중첩] T100 보상 계열 = T200 앞 100시행: 같음"; else echo "[3 중첩] 다름"; fi
REF="$R/research/experiments/logs"
cmpref() {  # 새 로그 원 로그 패턴
  local x y; x=$(grep "$3" "$1"); y=$(grep "$3" "$2")
  if [ -n "$x" ] && [ "$x" = "$y" ]; then echo "    같음: $x" | cut -c1-200; else echo "    다름 — 새: $x"; echo "           원: $y"; fi
}
echo "[5 회귀 E129 comparator learn w10 t600(800시행) — 원 로그 cp_learn_w10_t600]"
f="$OUT/regress_e129_w10.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 10 --trial-seed 600 $K50C --block 400 --trials 800 --samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic --kc-wiring comparator --mismatch-w 8 > "$f" 2>&1
cmpref "$f" "$REF/E129/cp_learn_w10_t600.log" '^\[KC배선\]'; cmpref "$f" "$REF/E129/cp_learn_w10_t600.log" '^=> SDLAB'
echo "[6 회귀 E130 developed corr frozen w10 t600 — 원 로그 dv_corr_frozen_w10_t600]"
f="$OUT/regress_e130_w10.log"; timeout 3600 python minimal_circuit.py --mode frozen --seed 10 --trial-seed 600 $K50C --block 400 --trials 800 --samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic --kc-wiring developed --mismatch-w 8 --dev-theta 0.15 --dev-rounds 1000 --dev-exposures 200 --dev-items 20 --dev-env corr > "$f" 2>&1
cmpref "$f" "$REF/E130/dv_corr_frozen_w10_t600.log" '^\[KC발달\]'; cmpref "$f" "$REF/E130/dv_corr_frozen_w10_t600.log" '^=> SDLAB'
echo "[E137 경로 검사] 종료"
