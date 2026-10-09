#!/bin/bash
# E167 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 배선 18(E136 보정 배선, 본실험 78~93 아님).
# (a) 발달 재현: E136 선택 설정(η 0.005·β 5) 발달을 새 저장 코드로 다시 돌려 가지치기(a,b,e,i)가 E136 보정 저장본과 정확히 같은가 + 후보 전체 키 저장.
# (b) ojafull learn t600: '[KC망안]' 줄(후보 수·같은 위치 몫·장치 대조 최대 |차|), SDLAB. (c) ojafull frozen t600.
# (d) 참고(판정 아님): 같은 발달 저장본의 호스트 경로(loaded) learn t600 — 같은 배선의 H 값.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E167/pathcheck"; WD="$R/research/experiments/traces/E167/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e167_run && cd /root/e167_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-exposures 400 --dev-mode oja --dev-oja-eta 0.005 --dev-oja-beta 5 --dev-oja-mmax-c 2.0 --dev-oja-mmax-i 4.0"
W=18; F="$WD/dev_corr_w$W.npz"
f="$OUT/dev_corr_w$W.log"
timeout 3600 python minimal_circuit.py --mode frozen --seed $W --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env corr --dev-hebb-save "$F" > "$f" 2>&1
echo "[(a) 발달 rc=$?] $(grep -oE '가지치기 후 같은 위치: 일치형 [0-9.]+ 불일치형 [0-9.]+' "$f") | $(grep '^\[OJA발달\]' "$f" | cut -c1-140)"
cd $R && python3 - <<'PY'
import numpy as np
z7 = np.load("research/experiments/traces/E167/pathcheck/dev_corr_w18.npz"); z6 = np.load("research/experiments/traces/E136/calib_eta0.005_b5_w18_corr.npz")
same = {k: bool(np.array_equal(z7[k], z6[k])) for k in ("a", "b", "e", "i")}
print("    가지치기 = E136 보정 저장본: %s | 후보 키 %s | 흥분 후보 %d·억제 후보 %d" % (same, sorted(set(z7.files) - {"a", "b", "e", "i"}), z7["cpre"].size, z7["ipre"].size))
PY
cd /root/e167_run
for MODE in learn frozen; do
  f="$OUT/N_${MODE}_w${W}_t600.log"
  timeout 3600 python minimal_circuit.py --mode $MODE --seed $W --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring ojafull --kc-wiring-file "$F" > "$f" 2>&1
  echo "[(N $MODE) rc=$?] $(grep '^\[KC망안\]' "$f" | sed 's/^.*npz | //') || $(grep '^=> SDLAB' "$f" | grep -oE 'mode=[a-z]+|train_lbal=[0-9.]+|novel_lbal=[0-9.]+' | tr '\n' ' ')"
done
f="$OUT/H_learn_w${W}_t600.log"
timeout 3600 python minimal_circuit.py --mode learn --seed $W --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$F" > "$f" 2>&1
echo "[(d) 참고 H learn rc=$?] $(grep '^\[KC불러옴\]' "$f" | sed 's/^.*npz | //') || $(grep '^=> SDLAB' "$f" | grep -oE 'train_lbal=[0-9.]+|novel_lbal=[0-9.]+' | tr '\n' ' ')"
echo "[E167 경로 검사] 종료"
