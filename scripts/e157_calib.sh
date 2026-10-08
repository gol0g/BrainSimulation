#!/bin/bash
# E157 경로 검사(조건 2)·k 보정 — 기준 고정(logs/E157/criteria_fixed.txt 01:06:04) 뒤. 표본 밖 뇌 15.
# [1] 재현: E141 경로 검사(기본, scale0 — [사전] +0.0255 [사후] −0.2103), E153 경로 검사(형성 1ep — +0.0125 → −0.1611). 맞으면 D·F 원 로그 재사용.
# [2] 배율 학습 경로: Fk(형성 × 0.8)·Dk(기본 × 1.2) 1ep — 적재·배율 검증 줄(학습·이식 평가 뇌).
# [3] kcrate(E156 진단과 같은 인자): k=1 재현(기본 48,856 · 형성 61,313) + 격자 Fk {0.70..0.90}·Dk {1.10..1.50} → e157_kstar.py 가 k 선택.
# [4] 고른 k 의 kcoverlap(겹침 경로·배율 줄).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E157/calib"; WD="$R/research/experiments/traces/E157/calib"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e157_run && cd /root/e157_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
KW="$R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz"
RATE="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'
mod() { echo "$(grep '^\[사전\]' "$1" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$1" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$1")"; }
# [1] 재현
f="$OUT/repro_D_b15.log"
timeout 7200 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
  --save-weights $WD/w_D_b15.npz --trace-kc-class $WD/tr_D_b15.npz --kc-rate-file $RATE > "$f" 2>&1; rc=$?
echo "[재현 D rc=$rc] $(mod "$f") | 기대 +0.0255 → -0.2103 | 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f")"
f="$OUT/repro_F_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW \
  --save-weights $WD/w_F_b15.npz --trace-kc-class $WD/tr_F_b15.npz --kc-rate-file $RATE > "$f" 2>&1; rc=$?
echo "[재현 F rc=$rc] $(mod "$f") | 기대 +0.0125 → -0.1611 | 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f")"
# [2] 배율 학습 경로
for A in Fk Dk; do
  f="$OUT/learn_${A}_b15.log"
  if [ "$A" = "Fk" ]; then XA="--kc-type-weights $KW --kc-type-scale 0.8"; else XA="--kc-type-scale 1.2"; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed 15 $XA \
    --save-weights $WD/w_${A}_b15.npz --trace-kc-class $WD/tr_${A}_b15.npz --kc-rate-file $RATE > "$f" 2>&1; rc=$?
  echo "[배율 학습 $A rc=$rc] $(mod "$f") | 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep -m1 "$SC" "$f" | cut -c1-200)"
done
# [3] kcrate — E156 진단과 같은 인자(반사 25, --episodes 0, --trials 100)
KR() {  # $1 태그 $2 추가 인자
  local f="$OUT/kcrate_$1_b15.log"
  timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 25 --episodes 0 --steps 100 --brain-seed 15 --rw-apm-scale 0 $2 --trials 100 \
    --decomp-weights $R/research/experiments/traces/E156/pathcheck/w_F200_b15.npz --decomp-mode kcrate --kc-rate-file $WD/kcrate_$1_b15.npz > "$f" 2>&1
  echo "[kcrate $1 rc=$?] 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep '^=> KCRATE' "$f" | grep -oE '^=> KCRATE kc_[lr]|제시 스파이크 [0-9]+' | tr '\n' ' ')"
}
KR D_k1.00 ""
KR F_k1.00 "--kc-type-weights $KW"
for K in 0.70 0.75 0.80 0.85 0.90; do KR Fk_k$K "--kc-type-weights $KW --kc-type-scale $K"; done
for K in 1.10 1.20 1.30 1.40 1.50; do KR Dk_k$K "--kc-type-scale $K"; done
cd $R && python3 scripts/e157_kstar.py
cd /root/e157_run
# [4] 고른 k 의 겹침(경로) — 보정 실패면 건너뜀
KF=$(grep -oE 'Fk k=[0-9.]+' $R/research/experiments/logs/E157/kstar.txt | cut -d= -f2)
KD=$(grep -oE 'Dk k=[0-9.]+' $R/research/experiments/logs/E157/kstar.txt | cut -d= -f2)
if [ -n "$KF" ] && [ -n "$KD" ]; then
  for A in Fk Dk; do
    f="$OUT/ov_${A}_b15.log"
    if [ "$A" = "Fk" ]; then XA="--kc-type-weights $KW --kc-type-scale $KF"; else XA="--kc-type-scale $KD"; fi
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed 15 --decomp-weights $R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz --decomp-mode kcoverlap --trials 200 $XA > "$f" 2>&1
    echo "[겹침 $A rc=$?] 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')"
  done
else
  echo "[겹침] 보정 실패 — 건너뜀"
fi
echo "[E157 경로 검사·보정] 종료"
