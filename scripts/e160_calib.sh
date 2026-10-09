#!/bin/bash
# E160 경로 검사(조건 2)·보정 — 기준 고정(logs/E160/criteria_fixed.txt 15:01:00) 뒤. 표본 밖 뇌 15.
# (a) Oja 꺼짐 기본 kcrate 총수 = 48,856(E156·E157 — 기본 모델 불변). (b) kcdevoja 격자 η × β 12런 → e160_pick.py.
# (c) 고른 칸 가중치를 정적 뇌에 적재한 kcoverlap(post 인덱스 일치 검증·자카드). (d) 같은 가중치로 학습 1ep(적재 2줄).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E160/calib"; WD="$R/research/experiments/traces/E160/calib"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e160_run && cd /root/e160_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
# (a) 기본 모델 불변
f="$OUT/kcrate_default_b15.log"
timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 25 --episodes 0 --steps 100 --brain-seed 15 --rw-apm-scale 0 --trials 100 \
  --decomp-weights $R/research/experiments/traces/E156/pathcheck/w_F200_b15.npz --decomp-mode kcrate --kc-rate-file $WD/kcrate_default_b15.npz > "$f" 2>&1
echo "[기본 kcrate rc=$?] $(grep '^=> KCRATE' "$f" | grep -oE 'kc_[lr] \||제시 스파이크 [0-9]+' | tr '\n' ' ') | Oja 줄 $(grep -c '^\[E160 종류 입력 Oja\]' "$f") (기대 23372+25484=48856, Oja 0줄)"
# (b) 격자
for E in 0.005 0.02 0.08; do for B in 0.1 0.3 1.0 3.0; do
  f="$OUT/oja_e${E}_b${B}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed 15 --decomp-weights $R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz --decomp-mode kcdevoja \
    --kc-dev-n 100 --kc-type-oja --kc-oja-eta $E --kc-oja-beta $B --kc-oja-mmax 32 --kc-oja-tau 20 --kc-dev-save $WD/oja_e${E}_b${B}_b15.npz > "$f" 2>&1; rc=$?
  echo "[η$E β$B rc=$rc] Oja 줄 $(grep -c '^\[E160 종류 입력 Oja\]' "$f") | $(grep '^=> KCDEVOJA' "$f" | cut -c1-300)"
  [ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
done; done
cd $R && python3 scripts/e160_pick.py
cd /root/e160_run
PE=$(grep -oE '^eta=[0-9.]+' $R/research/experiments/logs/E160/oja_pick.txt | cut -d= -f2)
PB=$(grep -oE 'beta=[0-9.]+' $R/research/experiments/logs/E160/oja_pick.txt | cut -d= -f2)
if [ -n "$PE" ] && [ -n "$PB" ]; then
  KW="$WD/oja_e${PE}_b${PB}_b15.npz"
  f="$OUT/ov_pick_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed 15 --decomp-weights $R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1
  echo "[겹침 rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')"
  f="$OUT/learn1_pick_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW \
    --save-weights $WD/w_learn1_b15.npz --trace-kc-class $WD/tr_learn1_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1
  echo "[학습 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
else
  echo "[겹침·학습] 보정 실패 — 건너뜀"
fi
echo "[E160 경로 검사·보정] 종료"
