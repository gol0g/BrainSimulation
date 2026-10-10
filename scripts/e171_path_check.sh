#!/bin/bash
# E171 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15(E170 경로 검사와 같은 가중치).
# --eval-diag 가 평가값을 바꾸지 않는가(AB base·agree, none agree 의 변조폭 = E170 경로 검사 −0.4783·−0.3423·−0.0039) + 진단 줄 2개·n=250·값이 0/nan 아님.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E171/pathcheck"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E162/pathcheck"; KW="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e171_run && cd /root/e171_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
B=15
for WS in "AB base" "AB agree" "none agree"; do
  set -- $WS; W=$1; S=$2
  [ "$W" = "AB" ] && X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" || X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode none"
  f="$OUT/ev_b${B}_${W}_$S.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S --eval-diag > "$f" 2>&1
  echo "[$W $S rc=$?] $(grep '^=> DECOMP' "$f" | grep -oE 'mod=[-+0-9.]+')"
  grep '^\[E171 평가 진단\]' "$f" | sed 's/^/    /'
done
echo "  기대: AB base mod=-0.4783 · AB agree mod=-0.3423 · none agree mod=-0.0039(E170 경로 검사), 진단 줄 2개 n=250"
echo "[E171 경로 검사] 종료"
