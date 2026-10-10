#!/bin/bash
# E170 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15, E162 경로 검사 가중치(w_AB_b15: 과제 A 200 + B 200)·형성 가중치(E160 보정 β 0.3).
# (a) 회귀: AB·none × base·bad 가 E162 경로 검사 값(−0.4783, +0.5105, +0.0228, +0.0120)을 정확히 재현 — 평가 경로가 새 변형 코드로 바뀌지 않았는가.
# (b) 새 변형: AB·none × agree·conflict — '[E170 자극]' 줄이 설계 구성(쪽별 good·bad·먹이 광선)과 같은가, 평가 완료.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E170/pathcheck"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E162/pathcheck"; KW="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e170_run && cd /root/e170_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
B=15
for W in AB none; do
  [ "$W" = "AB" ] && X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" || X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode none"
  for S in base bad agree conflict; do
    f="$OUT/ev_b${B}_${W}_$S.log"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S > "$f" 2>&1
    echo "[$W $S rc=$?] $(grep '^=> DECOMP' "$f" | grep -oE 'mode=[a-z]+ mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"
    grep '^\[E170 자극\]' "$f" | sed 's/^/    /'
  done
done
echo "  기대 (a): AB base −0.4783 · AB bad +0.5105 · none base +0.0228 · none bad +0.0120(E162 경로 검사)"
echo "  기대 (b): agree left good 0.90/0.00 bad 0.00/0.90 food 0.90/0.90, right 대칭 / conflict left good 0.90/0.00 bad 0.90/0.00 food 0.90/0.00, right 대칭"
echo "[E170 경로 검사] 종료"
