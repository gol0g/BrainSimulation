#!/bin/bash
# E200 재현 변동 점검(보고용 — 기준 정정 2, 본실험 결과 전): e200_premise.sh 가 다시 학습한 E199 b10 가중치(같은 커맨드·시드 — 배열은 원본과 다름)를
# E199 와 같은 이식 평가(learn × 맥락 X·Y)로 재서 원래 E199 b10 칸 값(logs/E199/ev_b10_learn_{X,Y}.log)과 비교한다 — 실행 간 행동 변동의 크기. 판정에 쓰지 않는다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E200/premise"; WD="$R/research/experiments/traces/E200/premise"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e200_premise_run && cd /root/e200_premise_run
cp $R/backend/genesis/*.py . 2>/dev/null
B=10
EBASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
echo "[E200 재현 변동 점검] 시작 $(date '+%F %T')"
for X in X Y; do
  g="$OUT/ev_e199rerun_b${B}_learn_$X.log"
  timeout 3600 python reflex_override_task.py $EBASE $ACT $FEAT $CTX --brain-seed $B --kc-type-weights $KW --decomp-weights $WD/w_e199_b$B.npz --decomp-mode all --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "  재현 b$B learn $X rc=$rc: $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"
  echo "  원래 E199: $(grep -E "e199 b$B learn $X:" $R/research/experiments/E199.log | sed 's/.*=> //')"
done
echo "[E200 재현 변동 점검] 종료 $(date '+%F %T')"
