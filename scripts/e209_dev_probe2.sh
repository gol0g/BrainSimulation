#!/bin/bash
# E209 개발 진단 2(등록 전 — 기전 확인, 학습 없음) — 뇌 15(표본 밖): 게이팅이 KC 절반을 꺼도 사례당 반응 KC 가 그대로인 까닭이
# 스파이크 되먹임 억제 고리(KC → KC 억제 뉴런 → KC — 총량 보존)인지. 2 × 2: {맥락 없음, 흥분 0·게이팅·맥락 X} × {되먹임 억제 배율 1, 0.01(사실상 끔)}.
# 예측(되먹임 보상이면): 배율 1 에서 X/없음 ≈ 1(E208 뇌 16~20 의 약 0.93), 배율 0.01 에서 X/없음 ≈ 0.5(꺼진 무리 몫을 보상하지 못함).
# 읽기: logs/E209/dev2/p_{none|X}_s{1|001}_b15.log 의 RCKC tag=post set=train 줄 n 평균(쪽·사례마다).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev2"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev2 && cd /root/e209_dev2
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단 2] 시작 $(date '+%F %T')"
for spec in "none 1 1.0" "X 1 1.0" "none 001 0.01" "X 001 0.01"; do
  set -- $spec; C=$1; SN=$2; SC=$3
  [ "$C" = "X" ] && EX="--rc-measure-ctx-on X" || EX=""
  g="$OUT/p_${C}_s${SN}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $F15 --kc-inh-scale $SC $EX > "$g" 2>&1; rc=$?
  echo "[($C·배율 $SC) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · 사례당 반응 KC(post·train) $(grep '^RCKC tag=post' "$g" | grep 'set=train' | grep -oE ' n=[0-9]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f(n=%d)", s/n, n; else print "없음"}') · $(grep -E '^\[E195 KC 되먹임 억제\]' "$g" | head -1)"
  grep '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단 2] 종료 $(date '+%F %T')"
