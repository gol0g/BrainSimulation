#!/bin/bash
# [기록 2026-10-25 22:43] 이 진단 뒤 임시 옵션 --kc-homeo-group 은 기각되어 코드에서 되돌렸다(3582b09 판) — 이 스크립트는 그 진단의 기록으로만 남긴다(지금 코드로는 'on' 이 인자 오류).
# E209 개발 진단(등록 전 — 기전 확인, 학습 없음) — 뇌 15(표본 밖), E208 경로 검사 (c) c0 측정 커맨드 그대로 두 번:
#   off = 무리별 항상성 끔(E208 c0 와 같아야 — 회귀: '[E207 측정 맥락]' 발화·RCJ 가 logs/E208/pathcheck/c0_meas_b15.log 와 같음)
#   on  = --kc-homeo-group(활성 무리 KC 발화·사례당 반응 KC 가 약 절반이어야, 단계 수 줄 '켜진 무리 기준' > 0)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev && cd /root/e209_dev
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단] 시작 $(date '+%F %T')"
for spec in "off -" "on --kc-homeo-group"; do
  set -- $spec; N=$1; HG=$2; [ "$HG" = "-" ] && HG=""
  g="$OUT/probe_${N}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE $HG --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $F15 --rc-measure-ctx-on X > "$g" 2>&1; rc=$?
  echo "[($N) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · $(grep -E '^\s*\[E209 무리별 항상성\]' "$g" | tr '\n' ' ')"
  grep '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  grep -E '^RCJ side=[lr] pair=(AB|AC|BD)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단] 종료 $(date '+%F %T')"
