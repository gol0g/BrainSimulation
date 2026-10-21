#!/bin/bash
# E207 본실험(측정, 학습 없음) — 기준 logs/E207/criteria_fixed.txt. 형성 표현 F(·U)의 범주 KC 집합을 측정 제시 동안 맥락 none 대 X 로(C2·게이팅·기저값은 학습과 같게).
# 뇌마다 U·none, U·X, F·none, F·X 측정 4 → 20줄. 요약 줄 "  e207 ... : => side=l pair=AC=J_pre/J_post ..." 재개 가능(P13). 정정 1(2026-10-22): 요약에 두 측정(첫·둘째) 모두.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E207.log"
RAW="$R/research/experiments/logs/E207"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e207_main_run && cd /root/e207_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
echo "[E207] 학습 조건의 범주 KC 겹침(측정) · U·F × 맥락 none·X · C2·게이팅·기저값 U −2800·F −1300 · 세계 38300 · 뇌 16~20"
for B in 16 17 18 19 20; do
  KW="$R/research/experiments/traces/E161/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  FF="$R/research/experiments/traces/E206/feat_formed_b$B.npz"
  W15="$R/research/experiments/traces/E185/w_1500_b$B.npz"
  [ -s "$SNAP/conn_b$B.npz" ] && [ -s "$KW" ] && [ -s "$FF" ] && [ -s "$W15" ] || { echo "  e207 b$B: [실패 rc=입력 없음]"; continue; }
  for REP in U F; do
    for C in none X; do
      tag="e207 $REP $C b$B"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      [ "$REP" = "F" ] && { EX="--feat-weights $FF"; T=-1300; } || { EX=""; T=-2800; }
      [ "$C" = "X" ] && EX="$EX --rc-measure-ctx-on X"
      g="$RAW/meas_${REP}_${C}_b$B.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --motor-tonic $T --brain-seed $B --kc-type-weights $KW \
        --decomp-weights $W15 --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300 $EX > "$g" 2>&1; rc=$?
      if grep -q "^=> RAWCAT " "$g" && [ "$(grep -c '^RCJ ' "$g")" = "12" ]; then
        echo "=> $(grep -E '^RCJ side=[lr] pair=(AB|AC|BD)' "$g" | grep -oE 'side=[lr] pair=[A-D]+ J_pre=[0-9.]+ J_post=[0-9.]+' | sed -E 's/ J_pre=([0-9.]+) J_post=([0-9.]+)/=\1\/\2/' | tr '\n' ' ')$(grep '^\[E207 측정 맥락\] tag=post' "$g" | grep -oE '켠 맥락 발화 [0-9]+ · 끈 맥락 발화 [0-9]+ · KC 발화 활성 무리 [0-9]+ · 비활성 무리 [0-9]+' | sed 's/^/|| /')"
      else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
    done
  done
done
echo "[E207] 전체 루프 종료"
