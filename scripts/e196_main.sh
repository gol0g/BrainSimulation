#!/bin/bash
# E196 본실험 — 기준 logs/E196/criteria_fixed.txt. L4 병목 위치(이상 가중치, 학습 없음): 뇌 10~14 × 팔 none·cat·sel·eq × 맥락 X·Y(E193 과 같은 평가).
# 지수 파일 traces/E196/sxi_b{B}.npz = scripts/kcxor_build.py(뇌마다 새로 잰 logs/E196/meas_b{B}_T4900.log — 정정 1: 경로 검사 (a) 에서 KC 발화가 기저값에 따라 달라
# 사전 규칙대로 E195 S1(−4000) 대신 E195 S1 과 같은 측정 커맨드를 motor 기저값 −4900(평가 동작점)으로). 요약 줄 "  e196 ... : => ..." 재개 가능(P13). 50줄(측정 5 + 지수 5 + 평가 40).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E196.log"
RAW="$R/research/experiments/logs/E196"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E196"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e196_main_run && cd /root/e196_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"   # E193 동작점(logs/E193/calib.out)
echo "[E196] L4 병목 위치 — 이상 KC→motor 가중치(none·cat·sel·eq) × 맥락 X·Y · C2 · 기저값 −4900 · 뇌 10~14"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  M="$RAW/meas_b${B}_T4900.log"
  tag="e196 meas b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    printf "  %s: " "$tag"
    # 정정 1: E195 S1 측정(scripts/e195_main.sh, 배율 1)과 같은 커맨드 — CTX 의 motor 기저값만 −4900
    timeout 7200 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $FEAT $CTX --kc-inh-scale 1 \
      --decomp-weights $R/research/experiments/traces/E179/w_bc_b$B.npz --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 3 > "$M" 2>&1; rc=$?
    if grep -q "^RCKXCHK " "$M"; then echo "=> RCKX $(grep -c '^RCKX ' "$M") || $(grep '^RCKXCHK' "$M" | grep -oE 'npres=[0-9]+|ctx_on_spikes=[0-9]+|ctx_off_spikes=[0-9]+|feat_inactive_spikes=[0-9]+|g_same=[0-9]+' | tr '\n' ' ')"
    else echo "[실패 rc=$rc]"; tail -2 "$M" | sed 's/^/      /'; fi
  fi
  tag="e196 sxi b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    printf "  %s: " "$tag"
    if grep -q "^RCKXCHK " "$M" && [ "$(grep -c '^RCKX ' "$M")" = "288" ] && python3 $R/scripts/kcxor_build.py "$M" $WD/sxi_b$B.npz > $RAW/sxi_b$B.out 2>&1; then
      echo "=> $(grep -oE '쪽 [lr] \| 반응 [0-9]+ · SXI ≥ 0.5 [0-9]+ · ≤ −0.5 [0-9]+ · CSI ≥ 0.5 [0-9]+ · ≤ −0.5 [0-9]+' $RAW/sxi_b$B.out | tr '\n' ' ')"
    else echo "[실패 rc=지수]"; tail -2 $RAW/sxi_b$B.out 2>/dev/null | sed 's/^/      /'; fi
  fi
  for W in none cat sel eq; do
    for X in X Y; do
      tag="e196 b$B $W $X"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      [ -s "$WD/sxi_b$B.npz" ] || { echo "  $tag: [실패 rc=지수 파일 없음]"; continue; }
      if [ "$W" = "none" ]; then XM="--decomp-mode none"; else XM="--decomp-mode kcxor --kc-xor-file $WD/sxi_b$B.npz --kc-xor-arm $W"; fi
      g="$RAW/ev_b${B}_${W}_$X.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed $B --kc-type-weights $KW --decomp-weights $R/research/experiments/traces/E193/w_b$B.npz $XM --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
      if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
    done
  done
done
echo "[E196] 전체 루프 종료"
