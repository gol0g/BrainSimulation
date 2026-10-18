#!/bin/bash
# E204 본실험 — 기준 logs/E204/criteria_fixed.txt. 맥락 게이팅 KC 무리(G1): E200 과 같은 학습(경쟁형 규칙 η 2.4 + 도파민 기준선, 활동 가중 정규화 없음).
# 팔: G18 게이팅·세계 18300·기본 시행열 / GN 게이팅·새 원형 세계 28300·새 시행열(env 1·배정 28500+뇌·맥락 28600+뇌) / UN 게이팅 없음(기저값 −4900)·GN 과 같은 새 세계.
# 게이팅 팔 기저값 = logs/E204/calib_tonic.txt 의 T_G(보정 규칙 — 결과 전). 평가 = E193 평가 커맨드(env 0) + 팔의 게이팅·세계·기저값.
# 요약 줄 "  e204 <팔> ... : => ..." 재개 가능(P13). 75줄(팔 3 × 뇌 5 × (학습 1 + 평가 4)).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E204.log"
RAW="$R/research/experiments/logs/E204"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E204"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
ETA=2.4
TG=$(grep -oE '^T_G=-[0-9]+' "$RAW/calib_tonic.txt" 2>/dev/null | cut -d= -f2)
[ -n "$TG" ] || { echo "[E204] 보정 기저값 없음(logs/E204/calib_tonic.txt) — 종료"; exit 1; }
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e204_main_run && cd /root/e204_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
echo "[E204] 맥락 게이팅 KC 무리 · E200 학습(경쟁형 규칙·η $ETA·도파민 기준선) · catctx 3,000 · C2 · 게이팅 기저값 $TG · 팔 G18·GN·UN · 뇌 10~14"
for ARM in G18 GN UN; do
  for B in 10 11 12 13 14; do
    case $ARM in
      G18) EX="$GATE"; T="$TG"; ENV=0; WSEED=""; ;;
      GN)  EX="$GATE"; T="$TG"; ENV=1; WSEED="--rc-seed 28300"; ;;
      UN)  EX="";      T="-4900"; ENV=1; WSEED="--rc-seed 28300"; ;;
    esac
    TSEQ=""; [ "$ARM" != "G18" ] && TSEQ="--rc-trial-seed $((28500 + B)) --rc-ctx-seed $((28600 + B))"
    KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
    FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
    tag="e204 $ARM train b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
      f="$RAW/train_${ARM}_b$B.log"; printf "  %s: " "$tag"
      timeout 14400 python reflex_override_task.py $B0 --kc-motor-eta $ETA --env-seed $ENV $ACT $FEAT $CW --motor-tonic $T $EX $WSEED $TSEQ --rc-task catctx --episodes 30 --steps 100 --transplant-eval \
        --brain-seed $B --kc-type-weights $KW --save-weights $WD/w_${ARM}_b$B.npz --kc-motor-hard --kc-motor-comp --da-baseline > "$f" 2>&1; rc=$?
      if grep -q "^\[맥락 범주 과제\] 시행" "$f" && grep -q "^\[학습\]" "$f" && grep -q "^\[E199 경쟁 정규화\] 적용" "$f" && grep -q "^\[E200 도파민 통계\]" "$f" && [ -s "$WD/w_${ARM}_b$B.npz" ]; then
        echo "=> $(grep '^\[맥락 범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || $(grep '^\[E200 도파민 통계\]' "$f" | grep -oE '평균 DA [-+0-9.]+' | head -1) || 게이팅 $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f" | grep -oE '비\(비활성/활성\) [0-9.]+' || echo 없음)"
      else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    fi
    for W in learn none; do
      for X in X Y; do
        tag="e204 $ARM b$B $W $X"
        if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
        [ -s "$WD/w_${ARM}_b$B.npz" ] || { echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; }
        [ "$W" = "learn" ] && XM=all || XM=none
        g="$RAW/ev_${ARM}_b${B}_${W}_$X.log"; printf "  %s: " "$tag"
        timeout 3600 python reflex_override_task.py $B0 --kc-motor-eta 0.15 --env-seed 0 $ACT $FEAT $CW --motor-tonic $T $EX $WSEED --brain-seed $B --kc-type-weights $KW \
          --decomp-weights $WD/w_${ARM}_b$B.npz --decomp-mode $XM --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
        if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
      done
    done
  done
done
echo "[E204] 전체 루프 종료"
