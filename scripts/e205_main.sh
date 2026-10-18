#!/bin/bash
# E205 본실험 — 기준 logs/E205/criteria_fixed.txt. L4 맥락 게이팅 확증(새 뇌 16~20·세계 38300·새 시행열, 게이팅 GW 대 없음 UW) + 같은 고정 체계의 앞 능력(R 범주 규칙·V 반전·P 유지, 맥락 X 고정).
# 고정 체계 = E204 그대로(게이팅 −20·p 0.10·반반, T_G −2800, 경쟁형 규칙 η 2.4, 도파민 기준선, 흔적 동결, C2). 학습 env 2·배정 38500+뇌·맥락 38600+뇌, 평가 env 0, 세계 --rc-seed 38300(학습·평가).
# 뇌마다: GW 학습 + 평가 4(learn·none × X·Y) · UW 학습 + 평가 4 · R·V·P 학습 · 평가 R_learn(A~D)·V_learn(A·B)·P_learn(A~D)·none(A~D, 맥락 X) = 17줄 → 85줄.
# 요약 줄 "  e205 ... : => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E205.log"
RAW="$R/research/experiments/logs/E205"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E205"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
TG=-2800
WS=38300
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e205_main_run && cd /root/e205_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
LRN="--kc-motor-hard --kc-motor-comp --da-baseline"
echo "[E205] L4 게이팅 확증(GW 대 UW) + 앞 능력(R·V·P, 맥락 X 고정) · 고정 체계 E204(게이팅·T_G $TG·경쟁형 η 2.4·기준선) · 세계 $WS · 학습 env 2 · 뇌 16~20"
train() {   # $1 tag  $2 log  $3 weights  $4.. 추가 인자 — 요약 줄 출력
  local tag="$1" f="$2" w="$3"; shift 3
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $B0 --kc-motor-eta 2.4 --env-seed 2 $ACT $FEAT $CW --rc-seed $WS --rc-trial-seed $((38500 + B)) "$@" \
    --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW --save-weights "$w" $LRN > "$f" 2>&1; local rc=$?
  if grep -q "^\[학습\]" "$f" && grep -q "^\[E199 경쟁 정규화\] 적용" "$f" && grep -q "^\[E200 도파민 통계\]" "$f" && grep -q "^\[범주 과제\] 시행" "$f" && [ -s "$w" ]; then
    echo "=> $(grep '^\[맥락 범주 과제\] 시행\|^\[범주 과제\] 시행' "$f" | tail -1) || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || $(grep '^\[E200 도파민 통계\]' "$f" | grep -oE '평균 DA [-+0-9.]+' | head -1) || 게이팅 $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f" | grep -oE '비\(비활성/활성\) [0-9.]+' || echo 없음) || $(grep '^\[범주 규칙 맥락\] 시행' "$f" || echo '맥락 고정 없음')$(grep '^\[범주 반전\] 반전 시행\|^\[둘째 범주 쌍\] 둘째 쌍 시행' "$f" | sed 's/^/ || /')"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
evalw() {   # $1 tag  $2 log  $3 weights  $4 mode  $5.. 추가 인자
  local tag="$1" g="$2" w="$3" xm="$4"; shift 4
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  [ -s "$w" ] || { echo "  $tag: [실패 rc=학습 가중치 없음]"; return; }
  printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $B0 --kc-motor-eta 0.15 --env-seed 0 $ACT $FEAT $CW --rc-seed $WS --brain-seed $B --kc-type-weights $KW \
    --decomp-weights "$w" --decomp-mode $xm --rc-eval --trials 96 --eval-diag "$@" > "$g" 2>&1; local rc=$?
  if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
}
for B in 16 17 18 19 20; do
  KW="$R/research/experiments/traces/E161/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  [ -s "$SNAP/conn_b$B.npz" ] && [ -s "$KW" ] || { echo "  e205 b$B: [실패 rc=스냅숏·종류 가중치 없음]"; continue; }
  for ARM in GW UW; do
    [ "$ARM" = "GW" ] && { EX="$GATE"; T="$TG"; } || { EX=""; T="-4900"; }
    train "e205 $ARM train b$B" "$RAW/train_${ARM}_b$B.log" "$WD/w_${ARM}_b$B.npz" $EX --motor-tonic $T --rc-ctx-seed $((38600 + B)) --rc-task catctx --episodes 30
    for W in learn none; do
      [ "$W" = "learn" ] && XM=all || XM=none
      for X in X Y; do
        evalw "e205 $ARM b$B $W $X" "$RAW/ev_${ARM}_b${B}_${W}_$X.log" "$WD/w_${ARM}_b$B.npz" $XM $EX --motor-tonic $T --rc-eval-ctx $X
      done
    done
  done
  train "e205 R train b$B" "$RAW/train_R_b$B.log" "$WD/w_R_b$B.npz" $GATE --motor-tonic $TG --rc-rule-ctx X --rc-task catrule --episodes 15
  train "e205 V train b$B" "$RAW/train_V_b$B.log" "$WD/w_V_b$B.npz" $GATE --motor-tonic $TG --rc-rule-ctx X --rc-task catrule --rc-reverse-after 1500 --episodes 30
  train "e205 P train b$B" "$RAW/train_P_b$B.log" "$WD/w_P_b$B.npz" $GATE --motor-tonic $TG --rc-rule-ctx X --rc-task catrule --rc-phase2-after 1500 --episodes 30
  evalw "e205 b$B R_learn" "$RAW/ev_b${B}_R_learn.log" "$WD/w_R_b$B.npz" all $GATE --motor-tonic $TG --rc-eval-ctx X --rc-eval-cats A,B,C,D
  evalw "e205 b$B V_learn" "$RAW/ev_b${B}_V_learn.log" "$WD/w_V_b$B.npz" all $GATE --motor-tonic $TG --rc-eval-ctx X --rc-eval-cats A,B
  evalw "e205 b$B P_learn" "$RAW/ev_b${B}_P_learn.log" "$WD/w_P_b$B.npz" all $GATE --motor-tonic $TG --rc-eval-ctx X --rc-eval-cats A,B,C,D
  evalw "e205 b$B none" "$RAW/ev_b${B}_none.log" "$WD/w_P_b$B.npz" none $GATE --motor-tonic $TG --rc-eval-ctx X --rc-eval-cats A,B,C,D
done
echo "[E205] 전체 루프 종료"
