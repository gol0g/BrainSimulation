#!/bin/bash
# E161 본실험 — 기준 logs/E161/criteria_fixed.txt. 쓰지 않은 뇌 16~20, 망 안 Oja 형성 설정 고정(η 0.02·β 0.3·m_max 32·τ 20·노출 100). 재개 가능(P13).
# 뇌마다: w0(0시행 이식 가중치) → rate(kcrate 200) → D(기본 학습 500, 추적) → dev(kcdevoja) → ov(kcoverlap) → F(형성 학습 500, 추적) → kcF·kcD(발화, 부지표).
# 요약 줄: "  e161 {w0|rate|D|dev|ov|F|kcF|kcD} b16: => ..." (판정은 원 로그를 직접 읽는다)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E161.log"
RAW="$R/research/experiments/logs/E161"; WD="$R/research/experiments/traces/E161"; mkdir -p "$RAW" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e161_main_run && cd /root/e161_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
OJA="--kc-type-oja --kc-oja-eta 0.02 --kc-oja-beta 0.3 --kc-oja-mmax 32 --kc-oja-tau 20"
LD='^\[E153 종류 입력 적재\].*검증 일치'
skip() { grep -qF "$1: =>" "$LOG" 2>/dev/null && { echo "  $1: [건너뜀]"; return 0; }; return 1; }
fail() { echo "[실패 rc=$1]"; tail -2 "$2" | sed 's/^/      /'; }
modsum() { echo "사전 $(grep '^\[사전\]' "$1" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$1" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$1" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$1")"; }
for B in 16 17 18 19 20; do
  KW="$WD/kctype_oja_b$B.npz"
  tag="e161 w0 b$B"; if ! skip "$tag"; then f="$RAW/w0_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 0 --steps 100 --transplant-eval --brain-seed $B --save-weights $WD/w0_b$B.npz > "$f" 2>&1; rc=$?
    if [ -f "$WD/w0_b$B.npz" ] && grep -q "^\[저장\]" "$f"; then echo "=> $(grep '^\[저장\]' "$f" | cut -c1-60)"; else fail $rc "$f"; continue; fi; fi
  tag="e161 rate b$B"; if ! skip "$tag"; then f="$RAW/rate_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 0 --episodes 0 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcrate --trials 200 --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1; rc=$?
    if [ "$(grep -c '^=> KCRATE' "$f")" = "2" ]; then echo "=> KCRATE 2줄 $(grep -oE '제시 스파이크 [0-9]+' "$f" | tr '\n' ' ')"; else fail $rc "$f"; continue; fi; fi
  tag="e161 D b$B"; if ! skip "$tag"; then f="$RAW/D_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT2 --episodes 5 --steps 100 --transplant-eval --brain-seed $B \
      --save-weights $WD/w_D_b$B.npz --trace-kc-class $WD/tr_D_b$B.npz --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then echo "=> $(modsum "$f")"; else fail $rc "$f"; fi; fi
  tag="e161 dev b$B"; if ! skip "$tag"; then f="$RAW/dev_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcdevoja --kc-dev-n 100 $OJA --kc-dev-save $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCDEVOJA" "$f"; then echo "=> $(grep '^=> KCDEVOJA' "$f" | grep -oE 'side=[lr] fired=[0-9]+|sel_med0=[0-9.]+|sel_med=[0-9.]+|sum_med=[0-9.]+' | tr '\n' ' ')|| Oja $(grep -c '^\[E160 종류 입력 Oja\]' "$f")"; else fail $rc "$f"; continue; fi; fi
  tag="e161 ov b$B"; if ! skip "$tag"; then f="$RAW/ov_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCOVERLAP" "$f"; then echo "=> $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')|| 적재 $(grep -c "$LD" "$f")"; else fail $rc "$f"; fi; fi
  tag="e161 F b$B"; if ! skip "$tag"; then f="$RAW/F_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT2 --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_F_b$B.npz --trace-kc-class $WD/tr_F_b$B.npz --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then echo "=> $(modsum "$f")"; else fail $rc "$f"; fi; fi
  for K in kcF kcD; do
    tag="e161 $K b$B"; if ! skip "$tag"; then f="$RAW/${K}_b$B.log"; printf "  %s: " "$tag"
      if [ "$K" = "kcF" ]; then XA="--kc-type-weights $KW"; else XA=""; fi
      timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 25 --episodes 0 --steps 100 --brain-seed $B --rw-apm-scale 0 $XA \
        --trials 100 --decomp-weights $WD/w0_b$B.npz --decomp-mode kcrate --kc-rate-file $WD/${K}_b$B.npz > "$f" 2>&1; rc=$?
      if grep -q "^=> KCRATE kc_r" "$f"; then echo "=> 스파이크 $(grep '^=> KCRATE' "$f" | grep -oE '제시 스파이크 [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')|| 적재 $(grep -c "$LD" "$f")"; else fail $rc "$f"; fi; fi
  done
done
echo "[E161] 전체 루프 종료"
