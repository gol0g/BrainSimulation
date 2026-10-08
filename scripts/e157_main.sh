#!/bin/bash
# E157 본실험 — 기준 logs/E157/criteria_fixed.txt(2026-10-09 01:06:04), k = logs/E157/kstar.txt(표본 밖 뇌 15 보정). 뇌 10~14. 재개 가능(P13: E157.log 의 "태그: =>" 건너뜀).
# 순서: kcrate(D·F·Fk·Dk — E156 진단 인자) → 겹침(Fk·Dk — E153 인자) → 반사 25 [사전](D·F·Fk·Dk, --episodes 0) → 학습(Fk·Dk — E153 학습 인자).
# D·F 학습은 E141·E153 원 로그 재사용(경로 검사 재현 확인 — logs/E157/calib.out).
# 요약 줄: "  e157 {kcrate|ov|r25|learn} {D|F|Fk|Dk} b10: => ..." (판정은 원 로그를 직접 읽는다)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E157.log"
RAW="$R/research/experiments/logs/E157"; WD="$R/research/experiments/traces/E157"; mkdir -p "$RAW" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e157_main_run && cd /root/e157_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
ACT25="--act-window 3 --act-current 5000 --judge exec --reflex-w 25 --rw-apm-scale 0"
KF=$(grep -oE 'Fk k=[0-9.]+' "$RAW/kstar.txt" 2>/dev/null | cut -d= -f2)
KD=$(grep -oE 'Dk k=[0-9.]+' "$RAW/kstar.txt" 2>/dev/null | cut -d= -f2)
if [ -z "$KF" ] || [ -z "$KD" ]; then echo "[E157] k 없음(보정 실패 또는 kstar.txt 없음) — 본실험 안 함"; exit 1; fi
echo "[E157] k: Fk $KF Dk $KD"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'
xa() {  # $1 팔 $2 뇌 → 종류 입력 인자
  case $1 in
    D) echo "" ;;
    F) echo "--kc-type-weights $R/research/experiments/traces/E153/kctype_b$2.npz" ;;
    Fk) echo "--kc-type-weights $R/research/experiments/traces/E153/kctype_b$2.npz --kc-type-scale $KF" ;;
    Dk) echo "--kc-type-scale $KD" ;;
  esac
}
ld() { echo "적재 $(grep -c "$LD" "$1") 배율 $(grep -c "$SC" "$1")"; }
skip() { grep -qF "$1: =>" "$LOG" 2>/dev/null && { echo "  $1: [건너뜀]"; return 0; }; return 1; }
fail() { echo "[실패 rc=$1]"; tail -2 "$2" | sed 's/^/      /'; }
# 1) kcrate
for B in 10 11 12 13 14; do for A in D F Fk Dk; do
  tag="e157 kcrate $A b$B"; skip "$tag" && continue
  f="$RAW/kcrate_${A}_b$B.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $ACT25 --episodes 0 --steps 100 --brain-seed $B $(xa $A $B) --trials 100 \
    --decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode kcrate --kc-rate-file $WD/kcrate_${A}_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCRATE kc_r" "$f"; then echo "=> 스파이크 $(grep '^=> KCRATE' "$f" | grep -oE '제시 스파이크 [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')|| $(ld "$f")"; else fail $rc "$f"; fi
done; done
# 2) 겹침
for B in 10 11 12 13 14; do for A in Fk Dk; do
  tag="e157 ov $A b$B"; skip "$tag" && continue
  f="$RAW/ov_${A}_b$B.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode kcoverlap --trials 200 $(xa $A $B) > "$f" 2>&1; rc=$?
  if grep -q "^=> KCOVERLAP" "$f"; then echo "=> $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')|| $(ld "$f")"; else fail $rc "$f"; fi
done; done
# 3) 반사 25 [사전](희석 — 부지표)
for B in 10 11 12 13 14; do for A in D F Fk Dk; do
  tag="e157 r25 $A b$B"; skip "$tag" && continue
  f="$RAW/r25_${A}_b$B.log"; printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $ACT25 --episodes 0 --steps 100 --brain-seed $B $(xa $A $B) > "$f" 2>&1; rc=$?
  if grep -q "^\[사전\]" "$f"; then echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') || $(ld "$f")"; else fail $rc "$f"; fi
done; done
# 4) 학습(Fk·Dk)
for A in Fk Dk; do for B in 10 11 12 13 14; do
  tag="e157 learn $A b$B"; skip "$tag" && continue
  f="$RAW/learn_${A}_b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT2 --episodes 5 --steps 100 --transplant-eval --brain-seed $B $(xa $A $B) \
    --save-weights $WD/w_${A}_b$B.npz --trace-kc-class $WD/tr_${A}_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(ld "$f")"
  else fail $rc "$f"; fi
done; done
echo "[E157] 전체 루프 종료"
