#!/bin/bash
# E165 본실험 — 기준 logs/E165/criteria_fixed.txt. 망 안 형성의 노출 통계 일반성(외부 검토 권고 1). 뇌 16~20, 형성 설정 고정(η 0.02·β 0.3·m_max 32·τ 20).
# 팔 NR: 무작위 순서 + 제시마다 강도 U[0.5, 0.9], good·bad × 좌·우 각 100. 팔 NU: NR + bad 세 배(good 각 100, bad 각 300). 노출 일정 시드 = 뇌 시드(기본).
# 뇌·팔마다: dev(kcdevoja) → ov(kcoverlap 200, 표준 강도) → F(E141 인자 500시행 + 형성 가중치, 추적). w0·rate 는 같은 뇌 E161 파일(결정적, E161 과 같은 경로).
# 요약 줄: "  e165 {NR|NU} {dev|ov|F} b16: => ..." (판정은 원 로그를 직접 읽는다). 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E165.log"
RAW="$R/research/experiments/logs/E165"; WD="$R/research/experiments/traces/E165"; mkdir -p "$RAW" "$WD"
W161="$R/research/experiments/traces/E161"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e165_main_run && cd /root/e165_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
OJA="--kc-type-oja --kc-oja-eta 0.02 --kc-oja-beta 0.3 --kc-oja-mmax 32 --kc-oja-tau 20"
LD='^\[E153 종류 입력 적재\].*검증 일치'
skip() { grep -qF "$1: =>" "$LOG" 2>/dev/null && { echo "  $1: [건너뜀]"; return 0; }; return 1; }
fail() { echo "[실패 rc=$1]"; tail -2 "$2" | sed 's/^/      /'; }
for B in 16 17 18 19 20; do
  for ARM in NR NU; do
    [ "$ARM" = "NR" ] && MULT=1 || MULT=3
    XS="--kc-dev-order random --kc-dev-bad-mult $MULT --kc-dev-int-lo 0.5 --kc-dev-int-hi 0.9"
    KW="$WD/kctype_${ARM}_b$B.npz"
    tag="e165 $ARM dev b$B"; if ! skip "$tag"; then f="$RAW/${ARM}_dev_b$B.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $W161/w0_b$B.npz --decomp-mode kcdevoja --kc-dev-n 100 $OJA $XS --kc-dev-save $KW > "$f" 2>&1; rc=$?
      if grep -q "^=> KCDEVOJA" "$f" && grep -q "^\[E165 노출\]" "$f"; then
        echo "=> $(grep '^\[E165 노출\]' "$f" | grep -oE 'good_l=[0-9]+ bad_l=[0-9]+|int_mean=[0-9.]+|cyc_match=[0-9.]+' | tr '\n' ' ')| $(grep '^=> KCDEVOJA' "$f" | grep -oE 'side=[lr]|sel_med0=[0-9.]+|sel_med=[0-9.]+|goodfrac=[0-9.]+' | tr '\n' ' ')"
      else fail $rc "$f"; continue; fi; fi
    tag="e165 $ARM ov b$B"; if ! skip "$tag"; then f="$RAW/${ARM}_ov_b$B.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $W161/w0_b$B.npz --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1; rc=$?
      if grep -q "^=> KCOVERLAP" "$f"; then echo "=> $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')|| 적재 $(grep -c "$LD" "$f")"; else fail $rc "$f"; fi; fi
    tag="e165 $ARM F b$B"; if ! skip "$tag"; then f="$RAW/${ARM}_F_b$B.log"; printf "  %s: " "$tag"
      timeout 14400 python reflex_override_task.py $BASE $ACT2 --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
        --save-weights $WD/w_${ARM}_F_b$B.npz --trace-kc-class $WD/tr_${ARM}_F_b$B.npz --kc-rate-file $W161/rate_b$B.npz > "$f" 2>&1; rc=$?
      if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
        echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f")"
      else fail $rc "$f"; fi; fi
  done
done
echo "[E165] 전체 루프 종료"
