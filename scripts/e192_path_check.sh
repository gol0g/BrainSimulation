#!/bin/bash
# E192 경로 검사(조건 2) — 기준 logs/E192/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# 새 코드: rawcat --rc-measure-ctx(측정 전용 — 맥락 켬·끔, RCKX·RCKXCHK).
# (a) C0(w 4·p 0.10) 측정(rc-reps 3 = 288 제시) → RCKXCHK(켠 맥락 > 0·끈 0·비활성 특징 0·g 불변)·RCKX 쪽마다 72/72·맥락 연결 약 60,000·특징 연결 줄; rcxor 해독(보고).
# (b) C2(w 20·p 0.02) 측정(rc-reps 1 = 96 제시) → 맥락 연결 약 12,000, 특징 연결 수 = (a)(맥락 설정이 다른 연결을 안 바꿈).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E192/pathcheck"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e192_run && cd /root/e192_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
for f_ in "$KW15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E192 경로 검사] 종료"; exit 1; }; done
echo "[E192 경로 검사] 시작 $(date '+%F %T')"
for V in C0 C2; do
  if [ "$V" = "C0" ]; then CTX="--ctx-n 200 --ctx-w 4 --ctx-p 0.10 --ctx2-n 200 --ctx2-w 4 --ctx2-p 0.10 --motor-tonic -4000"; REPS=3
  else CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"; REPS=1; fi
  g="$OUT/${V}_meas_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CTX --decomp-weights $WB --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps $REPS > "$g" 2>&1; rc=$?
  echo "[($V) 측정 rc=$rc] RCKX $(grep -c '^RCKX ' "$g") (l train $(grep -cE '^RCKX .* side=l k=[0-9]+ set=train ' "$g")·new $(grep -cE '^RCKX .* side=l k=[0-9]+ set=new ' "$g")) | $(grep '^RCKXCHK' "$g")"
  grep -E '^\s*\[(맥락 입력|맥락 입력 2|원시 특징)\]' "$g" | sed 's/^ */    /'
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "[(a) C0 해독(보고)]"; python3 $R/scripts/rcxor.py "$OUT/C0_meas_b15.log" | sed 's/^/    /'
echo "  기대: (a) RCKX 288(l 72/72), RCKXCHK npres 288·ctx_on > 0·ctx_off 0·feat_inactive 0·g_same 1·w 4·p 0.100, 맥락 연결 약 60,000; (b) RCKX 96, 맥락 연결 약 12,000(w 20·p 0.020), 특징 연결 수 (a) 와 같음"
echo "[E192 경로 검사] 종료 $(date '+%F %T')"
