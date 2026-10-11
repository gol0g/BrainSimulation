#!/bin/bash
# E179 경로 검사(조건 2) — 기준 logs/E179/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15, E160 보정 칸 형성 가중치, 초기 연결 스냅숏 conn_b15(E176).
# 새 코드: forager_brain zz_ctx2(맥락 전용 집단 2, 호스트 난수 17700 + genn_seed), 과제 --ctx2-n·--ctx-task bicond2·--eval-ctx2·ctx_state·kcctx 대칭 모드.
# (a) 기본 A 200(스냅숏 없음) → +0.0228 → −0.5094 보상 134(E162 정확 재현).
# (b) 스냅숏 + zz_ctx + zz_ctx2(둘 다 켜지 않음, 기저값 0) A 200 → (a) 와 사전·사후 변조폭·보상 수 정확히 같음, '[맥락 입력]'·'[맥락 입력 2]' 줄 각 ≥ 1.
# (c) kcctx 대칭(맥락 B = zz_ctx w 4, A = zz_ctx2 w 4, 기저값 −4000, 60 제시, (a) 가중치) → B 집단 발화 끔(= A 자리) 0·켬 > 0, A 집단 발화 끔 > 0·켬 0, good+A 대 good+B 자카드 < 1.
# (d) bicond2 4ep(기저값 −4000) + 이식 평가 학습 A·B(+ 진단): '[맥락 과제] bicond2' 줄·맥락 B 150~250, 추적 열 37·맥락별 규칙 일치 1.000·동결 잔차 ≤ 1e-3,
#     A 평가에만 '[맥락 평가 A]'·B 평가에만 '[맥락 평가] ctx=켬', 평가 동작점 줄·진단 줄. 탐욕 결정 비율은 기록만.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E179/pathcheck"; WD="$R/research/experiments/traces/E179/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e179_run && cd /root/e179_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
TN='^\s*\[motor 동작점\]'
for f_ in "$KW15" "$RT15" "$SN15"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E179 경로 검사] 종료"; exit 1; }; done
CTX2="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 4 --ctx-p 0.10 --ctx2-n 200 --ctx2-w 4 --ctx2-p 0.10"
for ARM in a b; do
  case $ARM in a) XC="" ;; b) XC="$CTX2 --motor-tonic 0" ;; esac
  f="$OUT/${ARM}_A200_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $XC \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
  echo "[($ARM) rc=$rc] 맥락 입력 줄 $(grep -c '^  \[맥락 입력\] 맥락 전용 집단 zz_ctx ' "$f") 맥락 입력 2 줄 $(grep -c '^  \[맥락 입력 2\]' "$f") 동작점 줄 $(grep -cE "$TN" "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
  [ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
done
f="$OUT/c_kcctx_sym_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX2 --motor-tonic -4000 --decomp-weights $WD/w_a_b15.npz --decomp-mode kcctx --trials 60 > "$f" 2>&1
echo "[(c) kcctx 대칭 rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-900)"
f="$OUT/d_bicond2_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $CTX2 --motor-tonic -4000 --ctx-task bicond2 \
  --save-weights $WD/w_bc2_b15.npz --trace-kc-class $WD/tr_bc2_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[(d) bicond2 rc=$rc] $(grep '^\[맥락 과제\]' "$f" | tr '\n' ' ' | cut -c1-300) | 동작점 줄 $(grep -cE "$TN" "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
for C in A B; do
  [ "$C" = "B" ] && XE="--eval-ctx" || XE="--eval-ctx2"
  g="$OUT/d_ev_learn_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX2 --motor-tonic -4000 --decomp-weights $WD/w_bc2_b15.npz --decomp-mode all $XE --eval-diag > "$g" 2>&1
  echo "[(d) learn $C rc=$?] 동작점 줄 $(grep -cE "$TN" "$g") 적재 $(grep -c "$LD" "$g") 맥락 평가 B 줄 $(grep -c '^\[맥락 평가\] ctx=켬' "$g") 맥락 평가 A 줄 $(grep -c '^\[맥락 평가 A\]' "$g") | $(grep '^=> DECOMP' "$g" | cut -c1-60) | $(grep '^\[E171 평가 진단\]' "$g" | sed -E 's/^\[E171 평가 진단\] variant=base //' | tr '\n' ' ')"
done
(cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e178 as J
s = J.bicond_stats(np.load("research/experiments/traces/E179/pathcheck/tr_bc2_b15.npz")["rows"])
print("  (d) 추적: 시행 %d 열37 %s 맥락 B 비율 %.3f 맥락별 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e | 탐욕 |v|≤0.02 B %.2f A %.2f(기록만)"
      % (s["n"], s["has_ctx"], s["frac_ctx"], s["agree"], s["res"], s["pre_ratio"], s["dec_on"], s["dec_off"]))
a = np.load("research/experiments/traces/E179/pathcheck/w_a_b15.npz"); b = np.load("research/experiments/traces/E179/pathcheck/w_b_b15.npz")
d = sum(int((np.asarray(a[k], float) != np.asarray(b[k], float)).sum()) for k in a.files)
print("  학습 가중치 (a) 대 (b): 다른 원소 %d / %d (비결정성 범위면 0 이 아닐 수 있음)" % (d, sum(a[k].size for k in a.files)))
PY
)
echo "[E179 경로 검사] 종료 $(date '+%F %T')"
