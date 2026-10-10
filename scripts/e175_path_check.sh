#!/bin/bash
# E175 경로 검사(조건 2) — 기준 logs/E175/criteria_fixed.txt 고정 뒤. (e173·e175 경로 검사를 맥락 인자만 바꿔 옮김 — 맥락 = assoc_binding 흥분) 표본 밖 뇌 15, E160 보정 칸 형성 가중치(E162 경로 검사와 같음).
# (a) 기본 경로 회귀(--ctx-i 0 기본 — 맥락 기전 없음): E162 경로 검사 A 200 → [사전] +0.0228 [사후] −0.5094 보상 134(코드 추가 뒤 기본 경로 불변).
# (b) assoc_binding Ioffset 동적화(--ctx-ab-i 2)를 켜되 맥락은 켜지 않음(과제 none): 같은 값이어야 한다(동적화가 연결·난수·궤적을 바꾸지 않음).
# (c) kcctx(I_ab 2, 60 제시): assoc_binding 발화 켬 > 끔, KC 반응 변화.
# (d) 쌍조건 짧은 학습(I_ab 2, 4ep = 400시행): '[맥락 과제]' 줄, 맥락 켬 약 반, 추적 열 37·맥락별 보상-규칙 일치 1.000·동결 잔차.
# (e) 이식 평가 학습 × 맥락 끔·켬: 켬에만 '[맥락 평가]' 줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E175/pathcheck"; WD="$R/research/experiments/traces/E175/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e175_run && cd /root/e175_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
CTX="--ctx-ab-i 2"
for ARM in a b; do
  [ "$ARM" = "a" ] && XC="" || XC="$CTX"
  f="$OUT/${ARM}_A200_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $XC \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
  echo "[($ARM) rc=$rc] 맥락 입력 줄 $(grep -c '^  \[맥락 입력\]' "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 기대 +0.0228 → -0.5094 보상 134회"
done
f="$OUT/c_kcctx_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WD/w_a_b15.npz --decomp-mode kcctx --trials 60 > "$f" 2>&1
echo "[(c) kcctx rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-460)"
f="$OUT/d_bicond_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $CTX --ctx-task bicond \
  --save-weights $WD/w_bc_b15.npz --trace-kc-class $WD/tr_bc_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[(d) bicond rc=$rc] $(grep '^\[맥락 과제\]' "$f" | tr '\n' ' ' | cut -c1-260) | 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
for C in off on; do
  [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
  g="$OUT/e_ev_learn_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WD/w_bc_b15.npz --decomp-mode all $XE > "$g" 2>&1
  echo "[(e) learn $C rc=$?] 적재 $(grep -c "$LD" "$g") 맥락 평가 줄 $(grep -c '^\[맥락 평가\]' "$g") | $(grep '^=> DECOMP' "$g" | cut -c1-60)"
done
(cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e175 as J
s = J.bicond_stats(np.load("research/experiments/traces/E175/pathcheck/tr_bc_b15.npz")["rows"])
print("  (d) 추적: 시행 %d 열37 %s 맥락 켬 비율 %.3f 맥락별 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e" % (s["n"], s["has_ctx"], s["frac_ctx"], s["agree"], s["res"], s["pre_ratio"]))
r = np.load("research/experiments/traces/E175/pathcheck/tr_a_b15.npz")["rows"]
print("  (a) 추적 열 수 %d(기본 37 — 맥락 열 없음)" % r.shape[1])
PY
)
echo "[E175 경로 검사] 종료"
