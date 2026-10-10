#!/bin/bash
# E176 경로 검사(조건 2) — 기준 logs/E176/criteria_fixed.txt 고정 뒤, 스냅숏(e176_snap.sh) 뒤. 표본 밖 뇌 15, E160 보정 칸 형성 가중치(E162 경로 검사와 같음).
# (a) 기본 A 200(스냅숏 없음) → +0.0228 → −0.5094 보상 134(E162 정확 재현).
# (b) 스냅숏 적재(맥락 집단 없음) A 200 → (a)와 정확히 같음 — 스냅숏 연결 = 장치 초기화 연결(적재 검사 '검증 일치' 2줄 포함).
# (c) 스냅숏 적재 + 맥락 전용 집단(--ctx-n 200 --ctx-w 2, 켜지 않음) A 200 → (a)와 정확히 같음 — 새 뉴런 집단이 기존 뇌를 바꾸지 않음.
# (d) kcctx(스냅숏 + 맥락 w 2, 60 제시): 맥락 집단 발화 끔 0·켬 > 0, KC 반응 변화.
# (e) 쌍조건 4ep(스냅숏 + 맥락 w 2) + 이식 평가 끔·켬: '[맥락 과제]' 줄, 맥락 켬 약 반, 추적 열 37·맥락별 규칙 일치·동결, 켬 평가에만 '[맥락 평가]' 줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E176/pathcheck"; WD="$R/research/experiments/traces/E176/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e176_run && cd /root/e176_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
[ -s "$SN15" ] || { echo "[실패] 스냅숏 없음 $SN15"; echo "[E176 경로 검사] 종료"; exit 1; }
CTX="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 2 --ctx-p 0.10"
for ARM in a b c; do
  case $ARM in a) XC="" ;; b) XC="--conn-snapshot $SN15" ;; c) XC="$CTX" ;; esac
  f="$OUT/${ARM}_A200_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $XC \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
  echo "[($ARM) rc=$rc] 스냅숏 줄 $(grep -c '^  \[E176 초기 연결 스냅숏\]' "$f") $(grep '스냅숏으로 만든 희소 집단' "$f" | head -1 | grep -oE '[0-9]+ 개') 맥락 집단 줄 $(grep -c '^  \[맥락 입력\] 맥락 전용 집단' "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 기대 +0.0228 → -0.5094 보상 134회"
  [ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
done
f="$OUT/d_kcctx_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WD/w_a_b15.npz --decomp-mode kcctx --trials 60 > "$f" 2>&1
echo "[(d) kcctx rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-600)"
f="$OUT/e_bicond_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $CTX --ctx-task bicond \
  --save-weights $WD/w_bc_b15.npz --trace-kc-class $WD/tr_bc_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[(e) bicond rc=$rc] $(grep '^\[맥락 과제\]' "$f" | tr '\n' ' ' | cut -c1-280) | 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
for C in off on; do
  [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
  g="$OUT/e_ev_learn_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WD/w_bc_b15.npz --decomp-mode all $XE > "$g" 2>&1
  echo "[(e) learn $C rc=$?] 적재 $(grep -c "$LD" "$g") 맥락 평가 줄 $(grep -c '^\[맥락 평가\]' "$g") | $(grep '^=> DECOMP' "$g" | cut -c1-60)"
done
(cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e176 as J
s = J.bicond_stats(np.load("research/experiments/traces/E176/pathcheck/tr_bc_b15.npz")["rows"])
print("  (e) 추적: 시행 %d 열37 %s 맥락 켬 비율 %.3f 맥락별 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e" % (s["n"], s["has_ctx"], s["frac_ctx"], s["agree"], s["res"], s["pre_ratio"]))
a = np.load("research/experiments/traces/E176/pathcheck/w_a_b15.npz"); b = np.load("research/experiments/traces/E176/pathcheck/w_b_b15.npz"); c = np.load("research/experiments/traces/E176/pathcheck/w_c_b15.npz")
for nm, z in (("b", b), ("c", c)):
    d = sum(int((np.asarray(a[k], float) != np.asarray(z[k], float)).sum()) for k in a.files)
    print("  학습 가중치 (a) 대 (%s): 다른 원소 %d / %d (KC→motor·D1 8 집단 — 비결정성 범위면 0 이 아닐 수 있음)" % (nm, d, sum(a[k].size for k in a.files)))
PY
)
echo "[E176 경로 검사] 종료"
