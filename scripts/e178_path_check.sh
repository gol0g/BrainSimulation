#!/bin/bash
# E178 경로 검사(조건 2) — 기준 logs/E178/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15, E160 보정 칸 형성 가중치, 초기 연결 스냅숏 conn_b15(E176).
# 새 코드: forager_brain motor_tonic(motor 좌·우 Ioffset 기저값, load 뒤 설정), 과제 --motor-tonic·행동 창/보상 창 침묵의 기저값 기준 설정·되돌림.
# (a) 기본 A 200(스냅숏 없음, 기저값 0) → +0.0228 → −0.5094 보상 134(E162 정확 재현 — 행동 창이 기저값 0 ± 전류로 바뀐 코드).
# (b) 스냅숏 + 맥락 전용 집단(w 4, 켜지 않음) + --motor-tonic 0 명시 A 200 → (a)와 정확히 같음, '[motor 동작점]' 줄 0.
# (c) --motor-tonic −1000(스냅숏 없음) A 200 → '[motor 동작점]' 줄 ≥ 2(학습 뇌·이식 평가 뇌), 학습 완료, 사전·사후 변조폭이 (a)와 다름(행동 도달),
#     추적 열 35(행동 창 실행 motor 발화율 합) > 0.
# (d) 무학습 이식 평가(맥락 없음, (a) 가중치) X = 0 대 −1000 + --eval-diag → X = −1000 의 motor 발화율 4개가 모두 X = 0 보다 낮음(reset 뒤에도 기저값 유지·평가 도달).
# (e) 쌍조건 4ep(스냅숏 + 맥락 w 4 + X = −1000) + 이식 평가 학습 끔·켬(+ --eval-diag): '[맥락 과제]' 줄, 맥락 켬 약 반, 추적 열 37·맥락별 규칙 일치·동결,
#     켬 평가에만 '[맥락 평가]' 줄, 평가 동작점 줄·진단 줄. 맥락 켬 탐욕 결정 |v| ≤ 0.02 비율은 기록만(400시행 — 판정 아님).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E178/pathcheck"; WD="$R/research/experiments/traces/E178/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e178_run && cd /root/e178_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
TN='^\s*\[motor 동작점\]'
for f_ in "$KW15" "$RT15" "$SN15"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E178 경로 검사] 종료"; exit 1; }; done
CTX4="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 4 --ctx-p 0.10"
for ARM in a b c; do
  case $ARM in a) XC="" ;; b) XC="$CTX4 --motor-tonic 0" ;; c) XC="--motor-tonic -1000" ;; esac
  f="$OUT/${ARM}_A200_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $XC \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
  echo "[($ARM) rc=$rc] 동작점 줄 $(grep -cE "$TN" "$f") 맥락 집단 줄 $(grep -c '^  \[맥락 입력\] 맥락 전용 집단' "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
  [ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
done
for X in 0 -1000; do
  g="$OUT/d_none_x${X}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --motor-tonic $X --decomp-weights $WD/w_a_b15.npz --decomp-mode none --eval-diag > "$g" 2>&1
  echo "[(d) none X=$X rc=$?] 동작점 줄 $(grep -cE "$TN" "$g") | $(grep '^=> DECOMP' "$g" | cut -c1-60) | $(grep '^\[E171 평가 진단\]' "$g" | sed -E 's/^\[E171 평가 진단\] variant=base //' | tr '\n' ' ')"
done
f="$OUT/e_bicond_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 $CTX4 --motor-tonic -1000 --ctx-task bicond \
  --save-weights $WD/w_bc_b15.npz --trace-kc-class $WD/tr_bc_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[(e) bicond rc=$rc] $(grep '^\[맥락 과제\]' "$f" | tr '\n' ' ' | cut -c1-280) | 동작점 줄 $(grep -cE "$TN" "$f") 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
for C in off on; do
  [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
  g="$OUT/e_ev_learn_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX4 --motor-tonic -1000 --decomp-weights $WD/w_bc_b15.npz --decomp-mode all $XE --eval-diag > "$g" 2>&1
  echo "[(e) learn $C rc=$?] 동작점 줄 $(grep -cE "$TN" "$g") 적재 $(grep -c "$LD" "$g") 맥락 평가 줄 $(grep -c '^\[맥락 평가\]' "$g") | $(grep '^=> DECOMP' "$g" | cut -c1-60) | $(grep '^\[E171 평가 진단\]' "$g" | sed -E 's/^\[E171 평가 진단\] variant=base //' | tr '\n' ' ')"
done
(cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e178 as J
s = J.bicond_stats(np.load("research/experiments/traces/E178/pathcheck/tr_bc_b15.npz")["rows"])
print("  (e) 추적: 시행 %d 열37 %s 맥락 켬 비율 %.3f 맥락별 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e | 탐욕 |v|≤0.02 켬 %.2f 끔 %.2f(기록만)"
      % (s["n"], s["has_ctx"], s["frac_ctx"], s["agree"], s["res"], s["pre_ratio"], s["dec_on"], s["dec_off"]))
a = np.load("research/experiments/traces/E178/pathcheck/w_a_b15.npz"); b = np.load("research/experiments/traces/E178/pathcheck/w_b_b15.npz")
d = sum(int((np.asarray(a[k], float) != np.asarray(b[k], float)).sum()) for k in a.files)
print("  학습 가중치 (a) 대 (b): 다른 원소 %d / %d (비결정성 범위면 0 이 아닐 수 있음)" % (d, sum(a[k].size for k in a.files)))
for nm in ("a", "c"):
    r = np.load("research/experiments/traces/E178/pathcheck/tr_%s_b15.npz" % nm)["rows"]
    print("  (%s) 행동 창 실행 motor 발화율 합(열 35) 평균 %.3f · 반대 motor(열 36) 평균 %.3f" % (nm, r[:, 35].mean(), r[:, 36].mean()))
PY
)
echo "[E178 경로 검사] 종료 $(date '+%F %T')"
