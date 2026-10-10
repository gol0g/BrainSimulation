#!/bin/bash
# E172 경로 검사(조건 2) — 기준 고정(logs/E172/criteria_fixed.txt) 뒤.
# (1) 학습 경로 회귀(표본 밖 뇌 15, E160 보정 선택 칸 형성 가중치): E162 경로 검사 A 200·AB 400(200 부터 과제 B)·E163 경로 검사 반전 400(200 부터)을
#     현재 코드(E164~E171 옵션 추가 뒤)로 다시 + 평가 2 — 기대값은 logs/E162/path_check.out·logs/E163/path_check.out 정확 재현.
# (2) 적재 경로(뇌 16~20, E161 형성 가중치): --episodes 0 이식 평가 → 적재 검증 줄과 [사전] = E161 F 팔 [사전](이미 아는 값 — 새 결과 아님).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E172/pathcheck"; WD="$R/research/experiments/traces/E172/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
W161="$R/research/experiments/traces/E161"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e172_run && cd /root/e172_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for ARM in A AB rev; do
  case $ARM in
    A) X="--episodes 2" ;;
    AB) X="--episodes 4 --task-b-after 200" ;;
    rev) X="--episodes 4 --reverse-after 200" ;;
  esac
  f="$OUT/train_${ARM}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $X --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
  echo "[$ARM rc=$rc] 적재 $(grep -c "$LD" "$f") $(grep -E '^\[과제 B\]|^\[반전\]' "$f" | cut -c1-30) $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 반사 $(grep '^\[반사가중치\] good_food_to_motor' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
done
for WS in "AB bad" "none base"; do
  set -- $WS; W=$1; S=$2
  [ "$W" = "AB" ] && X="--decomp-weights $WD/w_AB_b15.npz --decomp-mode all" || X="--decomp-weights $WD/w_A_b15.npz --decomp-mode none"
  g="$OUT/ev_b15_${W}_$S.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $X --eval-variant $S > "$g" 2>&1
  echo "[$W $S] 적재 $(grep -c "$LD" "$g") $(grep '^=> DECOMP' "$g" | cut -c1-60)"
done
echo "  기대(E162·E163 경로 검사): A +0.0228 → -0.5094 보상 134회 · AB +0.0228 → -0.4783 보상 274회 · rev +0.0228 → +0.2724 · AB bad mod=+0.5105 · none base mod=+0.0228 · 적재 2 · 반사 0→0"
(cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e172_ret as J, judge_e172_rev as V
for a, sw in (("A", None), ("AB", 200)):
    s = J.stats(np.load("research/experiments/traces/E172/pathcheck/tr_%s_b15.npz" % a)["rows"], sw)
    print("  %s 추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e" % (a, s["n"], s["agree"], s["res"], s["pre_ratio"]))
V.REV_AT = 200
s = V.rev_stats(np.load("research/experiments/traces/E172/pathcheck/tr_rev_b15.npz")["rows"])
print("  rev 추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e 블록 보상 %s" % (s["n"], s["agree"], s["res"], s["pre_ratio"], s["rew_blk"]))
PY
)
echo "  기대 추적: A 200·AB 400·rev 400, 규칙 일치 1.0000, 동결 잔차 ≈1.4e-08, 도파민전 0, rev 블록 보상 [59, 75, 29, 58]"
for B in 16 17 18 19 20; do
  f="$OUT/load_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 0 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $W161/kctype_oja_b$B.npz > "$f" 2>&1; rc=$?
  echo "[load b$B rc=$rc] 적재 $(grep -c "$LD" "$f") $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+')"
done
echo "  기대(E161 F 팔 [사전]): b16 +0.0149 · b17 +0.0064 · b18 +0.0094 · b19 +0.0208 · b20 +0.0166, 적재 ≥ 1"
echo "[E172 경로 검사] 종료"
