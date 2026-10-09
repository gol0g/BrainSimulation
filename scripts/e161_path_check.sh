#!/bin/bash
# E161 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15 에서 새 순서(w0 → rate → D → dev → ov → F)를 짧게(학습 1ep) 돌려 파일 연쇄를 확인한다.
# 확인: w0 저장([저장] 줄·파일), rate KCRATE 2줄·파일, D 학습 추적(새 rate 파일)·적재 0, dev Oja 2줄·KCDEVOJA, ov 적재·자카드, F 학습 적재 2·추적.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E161/pathcheck"; WD="$R/research/experiments/traces/E161/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e161_run && cd /root/e161_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
OJA="--kc-type-oja --kc-oja-eta 0.02 --kc-oja-beta 0.3 --kc-oja-mmax 32 --kc-oja-tau 20"
LD='^\[E153 종류 입력 적재\].*검증 일치'
B=15; KW="$WD/kctype_oja_b$B.npz"
f="$OUT/w0_b$B.log"; timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 0 --steps 100 --transplant-eval --brain-seed $B --save-weights $WD/w0_b$B.npz > "$f" 2>&1
echo "[w0 rc=$?] $(grep '^\[저장\]' "$f" | cut -c1-80) | 파일 $([ -f $WD/w0_b$B.npz ] && echo 있음 || echo 없음)"
f="$OUT/rate_b$B.log"; timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 0 --episodes 0 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcrate --trials 200 --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1
echo "[rate rc=$?] KCRATE $(grep -c '^=> KCRATE' "$f")줄 | 파일 $([ -f $WD/rate_b$B.npz ] && echo 있음 || echo 없음)"
f="$OUT/D_b$B.log"; timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed $B \
  --save-weights $WD/w_D_b$B.npz --trace-kc-class $WD/tr_D_b$B.npz --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1
echo "[D 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | KCTRACE3 $(grep -c '^=> KCTRACE3' "$f") | 분류 $(grep -oE '\[E139\] KC 계층 추적: 시냅스 역할 교차 [0-9]+ 같은쪽 [0-9]+' "$f")"
f="$OUT/dev_b$B.log"; timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcdevoja --kc-dev-n 100 $OJA --kc-dev-save $KW > "$f" 2>&1
echo "[dev rc=$?] Oja $(grep -c '^\[E160 종류 입력 Oja\]' "$f") | $(grep '^=> KCDEVOJA' "$f" | grep -oE 'side=[lr] fired=[0-9]+|sel_med0=[0-9.]+|sel_med=[0-9.]+|sum_med=[0-9.]+' | tr '\n' ' ')"
f="$OUT/ov_b$B.log"; timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WD/w0_b$B.npz --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1
echo "[ov rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')"
f="$OUT/F_b$B.log"; timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
  --save-weights $WD/w_F_b$B.npz --trace-kc-class $WD/tr_F_b$B.npz --kc-rate-file $WD/rate_b$B.npz > "$f" 2>&1
echo "[F 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | KCTRACE3 $(grep -c '^=> KCTRACE3' "$f")"
echo "  비교: 뇌 15 E160 보정 선택 칸(같은 설정) 선택성 0.858/0.800, 자카드 0.0169/0.0087 — 같은 뇌·같은 설정이면 dev·ov 가 같아야 한다(w0 는 연결 확인용)"
echo "[E161 경로 검사] 종료"
