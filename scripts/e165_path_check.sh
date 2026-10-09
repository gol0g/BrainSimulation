#!/bin/bash
# E165 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15.
# (a) 노출 일정 옵션 끔(고정 순환): E161 경로 검사 dev·ov 를 정확히 재현해야 한다(선택성 0.8575/0.8004, 합 0.9734/0.8821, 자카드 0.0169/0.0087) — 새 일정 코드가 기본 동작을 바꾸지 않았는가.
# (b) NR(무작위 순서·강도 0.5~0.9) dev·ov, (c) NU(+ bad 세 배) dev·ov — 노출 구성 줄(제시 수·강도·순환 일치)이 의도대로인가.
# (d) NU 형성 가중치로 학습 1ep — 적재 2·추적 연쇄. (e) 옵션을 kcdevoja 밖에서 주면 거부되는가(조용히 무시 금지).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E165/pathcheck"; WD="$R/research/experiments/traces/E165/pathcheck"; mkdir -p "$OUT" "$WD"
P161="$R/research/experiments/traces/E161/pathcheck"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e165_run && cd /root/e165_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
OJA="--kc-type-oja --kc-oja-eta 0.02 --kc-oja-beta 0.3 --kc-oja-mmax 32 --kc-oja-tau 20"
LD='^\[E153 종류 입력 적재\].*검증 일치'
B=15
for ARM in CY NR NU; do
  case $ARM in
    CY) XS="" ;;
    NR) XS="--kc-dev-order random --kc-dev-bad-mult 1 --kc-dev-int-lo 0.5 --kc-dev-int-hi 0.9" ;;
    NU) XS="--kc-dev-order random --kc-dev-bad-mult 3 --kc-dev-int-lo 0.5 --kc-dev-int-hi 0.9" ;;
  esac
  KW="$WD/kctype_${ARM}_b$B.npz"
  f="$OUT/${ARM}_dev_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $P161/w0_b$B.npz --decomp-mode kcdevoja --kc-dev-n 100 $OJA $XS --kc-dev-save $KW > "$f" 2>&1
  echo "[$ARM dev rc=$?] Oja $(grep -c '^\[E160 종류 입력 Oja\]' "$f") | $(grep '^\[E165 노출\]' "$f" | cut -c13-) "
  echo "    $(grep '^=> KCDEVOJA' "$f" | grep -oE 'side=[lr] fired=[0-9]+|sel_med0=[0-9.]+|sel_med=[0-9.]+|goodfrac=[0-9.]+|sum_med=[0-9.]+' | tr '\n' ' ')"
  f="$OUT/${ARM}_ov_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $P161/w0_b$B.npz --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1
  echo "[$ARM ov rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')"
done
echo "  기대 (a) CY = E161 경로 검사: side=l fired=243 sel_med0=0.5567 sel_med=0.8575 sum_med=0.9734 side=r fired=234 sel_med0=0.5536 sel_med=0.8004 sum_med=0.8821 | 자카드 0.0169/0.0087(good 55·61, bad 65·55)"
f="$OUT/NU_F_b$B.log"
timeout 3600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $WD/kctype_NU_b$B.npz \
  --save-weights $WD/w_NU_F_b$B.npz --trace-kc-class $WD/tr_NU_F_b$B.npz --kc-rate-file $P161/rate_b$B.npz > "$f" 2>&1
echo "[(d) NU F 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | KCTRACE3 $(grep -c '^=> KCTRACE3' "$f")"
f="$OUT/guard_b$B.log"
timeout 600 python reflex_override_task.py $BASE $ACT2 --episodes 1 --steps 100 --brain-seed $B --kc-dev-order random > "$f" 2>&1
echo "[(e) 옵션 거부 rc=$?] $(grep -c 'kcdevoja 노출 전용' "$f")줄(기대 1, rc 1)"
echo "[E165 경로 검사] 종료"
