#!/bin/bash
# E208 경로 검사(조건 2) — 표본 밖 뇌 15. 기준 logs/E208/criteria_fixed.txt. 새 코드 없음 — 새 것은 조합(F15 + C2 흥분 무게 0 + 게이팅 + T_F0)과 동작점 규칙(scripts/e208_mpick.py)뿐.
# (a)  회귀: E206 경로 검사 (l) 커맨드 그대로(흥분 20·T −1300) → 가중치 4배열이 E206 w_l_b15 와 같아야(현재 코드가 E206 학습 경로를 재현 — E206 F-GW 짝 재사용의 전제)
# (c)  맥락 흥분 0 확인: F15 맥락 X 켬 범주 측정(E207 옵션 --rc-measure-ctx-on X) CW0 대 CW20 — '[E207 측정 맥락]' 켠 발화 > 0·끈 0·비활성/활성 ≤ 0.01,
#      설정 줄 w 0.00 대 20.00·연결 수 같음, (보고) 겹침 S 첫/둘째 — 흥분 0 이면 맥락 없음(E207 경로 검사 (c) F15 none 2740/1376) 쪽일 것
# (m)  동작점 규칙(e208_mpick.py — 합성 17/17): 무학습 평가 motor 평균(맥락 X·Y × A·B × train·new × 두 쪽 × L/R = 32값). 기준 m_ref = F15·CW20·게이팅·T −1300(E206 F-GW 동작점) —
#      그 32값이 E206 보정 로그 m_F_1300_{X|Y}_b15 와 같아야(평가 경로 회귀). F15·CW0: −1300 에서 |차| ≤ 0.0170 이면 −1300, 아니면 방향대로 격자(구동 줄면 −1300 … +1700,
#      늘면 −1300 … −4300, 300 간격 — 0 을 지나지 않음) → 끝이면 넓힘(+3200 / −5800) → 최소 |차| ≤ 0.0170 일 때만 T_F0(logs/E208/calib_tonic.txt), 아니면 실패(구현 불가 — 미해결)
# (l)·(r)·(e)  E206 경로 검사와 같은 학습·평가 점검 + 모든 로그 맥락 흥분 무게 줄(w 0.00·p 0.020·연결 수 = (c) CW20 기준)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E208/pathcheck"; WD="$R/research/experiments/traces/E208/pathcheck"; mkdir -p "$OUT" "$WD"
CAL="$R/research/experiments/logs/E208/calib_tonic.txt"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
W6L="$R/research/experiments/traces/E206/pathcheck/w_l_b15.npz"   # E206 경로 검사 (l) — F15·CW20 같은 시드(가중치 다름 점검의 짝)
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"   # E206 경로 검사 형성 그대로
M6="$R/research/experiments/logs/E206/pathcheck"   # E206 보정 로그(m_F_1300_{X|Y}_b15 — m_F 0.6087)
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e208_run && cd /root/e208_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
B0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"     # E208 조작
CW20="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"  # 기준(E206)
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
LRN="--kc-motor-hard --kc-motor-comp --da-baseline"
for f_ in "$KW15" "$SN15" "$WB" "$W6L" "$F15" "$M6/m_F_1300_X_b15.log" "$M6/m_F_1300_Y_b15.log"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E208 경로 검사] 종료"; exit 1; }; done
echo "[E208 경로 검사] 시작 $(date '+%F %T')"
# (a) 회귀 — E206 경로 검사 (l) 그대로(흥분 20·T −1300·같은 시드)
fa="$OUT/a20_train_b15.log"
timeout 1800 python reflex_override_task.py $B0 --kc-motor-eta 2.4 --env-seed 2 $ACT $FEAT --feat-weights $F15 $CW20 $GATE --motor-tonic -1300 --rc-task catctx --episodes 2 --steps 20 \
  --transplant-eval --brain-seed 15 --kc-type-weights $KW15 --save-weights $WD/w_a20_b15.npz $LRN --rc-seed 38300 --rc-trial-seed 38515 --rc-ctx-seed 38615 > "$fa" 2>&1; rc=$?
echo "[(a) 회귀 — E206 (l) 커맨드 rc=$rc $(date '+%T')] $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$fa" | cut -c1-120)"
[ $rc -ne 0 ] && tail -4 "$fa" | sed 's/^/      /'
# (c) 맥락 흥분 0 확인 — F15 맥락 X 켬 범주 측정(학습 없음, E207 측정 커맨드), CW0 대 CW20(둘 다 T −1300 — 측정은 motor 를 안 씀)
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
for spec in "c0 CW" "c20 CW20"; do
  set -- $spec; N=$1; V=$2; g="$OUT/${N}_meas_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT ${!V} $GATE --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $F15 --rc-measure-ctx-on X > "$g" 2>&1; rc=$?
  echo "[($N) 맥락 X 켬 측정 rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · $(grep -E '^\s*\[맥락 입력' "$g" | grep -oE 'zz_ctx2? [0-9]+ → KC 좌·우, w=[0-9.]+, p=[0-9.]+, 연결 좌 [0-9]+·우 [0-9]+' | sort -u | tr '\n' ' ')"
  grep '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  grep -E '^RCJ side=[lr] pair=(AB|AC|BD)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
# (m) 동작점 — 기준(F15·CW20·T −1300) 대 F15·CW0, 맥락 X·Y, 게이팅
mrun() {   # $1 이름 $2 맥락 인자 변수 이름(CW|CW20) $3 기저값 $4 맥락
  local g="$OUT/m_$1_$4_b15.log" ex="--feat-weights $F15"
  if [ "$(grep -c '^RCEVAL ' "$g" 2>/dev/null)" = "4" ]; then return; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT ${!2} $GATE $ex --motor-tonic $3 --brain-seed 15 --kc-type-weights $KW15 \
    --decomp-weights $WB --decomp-mode none --rc-eval --rc-eval-ctx $4 --rc-seed 38300 --trials 96 --eval-diag > "$g" 2>&1
  echo "  (m $1 $4) RCEVAL $(grep -c '^RCEVAL ' "$g") | 특징 적재 $(grep -c '^\[E184 특징 가중치 적재\]' "$g") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ')"
}
for X in X Y; do mrun ref CW20 -1300 $X; mrun G0_m1300 CW -1300 $X; done
MP="python3 $R/scripts/e208_mpick.py $OUT $CAL"
if [ -s "$CAL" ]; then
  echo "[동작점 규칙] 이미 적용됨(다시 적용하지 않음 — 결과 뒤 조정 없음): $(cat "$CAL")"
else
  $MP first > "$OUT/mpick_first.out" 2>&1; cat "$OUT/mpick_first.out"
  if grep -q '^NEED_UP$' "$OUT/mpick_first.out"; then
    for T in -1000 -700 -400 -100 200 500 800 1100 1400 1700; do
      [ "$T" -lt 0 ] && NM="G0_m${T#-}" || NM="G0_p$T"
      for X in X Y; do mrun $NM CW $T $X; done
    done
    $MP up > "$OUT/mpick_up.out" 2>&1; cat "$OUT/mpick_up.out"
    if grep -q '^NEED_UPWIDE$' "$OUT/mpick_up.out"; then
      for T in 2000 2300 2600 2900 3200; do for X in X Y; do mrun G0_p$T CW $T $X; done; done
      $MP upwide > "$OUT/mpick_upwide.out" 2>&1; cat "$OUT/mpick_upwide.out"
    fi
  elif grep -q '^NEED_DOWN$' "$OUT/mpick_first.out"; then
    for T in -1600 -1900 -2200 -2500 -2800 -3100 -3400 -3700 -4000 -4300; do
      for X in X Y; do mrun G0_m${T#-} CW $T $X; done
    done
    $MP down > "$OUT/mpick_down.out" 2>&1; cat "$OUT/mpick_down.out"
    if grep -q '^NEED_DOWNWIDE$' "$OUT/mpick_down.out"; then
      for T in -4600 -4900 -5200 -5500 -5800; do for X in X Y; do mrun G0_m${T#-} CW $T $X; done; done
      $MP downwide > "$OUT/mpick_downwide.out" 2>&1; cat "$OUT/mpick_downwide.out"
    fi
  fi
fi
TF=$(grep -oE '^T_F=-?[0-9]+' "$CAL" 2>/dev/null | cut -d= -f2)
[ -n "$TF" ] || { echo "[실패] T_F0 를 정하지 못함"; echo "[E208 경로 검사] 종료"; exit 1; }
echo "[동작점] T_F0 $TF"
# (l)·(r) 학습 — F15 + 맥락 + 게이팅 + T_F + 새 세계·시행열
for spec in "l catctx -" "rx catrule -" "rv catrule rev" "rp catrule p2"; do
  set -- $spec; N=$1; TK=$2; AD=$3
  EXTRA="--rc-seed 38300 --rc-trial-seed 38515"
  [ "$TK" = "catctx" ] && EXTRA="$EXTRA --rc-ctx-seed 38615" || EXTRA="$EXTRA --rc-rule-ctx X"
  [ "$AD" = "rev" ] && EXTRA="$EXTRA --rc-reverse-after 20"; [ "$AD" = "p2" ] && EXTRA="$EXTRA --rc-phase2-after 20"
  f="$OUT/${N}_train_b15.log"
  timeout 1800 python reflex_override_task.py $B0 --kc-motor-eta 2.4 --env-seed 2 $ACT $FEAT --feat-weights $F15 $CW $GATE --motor-tonic $TF --rc-task $TK --episodes 2 --steps 20 \
    --transplant-eval --brain-seed 15 --kc-type-weights $KW15 --save-weights $WD/w_${N}_b15.npz $LRN $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) $TK $AD rc=$rc $(date '+%T')] $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f" | cut -c1-120) | $(grep '^\[범주 규칙 맥락\] 시행' "$f") | F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$f") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$f" | sort -u | tr '\n' ' ')"
  grep '^\[맥락 범주 점검\]\|^\[E204 맥락 게이팅 점검\]\|^\[범주 과제\] catrule\|^\[범주 과제\] 시행\|^\[맥락 범주 과제\]\|^\[범주 반전\]\|^\[둘째 범주 쌍\]' "$f" | cut -c1-170 | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$f" | sed 's/^/      /'
done
# (e) 평가
for spec in "el all $WD/w_l_b15.npz X A,B" "en none $WD/w_l_b15.npz Y A,B" "ev all $WD/w_rv_b15.npz X A,B,C,D" "e0 none $WD/w_rv_b15.npz X A,B,C,D"; do
  set -- $spec; N=$1; XM=$2; WF=$3; X=$4; CATS=$5
  g="$OUT/ev_${N}_b15.log"
  timeout 1800 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $F15 $CW $GATE --motor-tonic $TF --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WF --decomp-mode $XM \
    --rc-eval --rc-eval-cats $CATS --rc-eval-ctx $X --rc-seed 38300 --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(ev $N) $XM 맥락 $X 범주 $CATS rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep -oE 'pushed=[0-9]+' "$g" | head -1) | $(grep '^\[E204 평가 세계\]' "$g" | cut -c1-80) | F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$g") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
python3 $R/scripts/e208_path_check_eval.py "$R" "$OUT" "$WD" "$SN15" "$W6L" "$TF"
echo "[E208 경로 검사] 종료 $(date '+%F %T')"
