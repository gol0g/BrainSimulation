#!/bin/bash
# E183 경로 검사(조건 2) — 기준 logs/E183/criteria_fixed.txt 고정(2026-10-11 21:04:30) 뒤. 표본 밖 뇌 15.
# 새 코드: forager_brain 특징 집단(feat_*: zz_feat_l·r → 같은 쪽 KC, 호스트 연결, Oja eta 동적), 과제 --feat-*·--rc-* 인자·검증, 분해 모드 rawcat.
# (a) 특징 없음 — E182 경로 검사 (a) learn A(맥락 A 연속 평가) → −0.0629 정확 재현(코드 변경이 이전 경로를 안 바꿈).
# (b) 같은 평가 + 특징 집단(조용함, w 16, Oja) → −0.0629 정확(새 집단이 스냅숏 아래 기존 뇌를 안 바꿈), '[원시 특징]' 줄 1.
# (c) rawcat 짧은 O(노출 2 주기, eta 0.02, pre2) → RCCHK pre·pre2·post 충실도·goodbad 0·g_same 1·npres 144, RCW 두 쪽 dg_abs > 0, RAWCAT 줄.
# (d) rawcat 짧은 Z(eta 0) → RCW dg_abs 0·frac_changed 0, 형성 전 RCKC 줄이 (c) 와 같은지(다른 줄 수).
# 1차(21:06:53~21:08:50, logs/E183/path_check_try1.out·pathcheck_try1/): (a)(b) 통과, (c)(d) good·bad 눈 스파이크 기록 미설정(IndexError) → 특징 모드에서만 기록 켬(forager_brain), 전체 재실행.
# 2차(21:09:42~21:15:20, logs/E183/path_check_try2.out·pathcheck_try2/): (a)~(d) 작동, 단 형성 전 g 평균 15.86(초기 16) — 빌드 eta 0.02 로 이식 뇌 워밍업 중 KC 발화마다 Oja 감쇠 → rawcat 은 빌드 eta 0 강제(--feat-oja-eta 0), 노출 전 안정화도 eta 0, RCW g0_init 표시. 3차 전체 재실행.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E183/pathcheck"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e183_run && cd /root/e183_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 4 --ctx-p 0.10 --ctx2-n 200 --ctx2-w 4 --ctx2-p 0.10 --motor-tonic -4000"
FEAT="--feat-n 200 --feat-w 16 --feat-p 0.10 --feat-oja --feat-oja-eta 0"
for f_ in "$KW15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E183 경로 검사] 종료"; exit 1; }; done
echo "[E183 경로 검사] 시작 $(date '+%F %T')"
g="$OUT/a_nofeat_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WB --decomp-mode all --eval-ctx2 --eval-diag > "$g" 2>&1
echo "[(a) 특징 없음 rc=$?] 원시 특징 줄 $(grep -c '\[원시 특징\]' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+')"
g="$OUT/b_feat_silent_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX $FEAT --decomp-weights $WB --decomp-mode all --eval-ctx2 --eval-diag > "$g" 2>&1
echo "[(b) 특징 집단 조용함 rc=$?] 원시 특징 줄 $(grep -c '\[원시 특징\]' "$g") $(grep -m1 '\[원시 특징\]' "$g" | cut -c1-160) | $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+')"
echo "  기대 (a)(b): mod −0.0629 둘 다 정확, (a) 원시 특징 줄 0·(b) 2(주 뇌·이식 뇌 각 1)"
for ARM in O Z; do
  [ "$ARM" = "O" ] && EE=0.02 || EE=0
  g="$OUT/${ARM,,}_rawcat_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --conn-snapshot $SN15 $FEAT --decomp-weights $WB --decomp-mode rawcat --rc-expo 2 --rc-repeat-pre --rc-expo-eta $EE > "$g" 2>&1
  echo "[(${ARM}) rawcat 짧은 eta $EE rc=$?] RCKC $(grep -c '^RCKC ' "$g") | $(grep '^RCCHK ' "$g" | sed -E 's/ base_l=.*//' | tr '\n' ' ')| $(grep '^RCW ' "$g" | grep -oE 'side=[lr] nsyn=[0-9]+ g_mean0=[0-9.]+ g_mean1=[0-9.]+ dg_abs=[0-9.]+ frac_changed=[0-9.]+ wsel0=[0-9.na]+ wsel1=[0-9.na]+ kc_in_mean0=[0-9.]+ kc_in_mean1=[0-9.]+ g0_init=[01]' | tr '\n' ' ')| 노출 $(grep -c '^\[원시 특징 노출\]' "$g") 완료 $(grep -c '^=> RAWCAT ' "$g")"
done
echo "[(c)·(d) 형성 전 RCKC 줄 차이] $(diff <(grep '^RCKC tag=pre ' "$OUT/o_rawcat_b15.log") <(grep '^RCKC tag=pre ' "$OUT/z_rawcat_b15.log") | grep -c '^<') / $(grep -c '^RCKC tag=pre ' "$OUT/o_rawcat_b15.log")"
echo "  기대 (c)(d): RCW g0_init=1·g_mean0 16(형성 전 g = 초기값), RCKC 144(pre·pre2·post 각 48), RCCHK 3줄 npres=144·feat_active_frac ≥ 0.90·inactive ≤ 1%·other 0·goodbad 0·g_same 1, O dg_abs > 0·Z dg_abs 0, 형성 전 줄 차이 0(또는 극소)"
echo "[E183 경로 검사] 종료 $(date '+%F %T')"
