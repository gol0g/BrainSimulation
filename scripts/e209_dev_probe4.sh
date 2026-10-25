#!/bin/bash
# E209 개발 진단 4(등록 전 — 구현 확인, 보상 학습 없음) — 뇌 15(표본 밖). 개발 진단 3 결과: 구획 되먹임 억제(가중치 2배)는 공유 고리에서 형성한 F 의
# 맥락 없음 범주 겹침도 크게 되살린다(코사인 0.189 → 0.745, 집합 S 1377 → 18082) — F 가 공유 고리 동역학에서 형성됐기 때문인지, 구획 구조 자체가 범주 무관 반응을 만드는지 가른다.
# 절차: E206 형성 커맨드(E206 경로 검사 (f) — 세계 38300 네 범주 노출 망 안 Oja, 측정 가중치 WB, 맥락 없음)를 구획 구조에서 그대로 — 구획은 게이팅 무리가 필요해
#   맥락 집단·게이팅(흥분 0)을 함께 두되 노출 중 맥락은 끈다(맥락 집단 발화 0 → 게이팅 억제 없음). → F_C = traces/E209/dev/feat_formed_C_b15.npz,
#   그 뒤 F_C 로 맥락 없음·맥락 X 측정(E209 개발 진단 3 과 같은 커맨드에서 F 만 F_C).
# 예측·문턱(실행 전에 적음 — 파일 수정 시각 2026-10-25 23:25:41, 실행 23:25:56 [정정 2026-10-25: 처음 적은 '23:30' 은 date 로 확인하지 않은 시각]):
#   (가) 형성 로그 J_pre(구획·형성 전 U)가 E206 경로 검사 (f) 형성 로그 J_pre(공유·U)보다 크면 구획 구조 자체가 겹침을 늘린다는 쪽.
#   (나) 형성 성공 = F_C 의 AC·BD J_post 네 값 모두 ≤ 0.1000(E206 형성 문턱 JMAX). 실패면 '구획 구조에서는 형성이 겹침을 못 없앰' → 구획 후보는 개발 단계에서 기각(구현 불가 — 미해결).
#   (다) (나) 통과 시 F_C·맥락 X 의 코사인이 공유 F·X(0.406)보다 낮고 F_C·맥락 없음 쪽이면 다음 단계(사전등록 — F_C 형성을 본실험 절차에 넣음), 아니면 기각.
#   (라) 사례당 반응 KC: F_C·X / F_C·없음 ≤ 0.70(개발 진단 3 기전 1 문턱 그대로), F_C·없음 / 공유 F·없음(121.0) 0.80~1.25.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev4"; WD="$R/research/experiments/traces/E209/dev"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
FC="$WD/feat_formed_C_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev4 && cd /root/e209_dev4
cp $R/backend/genesis/*.py . 2>/dev/null
echo "[E209 개발 진단 4] 코드 md5 $(md5sum forager_brain.py | cut -c1-12) · 과제 $(md5sum reflex_o*.py | cut -c1-12)"
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단 4] 시작 $(date '+%F %T')"
g="$OUT/form_C_b15.log"
if [ -s "$FC" ] && grep -q '^=> RAWCAT ' "$g" 2>/dev/null; then echo "[형성 — 이미 있음, 건너뜀]"; else
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CW $GATE --kc-inh-compart --feat-oja --feat-oja-eta 0 \
    --decomp-weights $WB --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300 --rc-save $FC > "$g" 2>&1; rc=$?
  echo "[(형성·구획) rc=$rc $(date '+%T')] $(grep '^\[원시 특징 세계\]' "$g" | cut -c1-110) | $(grep '^\[원시 특징 노출\]' "$g" | grep -oE '제시 [0-9]+') | RCJ $(grep -c '^RCJ ' "$g")"
  grep -E '^\s*\[E209 구획 되먹임 억제 확인\]' "$g" | head -1 | cut -c1-200 | sed 's/^/    /'
  grep -E '^RCJ side=[lr] pair=(AC|BD|AB)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
fi
echo "  (대조) E206 경로 검사 (f) 공유 고리 형성 로그:"
grep -E '^RCJ side=[lr] pair=(AC|BD|AB)' "$R/research/experiments/logs/E206/pathcheck/form_f_b15.log" | sed 's/^/    /'
for C in none X; do
  [ "$C" = "X" ] && EX="--rc-measure-ctx-on X" || EX=""
  g="$OUT/p_${C}_FC_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --kc-inh-compart --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $FC $EX > "$g" 2>&1; rc=$?
  echo "[(F_C·$C·구획) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · 사례당 반응 KC(post·train) $(grep '^RCKC tag=post' "$g" | grep 'set=train' | grep -oE ' n=[0-9]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f(n=%d)", s/n, n; else print "없음"}') · F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$g")"
  grep -E '^\[E209 구획 억제 측정\]|^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단 4] 겹침(posthoc_e208_graded.brain_stats — AC·BD 두 쪽, 둘째 측정)"
python3 - "$OUT" "$R/research/experiments/logs/E209/dev3" <<'PYEOF'
import os, sys
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import posthoc_e208_graded as P
out, d3 = sys.argv[1], sys.argv[2]
for nm, p in (("F_C·없음·구획", os.path.join(out, "p_none_FC_b15.log")), ("F_C·X·구획", os.path.join(out, "p_X_FC_b15.log")),
              ("F·없음·공유(개발 진단 3)", os.path.join(d3, "p_none_off_b15.log")), ("F·X·공유(개발 진단 3)", os.path.join(d3, "p_X_off_b15.log"))):
    try:
        st = P.brain_stats(p)
        print("  %s: S 집합 %d · 가중 %d · 코사인 %.3f | A·B 코사인 %.3f" % (nm, st["S_set"], st["S_w"], st["cos"], st["AB_cos"]))
    except Exception as e:
        print("  %s: 실패 %s" % (nm, e))
PYEOF
echo "[E209 개발 진단 4] 종료 $(date '+%F %T')"
