#!/bin/bash
# E209 개발 진단 6(등록 전 — 구현 확인, 보상 학습 없음) — 뇌 15(표본 밖). 진단 5: 재배선 판은 탈억제를 막았으나(맥락 X/없음 0.55, 맥락 X 겹침 ≈ 자기 맥락 없음)
# 공유 고리에서 형성한 F 의 맥락 없음 겹침을 올렸다(코사인 0.189 → 0.360) — 표현을 그 구조에서 형성해야 하는지 본다(재배선 + 재형성 = F_R).
# (f0) 형성 회귀: E206 경로 검사 (f) 형성 커맨드 + 맥락 집단·게이팅(흥분 0, 노출 중 맥락 끔)·재배선 끔 → RCKC·RCJ·RCCHK 줄이 E206 형성 로그(form_f_b15)와 같아야
#      (조용한 맥락 집단·게이팅 시냅스가 형성을 바꾸지 않음 — E209 본실험 형성의 전제). 다르면 그 차이를 적고 F_R 비교는 (f0) 기준으로.
# (f)  재배선 형성: 같은 커맨드 + --kc-inh-rewire → F_R = traces/E209/dev/feat_formed_R_b15.npz
# (n)·(x) F_R 측정: 재배선 × 맥락 없음·맥락 X(진단 5 와 같은 커맨드에서 F 만 F_R).
# 문턱(실행 전에 적음 — 시각은 이 파일의 수정 시각):
#   (가) 형성 성공: F_R 의 AC·BD J_post 네 값 ≤ 0.1000(E206 형성 문턱).
#   (나) 맥락 없음 보존: 사례당 반응 KC F_R·없음 / 공유 F·없음(121.0) 0.80~1.25, 코사인 F_R·없음 ≤ 공유 F·없음(0.189) + 0.10.
#   (다) 탈억제 차단: 사례당 반응 KC F_R·X / F_R·없음 ≤ 0.70.
#   (라) 학습 조건 겹침: 남은 몫 q = (F_R·X − 공유 F·없음)/(공유 F·X − 공유 F·없음) — q ≤ 0.6667(겹침 남음 아님)이면 다음 단계(사전등록), 넘으면 구획 후보 기각(구현 불가 — 미해결).
#   (마) 게이팅: F_R·X 의 비활성 무리 KC 발화 ≤ 활성 × 0.10.
#   (보고) 간격 기저 KC 발화(RCCHK base), 억제 뉴런 무리별 발화, 허브(범주 ≥ 3) 수.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev6"; WD="$R/research/experiments/traces/E209/dev"; mkdir -p "$OUT" "$WD"
F6L="$R/research/experiments/logs/E206/pathcheck/form_f_b15.log"
D3="$R/research/experiments/logs/E209/dev3"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F0="$WD/feat_formed_0_b15.npz"; FR="$WD/feat_formed_R_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev6 && cd /root/e209_dev6
cp $R/backend/genesis/*.py . 2>/dev/null
echo "[E209 개발 진단 6] 코드 md5 $(md5sum forager_brain.py | cut -c1-12) · 과제 $(md5sum reflex_o*.py | cut -c1-12) · ctx_gate $(md5sum ctx_gate.py | cut -c1-12)"
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
FORM="--feat-oja --feat-oja-eta 0 --decomp-weights $WB --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단 6] 시작 $(date '+%F %T')"
for spec in "f0 - $F0" "f rw $FR"; do
  set -- $spec; N=$1; M=$2; FO=$3
  [ "$M" = "rw" ] && CP="--kc-inh-rewire" || CP=""
  g="$OUT/form_${N}_b15.log"
  if [ -s "$FO" ] && grep -q '^=> RAWCAT ' "$g" 2>/dev/null; then echo "[($N) 형성 — 이미 있음, 건너뜀]"; continue; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CW $GATE $CP $FORM --rc-save $FO > "$g" 2>&1; rc=$?
  echo "[($N) 형성 rc=$rc $(date '+%T')] $(grep '^\[원시 특징 노출\]' "$g" | grep -oE '제시 [0-9]+') | RCJ $(grep -c '^RCJ ' "$g") | $(grep '^RCCHK tag=post' "$g" | grep -oE 'base_l=[0-9]+ base_r=[0-9]+')"
  grep -E '^RCJ side=[lr] pair=(AC|BD|AB)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
a=$(grep -E '^(RCKC |RCJ |RCCHK )' "$OUT/form_f0_b15.log" | md5sum | cut -c1-12); b=$(grep -E '^(RCKC |RCJ |RCCHK )' "$F6L" | md5sum | cut -c1-12)
na=$(grep -cE '^(RCKC |RCJ |RCCHK )' "$OUT/form_f0_b15.log")
[ "$a" = "$b" ] && echo "  (f0) 형성 회귀(E206 경로 검사 (f) 와 RCKC·RCJ·RCCHK $na 줄): 같음 $a" || echo "  (f0) 형성 회귀: 다름 $a 대 $b — 차이 줄 $(diff <(grep -E '^(RCKC |RCJ |RCCHK )' "$OUT/form_f0_b15.log") <(grep -E '^(RCKC |RCJ |RCCHK )' "$F6L") | grep -c '^[<>]')"
python3 -c "import numpy as np; A=np.load('$F0'); B=np.load('$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz'); print('  (f0) 형성 가중치 = E206 F15:', all(np.array_equal(A[k], B[k]) for k in A.files) and set(A.files) == set(B.files))"
for C in none X; do
  [ "$C" = "X" ] && EX="--rc-measure-ctx-on X" || EX=""
  g="$OUT/p_${C}_FR_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --kc-inh-rewire --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $FR $EX > "$g" 2>&1; rc=$?
  echo "[(F_R·$C·재배선) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · 사례당 반응 KC(post·train) $(grep '^RCKC tag=post' "$g" | grep 'set=train' | grep -oE ' n=[0-9]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f(n=%d)", s/n, n; else print "없음"}') · F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$g") · $(grep '^RCCHK tag=post' "$g" | grep -oE 'base_l=[0-9]+ base_r=[0-9]+')"
  grep -E '^\[E209 구획 억제 측정\]|^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단 6] 겹침·허브(posthoc_e208_graded.brain_stats — AC·BD 두 쪽, 둘째 측정; 허브 = 왼쪽 범주 ≥ 3 반응 KC)"
python3 - "$OUT" "$D3" <<'PYEOF'
import os, sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import posthoc_e208_graded as P
out, d3 = sys.argv[1:3]
rows = {}
for nm, p in (("공유 F·없음(진단 3)", os.path.join(d3, "p_none_off_b15.log")), ("공유 F·X(진단 3)", os.path.join(d3, "p_X_off_b15.log")),
              ("재배선 F_R·없음", os.path.join(out, "p_none_FR_b15.log")), ("재배선 F_R·X", os.path.join(out, "p_X_FR_b15.log"))):
    try:
        st = P.brain_stats(p)
        cases, _ = P.read(p)
        ncat = np.zeros(3000, int)
        for c in "ABCD":
            cnt = np.zeros(3000)
            for s_ in cases[("l", c)]:
                for k in s_: cnt[k] += 1
            ncat += (cnt >= 3)
        n = [len(s) for v in cases.values() for s in v]
        rows[nm] = (st["cos"], sum(n) / len(n))
        print("  %s: S 집합 %d · 가중 %d · 코사인 %.3f | A·B 코사인 %.3f | 사례당 반응 KC %.1f · 허브 %d" % (nm, st["S_set"], st["S_w"], st["cos"], st["AB_cos"], sum(n) / len(n), int((ncat >= 3).sum())))
    except Exception as e:
        print("  %s: 실패 %s" % (nm, e))
try:
    cn, cx = rows["공유 F·없음(진단 3)"][0], rows["공유 F·X(진단 3)"][0]
    fn, fx = rows["재배선 F_R·없음"], rows["재배선 F_R·X"]
    print("  문턱: (나) 없음 반응 비 %.2f(0.80~1.25) · 없음 코사인 %.3f ≤ %.3f · (다) X/없음 %.2f(≤ 0.70) · (라) q %.3f(≤ 0.6667)"
          % (fn[1] / rows["공유 F·없음(진단 3)"][1], fn[0], cn + 0.10, fx[1] / fn[1], (fx[0] - cn) / (cx - cn)))
except Exception as e:
    print("  문턱 계산 실패 %s" % e)
PYEOF
echo "[E209 개발 진단 6] 종료 $(date '+%F %T')"
