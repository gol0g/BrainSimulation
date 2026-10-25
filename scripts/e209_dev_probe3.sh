#!/bin/bash
# E209 개발 진단 3(등록 전 — 구현 확인, 학습 없음) — 뇌 15(표본 밖): 구획 되먹임 억제(--kc-inh-compart — KC 억제 뉴런을 게이팅 무리마다 반반,
# KC ↔ 억제 연결은 같은 무리 안 쌍만, 두 방향 가중치 2배) 2 × 2: {끔, 켬} × {맥락 없음, 흥분 0·게이팅·맥락 X}.
# 예측(실행 전에 적음 — 파일 수정 시각 2026-10-25 23:10:58, 실행 23:11:33 [정정 2026-10-25: 처음 적은 '23:15' 는 date 로 확인하지 않은 시각]):
#   끔 = 개발 진단 2 배율 1 로그(logs/E209/dev2/p_{none|X}_s1_b15.log)와 RCKC·RCJ·[E207 측정 맥락] 줄이 정확히 같다(회귀 — 코드 수정이 끔 경로를 바꾸지 않음).
#   켬 = 장치 확인 줄(연결 같음·무리 밖 0·g 2배) 4집단 통과; 맥락 없음 사례당 반응 KC 가 끔(121.0)과 비슷(±15% 안이면 '거의 보존');
#        맥락 X 사례당 반응 KC 가 켬·맥락 없음의 약 절반(비 ≈ 0.5 — 끔은 0.90), 비활성 무리 억제 뉴런 발화 ≈ 0;
#        코사인 겹침(AC·BD)이 끔·X 보다 맥락 없음 쪽으로.
# 읽기: 사례당 반응 KC = RCKC tag=post set=train 줄 n 평균(쪽·사례마다), 겹침 = scripts/posthoc_e208_graded.py brain_stats(합성 14/14).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev3"; mkdir -p "$OUT"
REF="$R/research/experiments/logs/E209/dev2"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev3 && cd /root/e209_dev3
cp $R/backend/genesis/*.py . 2>/dev/null
echo "[E209 개발 진단 3] 코드 md5 $(md5sum forager_brain.py | cut -c1-12) · 과제 $(md5sum reflex_o*.py | cut -c1-12)"
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단 3] 시작 $(date '+%F %T')"
for spec in "none off" "X off" "none on" "X on"; do
  set -- $spec; C=$1; M=$2
  [ "$C" = "X" ] && EX="--rc-measure-ctx-on X" || EX=""
  [ "$M" = "on" ] && CP="--kc-inh-compart" || CP=""
  g="$OUT/p_${C}_${M}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE $CP --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $F15 $EX > "$g" 2>&1; rc=$?
  echo "[($C·구획 $M) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · 사례당 반응 KC(post·train) $(grep '^RCKC tag=post' "$g" | grep 'set=train' | grep -oE ' n=[0-9]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f(n=%d)", s/n, n; else print "없음"}')"
  grep -E '^\s*\[E209 구획' "$g" | sed 's/^/    /'
  grep '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  if [ "$M" = "off" ]; then
    r="$REF/p_${C}_s1_b15.log"
    if [ -f "$r" ]; then
      a=$(grep -E '^(RCKC |RCJ |\[E207 측정 맥락\])' "$g" | md5sum | cut -c1-12); b=$(grep -E '^(RCKC |RCJ |\[E207 측정 맥락\])' "$r" | md5sum | cut -c1-12)
      na=$(grep -cE '^(RCKC |RCJ |\[E207 측정 맥락\])' "$g")
      [ "$a" = "$b" ] && echo "    회귀(개발 진단 2 배율 1 과 RCKC·RCJ·맥락 줄 $na 줄): 같음 $a" || echo "    회귀: 다름 $a 대 $b"
    else
      echo "    회귀 기준 로그 없음: $r"
    fi
  fi
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단 3] 겹침(posthoc_e208_graded.brain_stats — AC·BD 두 쪽)"
python3 - "$OUT" <<'PYEOF'
import os, sys
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import posthoc_e208_graded as P
out = sys.argv[1]
for c in ("none", "X"):
    for m in ("off", "on"):
        p = os.path.join(out, "p_%s_%s_b15.log" % (c, m))
        try:
            st = P.brain_stats(p)
            print("  %s·구획 %s: S 집합 %d · 가중 %d · 코사인 %.3f | A·B 코사인 %.3f · 자기 점검 %s" % (c, m, st["S_set"], st["S_w"], st["cos"], st["AB_cos"], st["self_check"]))
        except Exception as e:
            print("  %s·구획 %s: 실패 %s" % (c, m, e))
PYEOF
echo "[E209 개발 진단 3] 종료 $(date '+%F %T')"
