#!/bin/bash
# E209 개발 진단 5(등록 전 — 구현 확인, 학습 없음) — 뇌 15(표본 밖): 구획 되먹임 억제 재배선 판(--kc-inh-rewire — 무리 밖 연결의 앞 뉴런을 같은 무리 안으로,
# 뒤 뉴런마다 받는 연결 수·가중치 그대로). 개발 진단 3 의 '연결 절반 + 가중치 2배' 판은 KC 마다 받는 억제 수의 편차를 키워 억제를 덜 받는 KC 가 범주 무관 허브가 됐다
# (맥락 없음 코사인 0.189 → 0.745, 허브 KC 억제 입력 평균 9.2 대 전체 16.0). 표현 F 는 E206 공유 고리 형성 그대로(재배선은 받는 수를 지켜 F 와 맞을 것).
# 2 측정: 재배선 × {맥락 없음, 흥분 0·게이팅·맥락 X} — 비교는 개발 진단 3 의 공유(구획 끔) 로그.
# 예측·문턱(실행 전에 적음 — 시각은 이 파일의 수정 시각):
#   (가) 맥락 없음 보존: 사례당 반응 KC 재배선/공유(121.0) 0.80~1.25, 코사인(AC·BD) ≤ 공유 맥락 없음(0.189) + 0.10 — 넘으면 재배선도 표현을 바꿈.
#   (나) 탈억제 차단: 사례당 반응 KC 맥락 X / 맥락 없음 ≤ 0.70(공유 0.90).
#   (다) 겹침: 맥락 X 코사인이 공유 맥락 X(0.406)보다 낮음 — 남은 몫 q = (재배선 X − 공유 없음)/(공유 X − 공유 없음).
#   (라) 허브: 범주 ≥ 3 에 반응하는 KC 수(왼쪽) — 공유 0·연결 절반 판 76 대비.
#   (마) 분리(보고): 맥락 X 에서 비활성 무리 억제 뉴런 발화 / 활성, 비활성 무리 KC 발화(게이팅 새는 몫).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E209/dev5"; mkdir -p "$OUT"
D3="$R/research/experiments/logs/E209/dev3"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e209_dev5 && cd /root/e209_dev5
cp $R/backend/genesis/*.py . 2>/dev/null
echo "[E209 개발 진단 5] 코드 md5 $(md5sum forager_brain.py | cut -c1-12) · 과제 $(md5sum reflex_o*.py | cut -c1-12) · ctx_gate $(md5sum ctx_gate.py | cut -c1-12)"
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 0 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 0 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
echo "[E209 개발 진단 5] 시작 $(date '+%F %T')"
for C in none X; do
  [ "$C" = "X" ] && EX="--rc-measure-ctx-on X" || EX=""
  g="$OUT/p_${C}_rw_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --kc-inh-rewire --motor-tonic -1300 --brain-seed 15 --kc-type-weights $KW15 $MEAS --feat-weights $F15 $EX > "$g" 2>&1; rc=$?
  echo "[($C·재배선) rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · 사례당 반응 KC(post·train) $(grep '^RCKC tag=post' "$g" | grep 'set=train' | grep -oE ' n=[0-9]+' | cut -d= -f2 | awk '{s+=$1; n++} END {if (n) printf "%.1f(n=%d)", s/n, n; else print "없음"}')"
  grep -E '^\s*\[E209 구획' "$g" | sort -u | sed 's/^/    /'
  grep -E '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
echo "[E209 개발 진단 5] 겹침·허브(posthoc_e208_graded.brain_stats·범주 ≥ 3 반응 KC — 왼쪽)"
python3 - "$OUT" "$D3" "$SN15" <<'PYEOF'
import os, sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import posthoc_e208_graded as P
out, d3, sn = sys.argv[1:4]
S = np.load(sn)
indeg = np.bincount(S["kc_inh_l_to_kc_l__post"].astype(int), minlength=3000)
for nm, p in (("공유·없음(진단 3)", os.path.join(d3, "p_none_off_b15.log")), ("공유·X(진단 3)", os.path.join(d3, "p_X_off_b15.log")),
              ("절반+2배·없음(진단 3)", os.path.join(d3, "p_none_on_b15.log")), ("재배선·없음", os.path.join(out, "p_none_rw_b15.log")),
              ("재배선·X", os.path.join(out, "p_X_rw_b15.log"))):
    try:
        st = P.brain_stats(p)
        cases, _ = P.read(p)
        ncat = np.zeros(3000, int)
        for c in "ABCD":
            cnt = np.zeros(3000)
            for s_ in cases[("l", c)]:
                for k in s_: cnt[k] += 1
            ncat += (cnt >= 3)
        hubs = np.nonzero(ncat >= 3)[0]
        print("  %s: S 집합 %d · 가중 %d · 코사인 %.3f | A·B 코사인 %.3f | 왼쪽 반응 KC %d · 허브(범주 ≥ 3) %d(스냅숏 억제 입력 평균 %s)"
              % (nm, st["S_set"], st["S_w"], st["cos"], st["AB_cos"], int((ncat >= 1).sum()), hubs.size, ("%.1f" % indeg[hubs].mean()) if hubs.size else "—"))
    except Exception as e:
        print("  %s: 실패 %s" % (nm, e))
PYEOF
echo "[E209 개발 진단 5] 종료 $(date '+%F %T')"
