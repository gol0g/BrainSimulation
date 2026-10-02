#!/bin/bash
# E138 경로 검사(조건 2) — 판정 기준 고정(logs/E138/criteria_fixed.txt) 뒤에 돈다.
# (1) 분해 경로가 E119 이식 평가를 재현하는가: 뇌 10 none = [사전] +0.0195, all = [사후] −0.0838 (이미 알려진 값)
# (2) kcpop 이 E119 P2 rev 를 재현하는가: 뇌 10 rev −0.5449 → R = 0.564 (이미 알려진 값)
# (3) 새 모드는 표본 밖 뇌 15(E119 경로 검사 가중치 path_rw0_b15 — [사전] +0.0255, [사후] −0.0734)에서만:
#     kcrate(발화 수 측정·저장, 편측성), kcsel·kcselonly(이식 정확 일치 TE.verify·평가), none(뇌 15 [사전] 재현)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E138/pathcheck"; WD="$R/research/experiments/traces/E138/pathcheck"; mkdir -p "$OUT" "$WD"
TR="$R/research/experiments/traces/E119"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e138_run && cd /root/e138_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
run() {  # 태그, 나머지 = 인자
  local tag="$1"; shift
  local f="$OUT/$tag.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 0 "$@" > "$f" 2>&1; local rc=$?
  local o; o=$(grep -E '^=> (DECOMP|KCRATE)|^\[E138\]' "$f" | cut -c1-400 | tr '\n' ' ')
  if [ -n "$o" ]; then echo "[$tag] $o"; else echo "[$tag 실패 rc=$rc]"; tail -3 "$f"; fi
}
echo "[1 재현 — 기대 none +0.0195, all −0.0838]"
run b10_none --brain-seed 10 --decomp-weights $TR/w_rw0_b10.npz --decomp-mode none
run b10_all --brain-seed 10 --decomp-weights $TR/w_rw0_b10.npz --decomp-mode all
echo "[2 권한 재현 — 기대 kcpop ≈ −0.5449 (R = 0.0195 − mod ≈ 0.564)]"
run b10_kcpop --brain-seed 10 --decomp-weights $TR/w_rw0_b10.npz --decomp-mode kcpop
echo "[3 새 모드 — 표본 밖 뇌 15, 기대 none +0.0255]"
run b15_kcrate --brain-seed 15 --decomp-weights $TR/path_rw0_b15.npz --decomp-mode kcrate --trials 200 --kc-rate-file $WD/rate_b15.npz
run b15_none --brain-seed 15 --decomp-weights $TR/path_rw0_b15.npz --decomp-mode none
run b15_kconly --brain-seed 15 --decomp-weights $TR/path_rw0_b15.npz --decomp-mode kc_only
run b15_kcsel --brain-seed 15 --decomp-weights $TR/path_rw0_b15.npz --decomp-mode kcsel --kc-rate-file $WD/rate_b15.npz
run b15_kcselonly --brain-seed 15 --decomp-weights $TR/path_rw0_b15.npz --decomp-mode kcselonly --kc-rate-file $WD/rate_b15.npz
echo "[E138 경로 검사] 종료"
