#!/bin/bash
# E207 경로 검사(조건 2) — 표본 밖 뇌 15. 새 코드 --rc-measure-ctx-on(형성 측정 제시 3처리 동안 맥락 켬, 기본 none).
# (a) 회귀: 기본 none 으로 E206 경로 검사 (f) 형성 커맨드를 그대로 다시 → 저장 가중치 배열·RCJ 줄이 E206 경로 검사 form_f_b15 와 같음(결정적 재현)
# (b) F15 + C2 + 게이팅 + 기저값 −1300 + 맥락 X 측정(노출 0) — '[E207 측정 맥락]' pre·post(켠 > 0·끈 0·게이팅 있음·비활성/활성 ≤ 0.01), RCJ 12·J_pre = J_post, 노출 0
# (c) 같은 F15 · 맥락 none 측정 — E207 줄 없음, RCJ 를 E206 형성 로그 post 와 비교(보고)
# (d) U(특징 정적) + 기저값 −2800 + 맥락 X 측정 — 같은 점검
# + 판정기 함수(judge_e207 read_meas·score)로 이 로그들을 한 번 읽기(형식 확인)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E207/pathcheck"; WD="$R/research/experiments/traces/E207/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$R/research/experiments/traces/E206/pathcheck/feat_formed_b15.npz"
FORM6="$R/research/experiments/logs/E206/pathcheck/form_f_b15.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e207_run && cd /root/e207_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
MEAS="--decomp-weights $WB --decomp-mode rawcat --rc-expo 0 --rc-expo-eta 0 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300"
for f_ in "$KW15" "$SN15" "$WB" "$F15" "$FORM6"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E207 경로 검사] 종료"; exit 1; }; done
echo "[E207 경로 검사] 시작 $(date '+%F %T')"
# (a) 회귀 — E206 경로 검사 (f) 와 같은 커맨드(새 코드, 기본 none)
g="$OUT/a_regress_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT --feat-oja --feat-oja-eta 0 \
  --decomp-weights $WB --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-seed 38300 --rc-save $WD/F15_regress.npz > "$g" 2>&1; rc=$?
echo "[(a) 회귀 형성 rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · E207 줄 $(grep -c 'E207 측정 맥락' "$g")"
[ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
# (b)(c)(d) 측정
for spec in "b F X -1300" "c F none -1300" "d U X -2800"; do
  set -- $spec; N=$1; REP=$2; C=$3; T=$4
  EX=""; [ "$REP" = "F" ] && EX="--feat-weights $F15"; [ "$C" = "X" ] && EX="$EX --rc-measure-ctx-on X"
  g="$OUT/${N}_meas_${REP}_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --motor-tonic $T --brain-seed 15 --kc-type-weights $KW15 $MEAS $EX > "$g" 2>&1; rc=$?
  echo "[($N) $REP·$C 측정 rc=$rc $(date '+%T')] RCJ $(grep -c '^RCJ ' "$g") · $(grep '^\[원시 특징 노출\]' "$g" | grep -oE '제시 [0-9]+') · F 적재 $(grep -c '^\[E184 특징 가중치 적재\].* 검증 일치' "$g") · 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ')"
  grep '^\[E207 측정 맥락\]' "$g" | sed 's/^/    /'
  grep -E '^RCJ side=[lr] pair=(AB|AC|BD)' "$g" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$g" | sed 's/^/      /'
done
python3 - "$R" "$OUT" "$WD" "$FORM6" "$F15" <<'PY'
import os, sys
import numpy as np
R, out, wd, form6, f15 = sys.argv[1:6]
sys.path.insert(0, os.path.join(R, "scripts"))
import judge_e207 as J7
def L(fn): return os.path.join(out, fn)
ok = True
# (a)
A, B = np.load(os.path.join(wd, "F15_regress.npz")), np.load(f15)
same_w = sorted(A.files) == sorted(B.files) and all(np.array_equal(A[k], B[k]) for k in A.files)
rj = lambda p: [ln for ln in open(p, encoding="utf-8", errors="replace") if ln.startswith("RCJ ")]
same_j = rj(L("a_regress_b15.log")) == rj(form6) and len(rj(form6)) == 12
no207 = not any("E207 측정 맥락" in ln for ln in open(L("a_regress_b15.log"), encoding="utf-8", errors="replace"))
a_ok = same_w and same_j and no207
ok &= a_ok
print("[검사 (a)] 회귀 — 저장 배열 %d개 같음 %s · RCJ 12줄 같음 %s · E207 줄 없음 %s → %s" % (len(A.files), same_w, same_j, no207, "통과" if a_ok else "실패"))
# (b)(c)(d)
res = {}
for N, rep, c, T in (("b", "F", "X", -1300.0), ("c", "F", "none", -1300.0), ("d", "U", "X", -2800.0)):
    r = J7.read_meas(L("%s_meas_%s_%s_b15.log" % (N, rep, c)))
    res[N] = r
    feat_ok = (len(r["feat"]) >= 1 and all(p and p.replace("\\", "/").endswith("traces/E206/pathcheck/feat_formed_b15.npz") for p in r["feat"])) if rep == "F" else (r["feat"] == [])
    base_ok = (r["rawcat"] == 1 and len(r["rcj"]) == 12 and all(v[0] == v[1] for v in r["rcj"].values()) and set(r["chk"]) == {"pre", "post"}
               and all(v == (288, 0, 0, 0, 1) for v in r["chk"].values()) and r["expo"] == 0 and r["world"] == [(38300, "5,14,15,19,29")]
               and r["tonic"] and all(t == T for t in r["tonic"]) and feat_ok)
    if c == "X":
        cx = r["ctx"]
        ctx_ok = set(cx) == {"pre", "post"} and all(v and v[0] == "X" and v[1] == 288 and v[2] > 0 and v[3] == 0 and v[6] == "있음" and v[4] > 0 and 100 * v[5] <= v[4] for v in cx.values())
    else:
        ctx_ok = not r["ctx"]
    k_ok = base_ok and ctx_ok
    ok &= k_ok
    print("[검사 (%s)] %s·%s — RAWCAT %d · RCJ %d(J_pre=J_post %s) · RCCHK %s · 노출 %s · 세계 %s · 동작점 %s · F 적재 %s · 측정 맥락 %s · S %d → %s"
          % (N, rep, c, r["rawcat"], len(r["rcj"]), all(v[0] == v[1] for v in r["rcj"].values()), sorted(r["chk"].items()), r["expo"], r["world"], sorted(set(r["tonic"])),
             len(r["feat"]), r["ctx"], J7.score(r), "통과" if k_ok else "실패"))
same_bc = [res["b"]["rcj"][(s, p)][1] for s in "lr" for p in J7.PAIRS] == [res["c"]["rcj"][(s, p)][1] for s in "lr" for p in J7.PAIRS]
print("[보고] F15 X 대 none RCJ 12값 같음 %s(같으면 조작 무효 의심) · S_none %d · S_X %d · J(A,B) 좌/우 none %d/%d X %d/%d"
      % (same_bc, J7.score(res["c"]), J7.score(res["b"]), res["c"]["rcj"][("l", "AB")][1], res["c"]["rcj"][("r", "AB")][1], res["b"]["rcj"][("l", "AB")][1], res["b"]["rcj"][("r", "AB")][1]))
ok &= not same_bc
f6 = {}
for ln in open(form6, encoding="utf-8", errors="replace"):
    if ln.startswith("RCJ "):
        kv = dict(x.split("=", 1) for x in ln.split()[1:])
        f6[(kv["side"], kv["pair"])] = J7.q4(kv["J_post"])
eq = [res["c"]["rcj"][k][1] == f6.get(k) for k in res["c"]["rcj"]]
print("[보고] F15 none(C2·게이팅 있음) RCJ 대 E206 형성 로그 post(맥락 집단 없음) — 같음 %d/12 · 차 최대 %d(1e-4)" % (sum(eq), max(abs(res["c"]["rcj"][k][1] - f6.get(k, 0)) for k in res["c"]["rcj"])))
print("[E207 경로 검사] %s" % ("통과" if ok else "실패"))
PY
echo "[E207 경로 검사] 종료 $(date '+%F %T')"
