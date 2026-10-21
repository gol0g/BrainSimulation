#!/bin/bash
# E206 [정정 1] 뒤 경로 검사 다시 — 처음 동작점 결과 보존 → 격자 확장(scripts/e206_calib_ext.sh) → (l)·(r)·(e) 를 새 T_F 로 다시(형성·처음 격자 측정은 그대로 씀).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
L="$R/research/experiments/logs/E206"; P="$L/pathcheck"; W="$R/research/experiments/traces/E206/pathcheck"
[ -s "$L/calib_tonic_grid1.txt" ] || mv "$L/calib_tonic.txt" "$L/calib_tonic_grid1.txt"
bash "$R/scripts/e206_calib_ext.sh" > "$L/calib_ext.out" 2>&1
grep -q '^T_F=' "$L/calib_tonic.txt" 2>/dev/null || { echo "[실패] 격자 확장 뒤 T_F 없음 — calib_ext.out 확인"; exit 1; }
mkdir -p "$P/run1_T2200" "$W/run1_T2200"
for n in l rx rv rp; do
  [ -e "$P/${n}_train_b15.log" ] && mv "$P/${n}_train_b15.log" "$P/run1_T2200/"
  [ -e "$W/w_${n}_b15.npz" ] && mv "$W/w_${n}_b15.npz" "$W/run1_T2200/"
done
for n in el en ev e0; do [ -e "$P/ev_${n}_b15.log" ] && mv "$P/ev_${n}_b15.log" "$P/run1_T2200/"; done
bash "$R/scripts/e206_path_check.sh" > "$L/path_check2.out" 2>&1
