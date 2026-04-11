#!/bin/bash
# M3 Detour Test — subprocess isolation per seed×condition
# Prevents GPU memory leak from multiple PyGeNN model builds

SCRIPT="/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation/backend/genesis/test_detour.py"
LEARNING_EPS=${1:-5}
SEEDS=${2:-5}
RESULTS_FILE="/tmp/detour_results.txt"

echo "=== M3 Detour Test (isolated, $SEEDS seeds × 3 conditions) ==="
echo "" > "$RESULTS_FILE"

for seed in $(seq 0 $((SEEDS-1))); do
    for cond in revaluation consolidation_only no_replay; do
        echo "  Running seed=$seed cond=$cond..."
        cd ~/pygenn_test && rm -rf forager_brain_CODE
        result=$(python "$SCRIPT" --single-run "$seed:$cond" --learning-episodes $LEARNING_EPS 2>/dev/null | grep "RESULT_JSON:" | sed 's/RESULT_JSON://')
        echo "$seed,$cond,$result" >> "$RESULTS_FILE"
        echo "    → $result"
    done
    echo "  seed $((seed+1))/$SEEDS complete"
done

echo ""
echo "=== All results saved to $RESULTS_FILE ==="
cat "$RESULTS_FILE"
