#!/bin/bash
# FlyWire 뉴런 분류(유형/영역) 받기. Shiu repo에는 ID만 있고 유형 라벨이 없다.
set -u
cd /mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/data/flywire
URLS="
https://storage.googleapis.com/flywire-data/codex/data/fafb/783/classification.csv.gz
https://storage.googleapis.com/flywire-data/codex/data/fafb/783/cell_stats.csv.gz
https://storage.googleapis.com/flywire-data/codex/data/fafb/630/classification.csv.gz
"
for U in $URLS; do
  N=$(basename "$U")
  echo "시도: $N"
  if timeout 120 curl -sfL -o "$N" "$U"; then
    SZ=$(stat -c%s "$N" 2>/dev/null || echo 0)
    echo "  성공: $N ($SZ bytes)"
  else
    echo "  실패"
    rm -f "$N"
  fi
done
ls -la *.gz 2>/dev/null || echo "(받은 파일 없음)"
