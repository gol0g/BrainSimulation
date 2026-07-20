#!/bin/bash
# WSL 환경 재구축 — 2026-07 디스크 사고로 소실된 실행 환경 복원
# 원래 구성 재현: Ubuntu-24.04 + CUDA 12.3 + ~/pygenn_wsl venv + PyGeNN 5.4.0
set -u

echo "=== [1/4] venv 생성 ==="
if [ ! -d "$HOME/pygenn_wsl" ]; then
    python3 -m venv "$HOME/pygenn_wsl"
fi
source "$HOME/pygenn_wsl/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel

echo "=== [2/4] numpy ==="
python -m pip install -q numpy
python -c "import numpy; print('numpy', numpy.__version__)"

echo "=== [3/4] PyGeNN 5.4.0 ==="
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64
if [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
    echo "!! nvcc 없음 ($CUDA_PATH/bin/nvcc) — CUDA 설치 미완"
    exit 3
fi
"$CUDA_PATH/bin/nvcc" --version | tail -2
python -m pip install pygenn==5.4.0 2>&1 | tail -5

echo "=== [4/4] 검증 ==="
mkdir -p "$HOME/pygenn_test"
python -c "import pygenn; print('pygenn', pygenn.__version__)"
echo "OK: 환경 준비 완료"
