#!/bin/bash
# WSL 환경 재구축 — 2026-07 디스크 사고로 소실된 실행 환경 복원
# 원래 구성 재현: Ubuntu-24.04 + CUDA 12.3 + ~/pygenn_wsl venv + PyGeNN 5.4.0
#
# set -e 필수: 이전 판은 pygenn 설치가 실패했는데도 마지막 echo가 실행돼
# "OK: 환경 준비 완료"를 출력했다. 실패를 성공으로 보고하는 스크립트는
# 없는 것만 못하다.
set -euo pipefail

CUDA_PATH=/usr/local/cuda-12.3
export CUDA_PATH
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64"

echo "=== [1/5] CUDA 확인 ==="
if [ ! -x "$CUDA_PATH/bin/nvcc" ]; then
    echo "!! nvcc 없음 ($CUDA_PATH/bin/nvcc)" >&2
    exit 3
fi
"$CUDA_PATH/bin/nvcc" --version | tail -2

echo "=== [2/5] 빌드 의존성 ==="
export DEBIAN_FRONTEND=noninteractive
# pkg-config + libffi-dev 필수: GeNN setup.py가 pkgconfig.parse("libffi")로
# libffi 링크 설정을 읽는다. pkg-config 바이너리가 없으면 Popen이 터지면서
# "Getting requirements to build wheel" 단계에서 실패한다.
apt-get install -y -qq git swig pkg-config libffi-dev >/dev/null

echo "=== [3/5] venv + numpy ==="
if [ ! -d "$HOME/pygenn_wsl" ]; then
    python3 -m venv "$HOME/pygenn_wsl"
fi
source "$HOME/pygenn_wsl/bin/activate"
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q numpy
python -c "import numpy; print('numpy', numpy.__version__)"

echo "=== [4/5] PyGeNN 5.4.0 소스 빌드 ==="
# PyPI에 pygenn 패키지가 없다(2026-07 확인: pypi.org/pypi/pygenn/json → Not Found).
# GeNN 5.4.0 릴리스에도 wheel 자산이 없어 소스 빌드가 유일한 경로다.
SRC="$HOME/genn-src"
if [ ! -d "$SRC" ]; then
    git clone --branch 5.4.0 --depth 1 https://github.com/genn-team/genn.git "$SRC"
fi
cd "$SRC"
python -m pip install .

echo "=== [5/5] 검증 ==="
mkdir -p "$HOME/pygenn_test"
cd "$HOME/pygenn_test"
python -c "import pygenn; print('pygenn', pygenn.__version__)"
echo "OK: 환경 준비 완료"
