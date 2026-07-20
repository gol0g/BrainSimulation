#!/bin/bash
# CUDA/PyGeNN 공통 환경 — 모든 실행 스크립트가 source 한다.
#
# NVCC_PREPEND_FLAGS 필수:
#   CUDA 12.3은 gcc 12까지만 지원하는데 Ubuntu 24.04 기본은 gcc 13/14다.
#   지정하지 않으면 host_config.h:143 "unsupported GNU version!"로
#   optimizeBlockSize 단계에서 NVCC가 죽는다.
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-12'
export PYTHONUNBUFFERED=1
