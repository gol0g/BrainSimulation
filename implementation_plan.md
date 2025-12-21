# 프로젝트 창세기 (Project Genesis): 구현 계획

## 개요
프로젝트 창세기는 단일 뉴런에서 시작하여 의식적 주체에 이르는 생물학적으로 타당한 뇌 시뮬레이션을 구축하는 것을 목표로 합니다. 이 시스템은 계산 효율성과 풍부한 동역학을 위해 **Izhikevich 뉴런 모델**을 사용합니다. 시뮬레이션은 Python FastAPI 백엔드에서 실행되며, 고충실도 React 프론트엔드를 통해 시각화됩니다.

## 구성 요소

### 백엔드 (`/backend`)
- **기술 스택**: Python, FastAPI.
- **역할**:
    - 시뮬레이션 상태 관리 (뉴런, 시냅스).
    - 시분할(Time-step) 방식의 미분 방정식 계산 (Euler 적분).
    - 프론트엔드에서 현재 상태를 조회할 수 있는 API 엔드포인트 제공.
- **핵심 모듈**:
    - `neuron.py`: `BiologicallyPlausibleNeuron` 클래스 정의 (Izhikevich 모델).
    - `simulation.py`: 전역 시간 및 뉴런 집합 관리.
    - `main.py`: API 계층.

### 프론트엔드 (`/frontend`)
- **기술 스택**: React, Vite, Chart.js (또는 Canvas API).
- **역할**:
    - 백엔드를 폴링하여 밀리초 단위의 전위 변화 수신.
    - "의료 등급"의 오실로스코프 스타일로 막전위(Membrane Potential) 시각화.
    - 전류 주입(자극)을 위한 제어 패널 제공.

## 1단계: 단일 뉴런 (현재 작업)
1.  **백엔드**: 상태 변수(`v`, `u`)를 가진 `IzhikevichNeuron` 클래스 구현. (완료)
2.  **백엔드**: 시뮬레이션을 `dt` 시간만큼 진행하는 `/step` 엔드포인트 생성. (완료)
3.  **프론트엔드**: 전압 그래프를 보여주는 다크 모드/SF 스타일의 인터페이스 제작.

## 향후 단계
- 2단계: 시냅스 전달 (Synaptic transmission).
- 3단계: STDP 학습 (Hebbian learning).
- 4단계: 네트워크 구조 및 창발적 행동.
