# 초파리 뇌 시뮬레이션 (Drosophila Brain Simulation)

> 조사일: 2026-03-20

---

## 1. 연구 개요

초파리 성체 뇌의 전체 connectome을 매핑하고, 이를 LIF 뉴런 모델로 시뮬레이션하여 행동을 재현한 연구.

### 3단계 성과

| 시기 | 달성 | 팀 | 논문 |
|------|------|-----|------|
| 2024.10 | 성체 초파리 전뇌 배선도 완성 | FlyWire Consortium (Princeton 등 146개 연구실) | Nature 9편 패키지 |
| 2024.10 | LIF 전뇌 시뮬레이션, 감각운동 95% 정확도 | Philip Shiu | Nature vol.634, pp.210-219 |
| 2026.03 | 가상 몸체 연결, 걷기/그루밍 행동 창발 | Eon Systems | NeuroMechFly v2 + MuJoCo |

---

## 2. 연구팀

### FlyWire Connectome
- **주저자**: Sven Dorkenwald (Princeton → UW / Allen Institute)
- **공저자**: Arie Matsliah (Princeton PNI)
- **기관**: Princeton (Murthy & Seung labs), MRC LMB (Jefferis Lab), UVM (Bock lab), Allen Institute, Harvard Medical School
- **규모**: 146개 연구실, 122개 기관

### 전뇌 계산 모델
- **주저자**: Philip Shiu
- **DOI**: 10.1038/s41586-024-07763-9

### 체화된 전뇌 에뮬레이션
- **팀**: Eon Systems (Philip Shiu 소속)
- **발표**: 2026-03-07
- Shiu 뇌 모델 + NeuroMechFly v2 (EPFL) + MuJoCo 물리 시뮬레이션 통합

---

## 3. 규모 및 기술

| 항목 | 수치 |
|------|------|
| **뉴런 수** | ~139,255 (connectome) / 127,400 (시뮬레이션) |
| **시냅스 수** | ~5천만 (50M) |
| **뉴런 모델** | Leaky Integrate-and-Fire (LIF) |
| **시뮬레이터** | Brian2 (Python SNN) |
| **몸체** | NeuroMechFly v2 + MuJoCo |
| **하드웨어** | 노트북에서 실행 가능 |
| **신경전달물질** | ML로 추정하여 시냅스 부호(흥분/억제) 결정 |

### 추가
- 2025.08: Intel Loihi 2 뉴로모픽 칩 위에 전체 connectome 구현 (arXiv:2508.16792)

---

## 4. 재현된 행동

2026.03 Eon Systems 데모 (프로그래밍이 아닌 connectome에서 **창발**):

- **걷기**: DNa01/DNa02 (회전), oDN1 (전진 속도) 뉴런 통해 보행 패턴
- **조향**: 감각 입력에 따른 방향 전환
- **안테나 그루밍**: 가상 먼지 → 기계감각 뉴런 활성 → 앞다리로 안테나 닦기
- **먹이 탐색**: 미각 단서 이용한 음식 방향 이동

---

## 5. 시뮬레이션 방법론

- **SNN 기반**: 각 뉴런 = 2개 1차 ODE (LIF)
- **시냅스**: 단순화 — 같은 신경전달물질 유형은 동일한 계산 속성
- **sensory-motor loop**: 감각 → 뇌 활성화 전파 → 하행 뉴런 → 모터 → MuJoCo 물리 → 감각 업데이트

### 주요 한계

| 누락 항목 | 영향 |
|-----------|------|
| **학습/가소성 없음** | 가장 큰 한계. 경험으로부터 적응 불가 |
| 신경조절물질 없음 | 도파민/세로토닌 등 보상/동기 시스템 부재 |
| 내부 상태 없음 | 배고픔, 각성, 짝짓기 상태 등 |
| 수상돌기 비선형성 없음 | 단일 구획 뉴런 모델 |
| 이온 채널 다양성 없음 | LIF 단순화 |
| 하행 뉴런 일부만 모델링 | 1,000+ 중 일부 |

---

## 6. 공개 자료

| 리소스 | URL |
|--------|-----|
| 코드 (GitHub) | https://github.com/philshiu/Drosophila_brain_model |
| 모델 출력 데이터 | https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.CZODIW |
| FlyWire Connectome | https://flywire.ai/ |
| Eon Systems GitHub | https://github.com/eonsystemspbc |
| NeuroMechFly v2 | Nature Methods (DOI: 10.1038/s41592-024-02497-y) |
| 논문 | Nature vol.634, DOI: 10.1038/s41586-024-07763-9 |

---

## 7. Genesis Brain과의 비교

| 항목 | Genesis Brain | Drosophila Simulation |
|------|--------------|----------------------|
| **뉴런** | 20,710 | 127,400 (6배) |
| **시냅스** | 41 학습 그룹 + static | 5천만 (1:1 매핑) |
| **학습** | R-STDP, Hebbian, Anti-Hebbian, DA | **없음** |
| **도파민** | VTA, RPE, dip | 없음 |
| **내부 상태** | hunger/satiety/fear | 없음 |
| **접근법** | 원리 재현 → 학습에서 행동 창발 | 구조 복사 → connectome에서 행동 창발 |
| **몸체** | 2D ForagerGym | 3D MuJoCo |
| **GPU** | PyGeNN + CUDA | CPU (Brian2) |

### 핵심 차이

- **초파리**: "뇌를 있는 그대로 복사" — 구조적 충실도 높지만 학습 불가
- **Genesis Brain**: "뇌의 원리를 재현" — 학습/적응 가능하지만 해부학적 충실도 낮음
- 궁극적으로는 **정확한 connectome + 학습/가소성**의 결합이 다음 단계

### Genesis Brain 우위
- 학습/가소성 (41 학습 시냅스 그룹)
- 신경조절 시스템 (도파민, RPE)
- 내부 상태 (항상성, 공포, 기억)
- 보상 기반 의사결정

### 초파리 시뮬레이션 우위
- 생물학적 구조 정확성
- 실제 동물 행동과 비교 가능
- 해부학적 충실도
- 3D 체화된 행동

---

## 8. 시사점

1. **LIF로 충분하다**: 139K LIF 뉴런으로 복잡한 행동 창발 가능 → 뉴런 모델 복잡도보다 연결 구조가 중요
2. **노트북 실행 가능**: 규모가 커도 효율적 시뮬레이션 가능 (Brian2)
3. **학습이 없는 한계**: connectome만으로는 적응적 행동 불가 → Genesis Brain의 학습 접근이 보완적
4. **3D 체화의 가치**: MuJoCo 물리 시뮬레이션이 더 풍부한 행동 창발을 가능하게 함
5. **환경 고도화 필요**: 우리도 ForagerGym을 더 풍부하게 만들어야 학습의 가치가 드러남
