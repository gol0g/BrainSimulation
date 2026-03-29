# C. elegans (예쁜꼬마선충) 전체 신경계 시뮬레이션 리서치

> 조사일: 2026-03-29

---

## 1. C. elegans 신경계 규모

### 기본 수치 (성체 자웅동체 hermaphrodite)

| 항목 | 수치 | 출처 |
|------|------|------|
| **뉴런 수** | 302 | White et al. 1986, Cook et al. 2019 |
| **화학 시냅스** | ~6,393 (2,194 연결) | Cook et al. 2019 |
| **Gap Junction (전기 시냅스)** | ~514쌍 (양방향) | Cook et al. 2019 |
| **총 시냅스 연결** | ~6,900+ | Cook et al. 2019 |
| **근육 세포** | 95 (body wall) / 138 (전체) | WormAtlas |
| **체세포 총 수** | 959 | |
| **수컷 뉴런** | 385 (추가 83 뉴런) | Cook et al. 2019 |

### 뉴런 분류

| 유형 | 수 | 역할 |
|------|-----|------|
| 감각 뉴런 (Sensory) | 83 | 화학, 기계, 온도 감각 |
| 개재 뉴런 (Interneuron) | 81 | 정보 통합, 의사결정 |
| 운동 뉴런 (Motor) | 108 | 근육 제어 |
| 인두 뉴런 (Pharyngeal) | 20 | 먹이 섭취 (별도 회로) |
| 기타 (end organs) | ~10 | |

### 다층 연결 구조 ("Multilayer Connectome")

C. elegans는 시냅스 연결만이 아닌 다층적 신경 소통을 한다:

1. **화학 시냅스** (Chemical Synapses): 6,393개 - 방향성 신호 전달
2. **Gap Junction** (Electrical): 514쌍 - 양방향, 동기화
3. **뉴로펩타이드 ("무선 connectome")**: ~300종 뉴로펩타이드 - 시냅스 connectome과 **5%만 중복**
   - 더 밀집, 더 탈중심화, 다른 허브 뉴런
   - 시냅스로 연결 안 된 영역도 연결
   - 행동 상태 전환 (roaming ↔ dwelling) 조절

> **핵심 교훈**: "wired connectome"만으로는 C. elegans의 행동을 완전히 설명할 수 없다. "wireless" 뉴로펩타이드 신호가 동등하게 중요.

---

## 2. 주요 프로젝트 / 연구팀

### 2.1 OpenWorm (2011~현재)

| 항목 | 내용 |
|------|------|
| **설립** | 2011년, 국제 오픈사이언스 프로젝트 |
| **목표** | C. elegans 전체를 세포 수준에서 시뮬레이션 |
| **단계** | 1단계: 302 뉴런 + 95 근육 → 운동 재현 |
| **라이센스** | MIT (전체 오픈소스) |
| **GitHub** | https://github.com/openworm |
| **웹사이트** | https://openworm.org |

**핵심 서브프로젝트:**

| 프로젝트 | 역할 | 기술 |
|----------|------|------|
| **c302** | 신경계 모델링 프레임워크 | Python, NeuroML, 다중 스케일 |
| **Sibernetic** | 물리/유체역학 시뮬레이터 | C++/OpenCL, PCISPH 알고리즘 |
| **owmeta** | 생물학 데이터 관리 | RDF 온톨로지 |
| **WormSim** | 브라우저 인터랙티브 시뮬레이션 | Web |
| **robots** | 물리적 로봇 구현 | Raspberry Pi/ESP32 |

**뉴런 모델:**
- Hodgkin-Huxley (HH) 모델 사용 (conductance-based)
- NeuroML 표준 형식으로 기술
- 단일 구획(single-compartment) 뉴런
- c302는 다중 스케일 지원 (간단한 IF부터 상세 HH까지)

**달성한 것:**
- 전진 운동을 위한 신경-근육 서브셋 모델링
- c302 + Sibernetic 통합 (Docker 컨테이너)
- 근육에서 이동파(travelling wave) 발생 재현
- 물리적 로봇 구현 (서보 기반 분절 체)

**한계:**
- 전체 302 뉴런의 통합 행동 재현은 아직 미완성
- 시냅스 가중치 미지 (connectome은 있지만 강도는 모름)
- 학습/가소성 미구현
- 2025-2026년 기준 대규모 업데이트 뉴스 부족 (유지보수 모드)

---

### 2.2 BAAIWorm / MetaWorm (BAAI, 2024)

| 항목 | 내용 |
|------|------|
| **팀** | Beijing Academy of Artificial Intelligence (BAAI) |
| **발표** | Nature Computational Science, 2024년 12월 |
| **GitHub** | https://github.com/Jessie940611/BAAIWorm |
| **접근법** | 데이터 기반 통합 모델 (brain + body + environment) |

**기술 상세:**

| 구성요소 | 상세 |
|----------|------|
| **뇌 모델** | 136 뉴런 (감각+운동 관련), 다중구획 모델 |
| **구획 크기** | < 2 μm (실제 형태 기반) |
| **몸체** | 3,341 사면체, 96 근육 (head→tail) |
| **환경** | 3D 물리 환경 |
| **프레임 레이트** | 30 FPS 실시간 |

**달성한 것:**
- **지그재그 운동 재현**: 실제 C. elegans의 화학주성(chemotaxis) 중 지그재그 이동 패턴 재현
- **Closed-loop**: brain↔body↔environment 실시간 상호작용
- **시냅스 교란 실험**: 특정 연결을 끊으면 행동 변화 → 인과관계 분석 가능
- **실제 C. elegans 활동 데이터와 유사한 신경 활성 패턴**

**한계:**
- 302 뉴런 중 136만 포함 (감각+운동 서브셋)
- **시냅스 가소성 미구현** (static weights)
- 모듈 구조라 미래에 가소성 추가 가능하다고 언급

---

### 2.3 NeuroSimWorm (2025)

| 항목 | 내용 |
|------|------|
| **발표** | Neurocomputing (ScienceDirect), 2025 |
| **접근법** | 다중감각 closed-loop 프레임워크 |
| **특징** | **최초로 다중감각 회로 통합** |

**4개 서브시스템:**
1. Environment (화학/기계/온도 자극)
2. Neural Computing (신경 회로)
3. Biomechanical Model (몸체 역학)
4. Visualization (시각화)

**4종 신경회로 통합:**
1. Locomotion (운동)
2. Chemosensation (화학 감각)
3. Thermosensation (온도 감각)
4. Mechanosensation (기계 감각)

**달성한 것:**
- 전진, 촉각 수축, 탐색(foraging) 등 복합 행동 재현
- **다중감각 과제 동시 처리** (기존 플랫폼은 운동+화학주성만)
- 실시간 시뮬레이션

---

### 2.4 Connectome-Based Digital Twin (MDPI, 2023)

| 항목 | 내용 |
|------|------|
| **발표** | MDPI Mathematics, 2023 |
| **접근법** | Connectome 기반 + 오프라인 행동 데이터로 학습 |

**기술 상세:**
- 469 노드 (뉴런 + 근육 + 기관)
- 4,869 화학 연결 + 1,433 전기 연결
- PID 컨트롤러로 생성한 화학주성 행동 데이터로 **오프라인 학습**
- 119 뉴런이 사인파 크롤링에 핵심적임을 ablation으로 확인

**주의:** 이것은 STDP/R-STDP 같은 생물학적 학습이 아닌, connectome의 가중치를 행동 데이터로 피팅한 것.

---

### 2.5 Si elegans (FPGA 하드웨어 에뮬레이션)

| 항목 | 내용 |
|------|------|
| **접근법** | 302개 FPGA 각각에 뉴런 1개 모델 탑재 |
| **목표** | 실시간 하드웨어 신경망 에뮬레이션 |
| **체화** | 프로토타입 로봇으로 화학주성 행동 검증 |

---

### 2.6 Nematoduino (Arduino 구현)

| 항목 | 내용 |
|------|------|
| **플랫폼** | Arduino UNO |
| **뉴런 모델** | 간단한 Leaky Integrate-and-Fire |
| **connectome 크기** | 8 KB (압축) |
| **메모리 사용** | 프로그램 40%, SRAM 42% |
| **행동** | 화학주성, 코 터치 회피 |
| **GitHub** | https://github.com/nategri/nematoduino |

> **교훈**: 302 뉴런 LIF + connectome이 Arduino UNO에서 실행 가능. 간단한 모델로도 기본 행동 창발.

---

## 3. 재현된 행동 종합

| 행동 | 재현 여부 | 프로젝트 | 방법 |
|------|-----------|----------|------|
| **전진 운동 (Forward Locomotion)** | O | OpenWorm, BAAIWorm, NeuroSimWorm | Connectome-driven muscle waves |
| **후진 (Reversal)** | O | Nematoduino, Digital Twin | Touch → motor neuron switch |
| **사인파 크롤링 (Sinusoidal)** | O | BAAIWorm, Digital Twin | Body mechanics + neural oscillation |
| **지그재그 화학주성 (Chemotaxis)** | O | BAAIWorm | Closed-loop brain-body-env |
| **Klinotaxis (점진적 방향 전환)** | O | 여러 모델 | Sensory→interneuron→motor |
| **코 터치 회피** | O | Nematoduino, NeuroSimWorm | Mechanosensation circuit |
| **탐색 (Foraging)** | O | NeuroSimWorm | Multi-sensory integration |
| **온도 주성 (Thermotaxis)** | O | NeuroSimWorm | Thermosensation circuit |
| **오메가 턴** | O | ElegansBot (2024) | Rigid body chain model |
| **Roaming ↔ Dwelling 전환** | X | 미구현 | 뉴로펩타이드 필요 |
| **연합 학습 (Associative)** | X | 미구현 (시뮬레이션에서) | 가소성 미구현 |
| **습관화 (Habituation)** | X | 미구현 (시뮬레이션에서) | GLR-1 수용체 조절 필요 |
| **미로 탐색 (Maze)** | X | 미구현 | 다중감각 + 학습 필요 |

---

## 4. 학습/가소성 구현 현황

### 4.1 실제 C. elegans의 학습 능력

생물학적으로 C. elegans는 **다양한 학습을 수행**:

| 학습 유형 | 근거 |
|-----------|------|
| **습관화 (Habituation)** | 반복 자극에 반응 감소, GLR-1 수용체 하향 조절 |
| **민감화 (Sensitization)** | 짧은 ISI에서 초기 반응 증가 후 습관화 |
| **고전적 조건화** | 냄새-음식 연합 학습 (appetitive & aversive) |
| **장기 기억 (LTM)** | 냄새 연합 학습 24시간 유지 |
| **조작적 학습** | 미로에서 음식 위치 학습 (dopamine 의존) |
| **도파민 매개 학습** | DOP-3 수용체 없으면 음식 탐색 학습 실패 |
| **Garcia Effect (맛 혐오)** | 독성 음식 일회 학습으로 회피 |

### 4.2 시뮬레이션에서의 학습 구현 — 거의 없음

```
╔═══════════════════════════════════════════════════════════════╗
║  현재 C. elegans 시뮬레이션 프로젝트 대부분은                  ║
║  시냅스 가소성을 구현하지 않았다.                              ║
║  BAAIWorm (2024 Nature): "does not model synaptic plasticity"  ║
║  OpenWorm: static weights                                     ║
║  NeuroSimWorm: static circuits                                ║
╚═══════════════════════════════════════════════════════════════╝
```

**유일한 예외:**
- STDP 네트워크 연구 (Ren & Kolwankar, 2010): C. elegans connectome 토폴로지에 STDP를 적용했을 때 실제 C. elegans와 유사한 motif 패턴이 나타남 → 하지만 행동 재현과 연결하지 않음
- Digital Twin (2023): 오프라인 학습으로 가중치 피팅 → 생물학적 가소성이 아님

**왜 구현이 어려운가:**
1. **시냅스 강도 데이터 부재**: Connectome은 연결 존재 여부만 알려줌, 실제 강도(weight) 미지
2. **뉴로모듈레이션 복잡성**: 가소성은 dopamine, serotonin 등 조절 필요 → "wireless connectome" 레이어 미구현
3. **교정 기준 부재**: 학습 전후 가중치 변화를 측정한 데이터가 부족
4. **우선순위**: 대부분 프로젝트가 먼저 "static connectome으로 기본 행동 재현"에 집중

---

## 5. Connectome 기반 시뮬레이션의 근본적 한계

### 5.1 "Connectome ≠ Behavior" 문제

같은 해부학적 회로가 **신경조절 상태에 따라 완전히 다른 행동**을 생성:

| 상태 | 행동 | 조절자 |
|------|------|--------|
| Roaming (탐색) | 빠른 직선 이동 | PDF-1 뉴로펩타이드 |
| Dwelling (머무름) | 느린 곡선 이동 | 세로토닌 |
| 기아 (Fasting) | 화학주성 역전 | 인슐린 신호 |
| 교미 (Mating) | 완전히 다른 행동 | 성 페로몬 |

→ **동일 connectome, 다른 행동**. Connectome만으로는 이 상태 전환 불가능.

### 5.2 "Wireless Connectome"의 중요성

2023년 Neuron지에 발표된 연구 (Ripoll-Sanchez et al.):
- **~300종 뉴로펩타이드** 매핑
- 시냅스 connectome과 뉴로펩타이드 connectome은 **5%만 겹침**
- 뉴로펩타이드 네트워크가 더 밀집, 더 탈중심화
- 시냅스로 연결되지 않은 뉴런 간에도 뉴로펩타이드로 소통

→ **시냅스만 모델링하면 신경계의 절반만 보는 것**

### 5.3 C. elegans vs 초파리 비교

| 항목 | C. elegans | 초파리 (Drosophila) | 비율 |
|------|------------|---------------------|------|
| **뉴런** | 302 | ~139,255 | 1 : 461 |
| **시냅스** | ~6,900 | ~50,000,000 | 1 : 7,246 |
| **행동 복잡도** | 기본 반사, 화학주성 | 비행, 보행, 그루밍, 구애, 학습 | |
| **뉴런 모델** | HH (conductance) | LIF | |
| **Connectome 완성** | 1986 (White), 2019 (Cook) | 2024 (FlyWire) | |
| **체화 시뮬레이션** | BAAIWorm (2024) | Eon Systems (2026) | |
| **학습 재현** | 미구현 | Mushroom Body R-STDP | |
| **감각 계층** | 1-2층 | 다층 (V1→V2→...→MB) | |
| **중앙 복합체** | 없음 | Central Complex (내비게이션) | |
| **Mushroom Body** | 없음 | MB (연합 학습 전용) | |

### 5.4 302 뉴런의 한계와 가능성

**가능한 것:**
- 감각-운동 반사 (화학주성, 회피)
- 기본적 의사결정 (approach vs avoid)
- 약한 형태의 기억 (뉴로모듈레이션으로)

**불가능한 것 (connectome-only):**
- 상태 의존적 행동 전환 (뉴로펩타이드 필요)
- 연합 학습 (가소성 필요)
- 장기 기억 (분자 수준 변화 필요)
- 복잡한 공간 탐색 (공간 기억 회로 부재)

---

## 6. 공개 코드/데이터 목록

### 코드

| 프로젝트 | URL | 언어 |
|----------|-----|------|
| OpenWorm (전체) | https://github.com/openworm | Python/C++ |
| c302 | https://github.com/openworm/c302 | Python/NeuroML |
| Sibernetic | https://github.com/openworm/sibernetic | C++/OpenCL |
| BAAIWorm | https://github.com/Jessie940611/BAAIWorm | Python |
| Nematoduino | https://github.com/nategri/nematoduino | C++ (Arduino) |
| OpenWorm Robots | https://github.com/openworm/robots | Python |
| ConnectomeToolbox | https://github.com/openworm/ConnectomeToolbox | Python |

### 데이터

| 데이터 | URL | 내용 |
|--------|-----|------|
| WormAtlas | https://www.wormatlas.org | 해부학, 뉴런 맵 |
| WormWiring | https://www.wormwiring.org | Connectome 인접 행렬 |
| Cook 2019 | Nature 논문 supplementary | 전체 connectome (양 성) |
| NeuroPAL | Cell 논문 | 뉴런 컬러 아틀라스 |

---

## 7. 최신 성과 타임라인 (2023-2026)

| 시기 | 성과 | 팀/논문 |
|------|------|---------|
| 2023.05 | Connectome-Based Digital Twin, 오프라인 학습으로 화학주성 | MDPI Mathematics |
| 2023.11 | 뉴로펩타이드 "무선 connectome" 완전 매핑 | Ripoll-Sanchez, Neuron |
| 2024.03 | ElegansBot: 오메가 턴 포함 운동 재현 | eLife |
| 2024.12 | BAAIWorm: Nature Comp Sci, 136 뉴런 brain-body-env 통합 | BAAI |
| 2024.12 | Biophysically detailed neurons + muscle (별도 Nature 논문) | Nature Comp Sci |
| 2024.12 | 포괄적 connectome 분석, 미연구 뉴런 기능 발견 | PLOS Biology |
| 2025 | NeuroSimWorm: 최초 다중감각 closed-loop | Neurocomputing |
| 2025 | C. elegans wired+wireless connectome 통합 리뷰 | J. Biosciences |

---

## 8. Genesis Brain (27,910 뉴런)과의 비교

### 규모 비교

| 항목 | C. elegans (실제) | C. elegans (시뮬레이션 best) | Genesis Brain | 초파리 (시뮬레이션) |
|------|-------------------|------------------------------|---------------|---------------------|
| **뉴런** | 302 | 136 (BAAIWorm) | 27,910 | 127,400 |
| **시냅스** | ~6,900 | ~2,000 (추정) | 47 population + ~30K KC | 50M |
| **뉴런 모델** | - | HH 다중구획 | LIF/SensoryLIF | LIF |
| **학습** | STDP+도파민 | 없음 | R-STDP, Hebbian, Anti-Hebbian | (Eon: 미공개) |
| **체화** | 실제 몸 | 3D 물리 | 2D Forager Gym | 3D MuJoCo |

### 구조적 비교

| 특성 | C. elegans 시뮬레이션 | Genesis Brain |
|------|----------------------|---------------|
| **Connectome 기반** | O (실제 배선도 사용) | X (기능적 설계) |
| **뉴런 개별 ID** | O (각 뉴런에 이름) | X (population 단위) |
| **시냅스 가소성** | X (static) | O (47 학습 시냅스) |
| **도파민 시스템** | X | O (VTA, D1/D2 MSN) |
| **행동 복잡도** | 화학주성, 운동 | 먹이 탐색, 포식자 회피, 장애물, 학습 |
| **감각 모달리티** | 화학 (+ 최근 기계/온도) | 시각(food/wall/predator), 통증, 체내상태 |
| **의사결정** | 단순 (approach/avoid) | 다단계 (BG, GW, PFC) |
| **기억** | 없음 | 해마 공간기억, SWR replay |

### 핵심 차이점

1. **Bottom-up vs Top-down 설계**
   - C. elegans: connectome에서 시작 → 행동이 창발되길 기대 (bottom-up)
   - Genesis: 행동 목표에서 시작 → 생물학적 회로로 구현 (top-down, 생물학 제약 내)

2. **Static vs Learning**
   - C. elegans: 모든 시냅스 고정 → 행동 고정
   - Genesis: 47 학습 시냅스 + 30K KC 연결 → 경험에서 행동 변화

3. **Connectome Fidelity vs Functional Organization**
   - C. elegans: 실제 연결 재현 (높은 생물학적 정확도)
   - Genesis: 기능적 영역 재현 (시상하부, 편도체, 해마, BG 등)

---

## 9. 차용 가능한 기술/패턴

### 9.1 Gap Junction (전기 시냅스) 모델링

**C. elegans 교훈:** Gap junction이 화학 시냅스보다 네트워크 동기화에 더 큰 영향.

**Genesis 적용:**
- 현재 Genesis는 화학 시냅스만 사용
- Gap junction은 뉴런 간 빠른 동기화 → WTA 회로, 리듬 생성에 유용
- PyGeNN에서 `ContinuousDendDend` 연결로 구현 가능

### 9.2 Closed-loop Brain-Body-Environment 패턴

**C. elegans 교훈 (BAAIWorm):** 뇌-몸-환경의 실시간 closed-loop가 행동 창발의 핵심.

**Genesis 현황:** 이미 ForagerGym으로 구현 중. BAAIWorm의 3D 물리보다는 단순하지만 원리는 동일.

### 9.3 Neuromodulation Layer

**C. elegans 교훈:** 뉴로펩타이드 "무선 connectome"이 행동 상태를 전환. 시냅스만으로 불충분.

**Genesis 적용:**
- 현재 도파민만 구현 (VTA→D1/D2)
- 세로토닌 (행동 상태: 탐색 ↔ 착취), 노르에피네프린 (각성) 추가 고려
- 뉴로모듈레이션 → 시냅스 전체의 gain 조절 (현재 도파민이 이미 이렇게 동작)

### 9.4 Ablation Study 패턴

**C. elegans 교훈:** 특정 뉴런/시냅스를 끊고 행동 변화를 관찰 → 인과관계 파악.

**Genesis 적용:**
- population을 일시 비활성화하고 생존율 변화 측정
- `--ablate <population>` CLI 옵션 가능
- 예: "편도체 ablation → 포식자 회피 소멸 확인" → 회로 검증

### 9.5 다중감각 통합 패턴 (NeuroSimWorm)

**C. elegans 교훈:** 4종 감각 회로(운동/화학/온도/기계)를 통합 프레임워크에서 관리.

**Genesis 현황:** 이미 다중감각 (시각, 통증, 체내상태)이 있으나, 감각별로 독립적. NeuroSimWorm처럼 모듈화하면 새 감각 추가가 용이해질 것.

### 9.6 Connectome 데이터에서 회로 패턴 추출

**C. elegans 교훈:** 302 뉴런에서 발견된 회로 motif (feedforward, feedback, mutual inhibition)가 더 큰 뇌에서도 반복.

**Genesis 적용:**
- C. elegans에서 검증된 회로 motif를 Genesis population 설계에 참고
- 예: command interneuron (AVA/AVB) 패턴 → Go/NoGo 의사결정과 유사 → BG D1/D2와 대응

### 9.7 Arduino/FPGA 경량 구현 패턴 (미래 하드웨어)

**C. elegans 교훈:** 302 LIF 뉴런이 Arduino UNO(8KB)에서 실행 가능.

**Genesis 적용 (미래):**
- ESP32(520KB SRAM)이면 수천 뉴런 LIF 가능
- 핵심 반사 회로(Pain Push-Pull, 먹이 접근)만 추출하면 임베디드 가능
- Nematoduino의 connectome 압축 기법 참고

---

## 10. 핵심 교훈 요약

### 302 뉴런으로 가능한 것과 불가능한 것

```
302 뉴런 (static) → 기본 반사, 화학주성, 운동
302 뉴런 (가소성) → 연합학습, 습관화 (실제 C. elegans가 하는 것)
302 뉴런 (가소성+뉴로모듈레이션) → 상태 전환, 맥락 의존 행동

→ 뉴런 수보다 "시냅스 가소성 + 뉴로모듈레이션"이 행동 복잡도를 결정
```

### Genesis Brain에 대한 시사점

1. **가소성이 왕이다**: C. elegans 프로젝트들이 행동 재현에 고전하는 이유는 가소성 미구현. Genesis의 47 학습 시냅스 + 30K KC는 이미 큰 강점.

2. **Connectome Fidelity < Functional Learning**: 실제 배선도를 완벽히 복제해도 학습 없으면 static 행동만 가능. Genesis의 top-down 기능적 설계 + 학습이 더 유연한 행동 생성.

3. **뉴로모듈레이션 확장이 다음 도약**: C. elegans 연구가 보여준 "wireless connectome"의 중요성. Genesis도 도파민 너머 세로토닌/노르에피네프린 추가 시 행동 상태 전환 가능.

4. **Ablation 테스트 도입**: C. elegans 연구의 표준 방법론. Genesis에서도 각 brain region의 필요성을 ablation으로 검증하면 불필요한 회로 정리 가능.

5. **규모의 역설**: C. elegans 302 뉴런 full connectome보다 Genesis 27,910 뉴런 functional design이 더 복잡한 행동을 산출. 뉴런 수가 아니라 **학습 가능한 연결의 구조**가 핵심.

---

## 참고 문헌

- White, J.G. et al. (1986). "The Structure of the Nervous System of the Nematode C. elegans." *Phil. Trans. R. Soc.*
- Cook, S.J. et al. (2019). "Whole-animal connectomes of both C. elegans sexes." *Nature*
- Ripoll-Sanchez, L. et al. (2023). "The neuropeptidergic connectome of C. elegans." *Neuron*
- BAAIWorm (2024). "An integrative data-driven model simulating C. elegans brain, body and environment interactions." *Nature Computational Science*
- NeuroSimWorm (2025). "A multisensory framework for modeling and simulating neural circuits of C. elegans." *Neurocomputing*
- Ren, Q. & Kolwankar, K.M. (2010). "STDP-driven networks and the C. elegans neuronal network." *Physica A*
