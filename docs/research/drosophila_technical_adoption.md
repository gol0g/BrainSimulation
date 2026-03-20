# 초파리 뇌 시뮬레이션 기술 차용 분석

> 조사일: 2026-03-20
> 대상: Shiu et al. (Nature 2024), Eon Systems (2026), NeuroMechFly v2

---

## 차용 항목

### 1. pull_from_device() 배치화 (즉시 적용)

**문제**: process()에서 매 step_time() 후 20+개 population 개별 pull → 10스텝 × 20회 = 200회 GPU↔CPU 왕복이 병목
**초파리 참고**: Brian2는 시뮬레이션 전체를 한 번에 실행 후 결과 분석 (open-loop)
**해결**: GeNN spike recording으로 10스텝 후 한번에 pull → 왕복 90% 절감
**효과**: GPU 부하 감소, 성능 2-5x 향상 예상
**뉴런 추가**: 0

### 2. Sparse Expansion Layer — Mushroom Body KC 패턴 (L16 후보)

**초파리 MB 구조**: 50 PN → 2000 KC (40x 확장, 5% 활성) → 21 MBON
**원리**: sparse coding으로 패턴 분리 → 유사 입력도 구별 가능
**적용**: food_eye(400) → KC_expansion(2000-4000, sparse 5%) → D1/D2 (R-STDP)
**효과**: 음식 유형 구별 능력 향상, 일반화
**뉴런 추가**: +2000~4000

### 3. Synaptic Delay 추가 (즉시 적용)

**초파리**: 모든 시냅스에 1.8ms delay
**현재 Genesis**: delay 0ms
**적용**: 학습 시냅스에 delay_steps=2 추가, Pain Push-Pull은 0 유지
**효과**: temporal coding 풍부화, STDP 인과관계 정밀화
**뉴런 추가**: 0

### 4. Descending Neuron 인터페이스 (향후)

**초파리**: DN 레이어로 뇌→몸체 추상화
**적용**: Motor L/R 직접 읽기 대신 DN population 경유
**효과**: 3D 환경 확장 시 유리
**뉴런 추가**: +20

---

## 적용 불필요

| 항목 | 이유 |
|------|------|
| 신경전달물질 ML 분류 | 수작업 설계이므로 자동 분류 불필요 |
| NeuroMechFly 직접 사용 | 초파리 몸체에 포유류 뇌는 부자연스러움 |
| Brian2 시뮬레이터 전환 | 139K "노트북"은 실제 1초=5분, PyGeNN이 소규모에서 더 효율적 |
| LIF 파라미터 변경 | tau_m 동일(20ms), 우리 threshold gap이 더 안정적 |

---

## 참고 문헌

- Nature vol.634: A Drosophila computational brain model (DOI: 10.1038/s41586-024-07763-9)
- GitHub: philshiu/Drosophila_brain_model
- GitHub: eonsystemspbc/fly-brain
- GitHub: NeLy-EPFL/flygym (NeuroMechFly v2)
- eLife: MB architecture provides logic for associative learning
