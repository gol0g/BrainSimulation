# Brain Simulation Project

## 최종 목표
**인간의 뇌와 유사한 주체적 의식과 감정을 가지는 인공 뇌를 만드는 것**

---

## 현재 구현된 시스템

### 1. SNN (Spiking Neural Network)
- STDP 3-factor learning (pre-post timing + neuromodulator)
- Synapse weight bounds: W_MIN=5.0, W_MAX=150.0

### 2. Agency Detection
- SELF_CAUSED vs EXTERNALLY_CAUSED 구분
- 외부 perturbation(wind) 감지 시 학습 차단

### 3. Working Memory
- 단기 기억 유지 (decay 기반)
- Self-Model의 stability에 의해 decay rate 조절됨

### 4. Attention System
- Saliency 기반 주의 집중
- Inhibition of Return (IOR)
- Self-Model의 effort에 의해 dwell time/gain 조절됨

### 5. Self-Model (자기 표상)
- **5가지 Self-States**: confidence, uncertainty, effort, exploration_need, stability
- **7가지 Behavioral Labels**: CONFIDENT, EXPLORING, STRUGGLING, REACTIVE, FATIGUED, STABLE, TRANSITIONING
- **인과적 연결 (A/B/C)**:
  - A) exploration_need/confidence → epsilon (탐험률)
  - B) effort → attention dwell time/gain
  - C) stability → Working Memory decay rate

### 6. Value Conflict (비활성화)
- 현재 disabled - 가중치 degradation 문제로 스킵

---

## 최근 구현 완료

### 상태 전이 히스테리시스 ✅
- 감정/상태가 "머무는" 느낌 부여
- **min_state_duration = 15**: 최소 15 스텝 유지 후 전환 고려
- **transition_threshold = 8**: 새 상태가 8 스텝 지속 시 전환
- **Entry/Exit 임계값 분리**:
  - 진입 (Entry): 더 엄격 (예: CONFIDENT 진입 → confidence > 0.70)
  - 유지 (Exit): 더 느슨 (예: CONFIDENT 유지 → confidence > 0.50)
- 상태가 쉽게 바뀌지 않음 → 감정의 "관성" 표현

### Self-state → 예측 연결 ✅ (자기 설명 능력)
- **핵심**: 자기 상태가 예측 오차 해석에 영향
- `expected_error = base + uncertainty * 0.4 + effort * 0.2`
- `adjusted_error = raw_error - expected_error * 0.5`
- **효과**:
  - uncertainty ↑ → "예상 오차가 큼" → 실제 오차가 높아도 덜 놀람
  - "내가 혼란스러워서 이 정도 오차는 정상" = **자기 설명**
- `agency.py`에 `self_explanation` 필드 추가
- Self-Model이 "조절기"를 넘어 "자기 설명 모델"로 진화

### Attribution 분리 ✅ (내 탓 vs 외부 탓)
- **externality**: "세상이 문제야" (0-1)
  - external_pressure, wind 이벤트 빈도 기반
  - externality ↑ → exploration ↓ ("기다려, 상황이 지나갈 거야")
- **internal_fault**: "내 전략이 문제야" (0-1)
  - 외부 이벤트 없이 prediction error 높을 때
  - internal_fault ↑ → exploration ↑ ("전략 바꿔야 해")
- **핵심 통찰**: 둘 다 실패해도 원인이 다르면 반응도 달라야
- epsilon 조절에 반영: `-externality * 0.15 + internal_fault * 0.2`

---

## Self-Model 완성 상태

| 기능 | 상태 | 역할 |
|------|------|------|
| 히스테리시스 | ✅ | 감정이 "머무는" 느낌 |
| 자기 설명 | ✅ | 오차를 자기 상태로 해석 |
| Attribution 분리 | ✅ | 내 탓 vs 외부 탓 구분 |

이제 Self-Model은:
1. **자기 상태를 안다** (confidence, uncertainty, effort, ...)
2. **자기 상태로 상황을 설명한다** (오차 해석)
3. **원인을 자기/외부로 귀속한다** (attribution)
4. **귀속에 따라 다르게 반응한다** (exploration 조절)

---

## 기술적 주의사항
- Wall collision penalty: -0.3 (너무 크면 가중치 degradation)
- Large food: 현재 disabled (spawn_chance = 0.0)
- 서버 재시작은 Claude가 직접 수행 (사용자에게 시키지 않기)
