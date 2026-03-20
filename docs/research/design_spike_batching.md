# 설계: Spike Recording 배치화 (GPU 최적화)

> 날짜: 2026-03-20
> 근거: 초파리 뇌 시뮬레이션 기술 분석에서 도출

---

## 1. 현재 문제

### process() 루프 구조 (현재)

```python
for _ in range(10):          # 10 timestep
    self.model.step_time()   # GPU에서 1ms 시뮬레이션

    # 매 스텝마다 개별 pull (131개 population)
    self.motor_left.vars["RefracTime"].pull_from_device()     # GPU→CPU
    self.motor_right.vars["RefracTime"].pull_from_device()    # GPU→CPU
    self.hunger_drive.vars["RefracTime"].pull_from_device()   # GPU→CPU
    # ... 128개 더 ...

    # 스파이크 카운팅
    motor_left_spikes += np.sum(self.motor_left.vars["RefracTime"].view > threshold)
    # ...
```

**병목**: 131 pulls × 10 steps = **1,310회 GPU↔CPU 왕복/process()**
- 각 왕복: PCIe 레이턴시 ~5μs + 데이터 전송
- 총: ~6.5ms/process() (순수 통신 오버헤드)
- GPU는 pull 대기 중 idle → 다시 연산 → 다시 idle → 사용률 100%이지만 효율은 낮음

---

## 2. 해결: GeNN Spike Recording

### PyGeNN 5.4 spike_recording API

```python
# 빌드 전에 recording 활성화
pop = model.add_neuron_population(...)
pop.spike_recording_enabled = True

# 시뮬레이션 후 한번에 데이터 가져오기
model.pull_recording_buffers_from_device()
spike_data = pop.spike_recording_data  # (times, neuron_ids) 튜플
```

### 변경된 process() 루프

```python
# === 3. 시뮬레이션 (10 timestep) ===
for _ in range(10):
    self.model.step_time()   # 10회 연속 GPU 연산 (중간 pull 없음)

# 한번에 recording buffer pull (1회 호출)
self.model.pull_recording_buffers_from_device()

# 스파이크 카운팅 (CPU에서)
motor_left_spikes = len(self.motor_left.spike_recording_data[0])
motor_right_spikes = len(self.motor_right.spike_recording_data[0])
hunger_spikes = len(self.hunger_drive.spike_recording_data[0])
# ...
```

**개선**: 1,310회 → **1회** GPU↔CPU 왕복

---

## 3. 구현 계획

### Phase 1: spike_recording 활성화

모든 스파이크 카운팅 대상 population에 `spike_recording_enabled = True` 설정.

**대상 population 목록** (131개 pull에 해당):

| 그룹 | Population | 수 |
|------|-----------|-----|
| Core | motor_left/right, hunger, satiety, low/high_energy | 6 |
| Amygdala | lateral_amygdala, central_amygdala, fear_response | 3 |
| Hippocampus | place_cells, food_memory_left/right | 3 |
| BG | d1_l/r, d2_l/r, direct_l/r, indirect_l/r, dopamine | 9 |
| PFC | working_memory, goal_food/safety, inhibitory | 4 |
| Cerebellum | granule, purkinje, deep_nuclei, error | 4 |
| Thalamus | food_relay, danger_relay, trn, arousal | 4 |
| V1 | v1_food_l/r, v1_danger_l/r | 4 |
| V2/V4 | v2_edge_food/danger, v4_food/danger/novel | 5 |
| IT | it_food/danger/neutral/assoc/buffer | 5 |
| L6 PE | pe_food_l/r, pe_danger_l/r | 4 |
| L10 NAc | nac_value | 1 |
| L12 GW | gw_food_l/r, gw_safety | 3 |
| L14 Agency | agency_pe | 1 |
| Phase 20 Self | self_body, self_narrative, self_agency | 3 |
| 기타 고차 피질 | (필요시 추가) | ~20+ |

### Phase 2: process() 루프 리팩토링

```python
def process(self, observation, debug=False):
    # === 1. 감각 입력 설정 (현재와 동일) ===
    # ... push_to_device() ...

    # === 2. 시뮬레이션 (10 timestep, 중간 pull 없음) ===
    for _ in range(10):
        self.model.step_time()

    # === 3. 한번에 recording buffer pull ===
    self.model.pull_recording_buffers_from_device()

    # === 4. 스파이크 카운팅 (CPU, recording_data에서) ===
    motor_left_spikes = len(self.motor_left.spike_recording_data[0])
    motor_right_spikes = len(self.motor_right.spike_recording_data[0])
    # ...

    # === 5. 발화율 계산 + 행동 결정 (현재와 동일) ===
```

### Phase 3: 검증

1. **기능 검증**: 20ep 돌려서 이전 결과와 survival/reward freq 비교
2. **성능 측정**: process() 소요 시간 비교 (before/after)
3. **GPU 사용률**: gpu_check.ps1로 3D% 비교

---

## 4. 주의사항

### spike_recording_data 형식

```python
# spike_recording_data = (times_array, neuron_ids_array)
# times_array: 스파이크 발생 시간 (model time)
# neuron_ids_array: 스파이크 발생 뉴런 ID
# len(times_array) = 총 스파이크 수 (10 timestep 동안)
```

### 현재 코드와의 호환

- 현재: `np.sum(pop.vars["RefracTime"].view > threshold)` = 매 스텝 "지금 refractory인 뉴런 수"
- 변경: `len(pop.spike_recording_data[0])` = 10스텝 동안 총 스파이크 수
- **차이**: 한 뉴런이 10스텝 중 2번 스파이크 → 현재는 2회 카운트 (매 스텝 체크), recording도 2회 카운트 → **동일 의미**

### Pain Push-Pull 영향

- Push-Pull은 시냅스 가중치 기반이므로 spike recording과 무관
- Motor 출력만 마지막에 reading → 문제 없음

### IT_Food 활성도 캐싱 (L9)

```python
# 현재: 매 스텝마다 IT_Food 활성 여부 체크
self._it_food_active = 1.0 if (it_food_category_spikes / n_it_f) > 0.05 else 0.0
```

10스텝 합산으로 변경 시 임계값 조정 필요: `0.05 × 10 = 0.5` (10스텝 중 1스텝이라도 활성이면)

---

## 5. 예상 효과

| 지표 | Before | After (예상) |
|------|--------|-------------|
| GPU↔CPU 왕복/process() | 1,310회 | ~1회 |
| process() 시간 | ~10ms | ~3-4ms |
| GPU 3D 사용률 | 90-100% | 50-70% (추정) |
| 학습 결과 | baseline | 동일 (로직 변화 없음) |
| 50ep 배치 소요 시간 | ~90분 | ~40-60분 (추정) |

---

## 6. 추가 최적화 (향후)

### Synaptic Delay 추가

```python
# 학습 시냅스에 delay 추가
model.add_synapse_population(
    ...,
    delay_steps=2  # 2ms delay (dt=1.0ms 기준)
)
# Pain Push-Pull은 delay_steps=0 유지
```

### Sparse Expansion Layer (L16)

```python
# Mushroom Body KC 패턴
# food_eye(400) → KC(2000-4000) → D1/D2(200)
# KC: random sparse connectivity 10%, strong lateral inhibition → 5% activation
kc = model.add_neuron_population("KC_expansion", 4000, "LIF", ...)
model.add_synapse_population("food_eye_to_kc", "SPARSE_GLOBALG", ...,
    connectivity_init=init_sparse(connect_prob=0.10))
model.add_synapse_population("kc_wta", "SPARSE_GLOBALG", ...,  # lateral inhibition
    connectivity_init=init_sparse(connect_prob=0.05),
    weight=-15.0)
model.add_synapse_population("kc_to_d1", "SPARSE", ...,  # R-STDP
    ...)
```
