# 즉시 수정이 필요한 이슈들

## 🔴 High Priority (즉시 수정 권장)

### 1. 수치 안정성 문제

**파일**: `backend/genesis/preference_distributions.py`

**문제**:
```python
# 극단값에서 Beta 분포 평가 시 -inf 발생 가능
log_prob = (alpha - 1) * np.log(obs) + (beta - 1) * np.log(1 - obs)
```

**해결책**:
```python
def safe_log(x, eps=1e-10):
    """안전한 로그 계산"""
    return np.log(np.clip(x, eps, None))

def safe_divide(a, b, eps=1e-10):
    """안전한 나눗셈"""
    return a / (b + eps)

# 사용:
log_prob = (alpha - 1) * safe_log(obs) + (beta - 1) * safe_log(1 - obs)
```

---

### 2. Regret Baseline 버그

**파일**: `backend/genesis/regret.py:88`

**문제**:
```python
if len(self.recent_regret) >= 10:
    self.regret_baseline = np.mean(self.recent_regret[-20:])
```

`recent_regret` 길이가 10일 때 `[-20:]`은 10개만 반환 (의도: 20개)

**해결책**:
```python
if len(self.recent_regret) >= 20:
    self.regret_baseline = np.mean(self.recent_regret[-20:])
elif len(self.recent_regret) >= 5:
    self.regret_baseline = np.mean(self.recent_regret)
else:
    self.regret_baseline = 0.1  # 기본값
```

---

### 3. Context-weighted Delta 이중 클리핑

**파일**: `backend/genesis/action_selection.py:2153-2170`

**문제**:
```python
delta_ctx = np.clip(delta_ctx, -self.delta_ctx_clamp, self.delta_ctx_clamp)
delta_combined = (1 - alpha_eff) * delta_base + alpha_eff * delta_ctx
# 블렌딩 후 다시 커질 수 있음
```

**해결책**:
```python
delta_ctx = np.clip(delta_ctx, -self.delta_ctx_clamp, self.delta_ctx_clamp)
delta_combined = (1 - alpha_eff) * delta_base + alpha_eff * delta_ctx
# 최종 안전장치
delta_combined = np.clip(delta_combined, -0.15, 0.15)
```

---

## 🟡 Medium Priority (단기 개선)

### 4. Action History 무제한 증가

**파일**: `backend/genesis/action_selection.py:157`

**문제**:
```python
self._action_history = []  # 제한 없음
```

**해결책**:
```python
self._action_history = []
self._action_history_max = 1000  # 최대 1000개

# append 시:
self._action_history.append(action)
if len(self._action_history) > self._action_history_max:
    self._action_history.pop(0)
```

---

### 5. Adaptive Learning Rate

**파일**: `backend/genesis/action_selection.py:1600-1700`

**현재**:
```python
self.transition_lr = 0.1  # 고정
```

**개선**:
```python
# Uncertainty 기반 적응적 학습률
def get_adaptive_lr(self, action: int) -> float:
    base_lr = 0.1
    uncertainty = self.transition_model['delta_std'][action].mean()
    # 불확실할수록 빠르게 학습
    adaptive_lr = base_lr * (1.0 + uncertainty)
    return min(adaptive_lr, 0.3)  # 최대 0.3
```

---

### 6. Temperature 적응

**파일**: `backend/genesis/action_selection.py:149`

**현재**:
```python
self.temperature = 0.3  # 고정
```

**개선**:
```python
# Uncertainty 기반 온도 조절
def get_adaptive_temperature(self) -> float:
    if self.uncertainty_enabled and self._last_uncertainty_state:
        u = self._last_uncertainty_state.global_uncertainty
        # u=0 → temp=0.1 (확신), u=1 → temp=0.5 (불확실)
        return 0.1 + 0.4 * u
    return 0.3
```

---

## 🟢 Low Priority (장기 개선)

### 7. 단위 테스트 추가

**신규 파일**: `backend/tests/test_math.py`

```python
import pytest
import numpy as np
from genesis.preference_distributions import PreferenceDistributions

def test_beta_kl_non_negative():
    """KL divergence는 항상 0 이상"""
    prefs = PreferenceDistributions()
    for _ in range(100):
        obs = np.random.uniform(0.01, 0.99, 8)
        kl = prefs.kl_divergence('energy', obs[6])
        assert kl >= 0, f"KL should be non-negative, got {kl}"

def test_extreme_observations():
    """극단값에서도 안정적으로 작동"""
    prefs = PreferenceDistributions()
    # 0에 가까움
    kl1 = prefs.kl_divergence('energy', 0.001)
    assert np.isfinite(kl1), "Should handle obs near 0"
    
    # 1에 가까움
    kl2 = prefs.kl_divergence('energy', 0.999)
    assert np.isfinite(kl2), "Should handle obs near 1"
```

---

### 8. Config 분리

**신규 파일**: `backend/genesis/config.py`

```python
from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class GenesisConfig:
    """모든 하이퍼파라미터 통합 관리"""
    
    # Action Selection
    temperature: float = 0.3
    complexity_weight: float = 0.5
    transition_lr: float = 0.1
    
    # THINK
    think_entropy_threshold: float = 1.0
    think_cooldown: int = 5
    
    # Memory
    max_episodes: int = 1000
    store_threshold: float = 0.5
    similarity_threshold: float = 0.95
    
    # Uncertainty
    belief_weight: float = 0.25
    action_weight: float = 0.30
    model_weight: float = 0.20
    surprise_weight: float = 0.25
    
    # Drift Suppression
    drift_error_threshold: float = 2.5
    drift_recovery_rate: float = 0.05
    
    @classmethod
    def from_yaml(cls, path: str) -> 'GenesisConfig':
        """YAML 파일에서 로드"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """YAML 파일로 저장"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)
```

---

## 적용 순서 권장

1. ✅ **수치 안정성** (30분) - 즉시
2. ✅ **Regret baseline 버그** (10분) - 즉시  
3. ✅ **Delta clipping** (15분) - 즉시
4. 📝 **Action history 제한** (10분) - 오늘 중
5. 📝 **Adaptive LR** (1시간) - 이번 주
6. 📝 **Temperature 적응** (30분) - 이번 주
7. 🔄 **단위 테스트** (2-3시간) - 다음 주
8. 🔄 **Config 분리** (1시간) - 다음 주

---

## 테스트 방법

각 수정 후:

```bash
# 1. 기본 동작 확인
cd backend
python main_genesis.py

# 2. 재현성 테스트
curl -X POST http://localhost:8002/reproducibility/test

# 3. Drift 시나리오 테스트
curl -X POST http://localhost:8002/drift/enable?drift_type=rotate
# 200 스텝 실행
curl -X GET http://localhost:8002/scenario/drift_report

# 4. 수동 확인
# - NaN/Inf 에러 없는지
# - Drift 적응 정상 작동하는지
# - Regret 값이 합리적인지
```

---

**우선순위 요약**:
- 🔴 **즉시**: 1-3번 (수치 안정성, 버그 수정)
- 🟡 **단기**: 4-6번 (메모리, 적응성)
- 🟢 **장기**: 7-8번 (인프라, 구조)
