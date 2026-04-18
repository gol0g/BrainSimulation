# M4 Context-Dependent Food Rules — GPT Consultation v9

> Date: 2026-04-13
> Source: ChatGPT (3m 5s extended thinking)

## Core Diagnosis

**"Context-blind 주회로를 살려둔 채 context 회로를 옆에 더한 구조" — 이것이 8번 실패의 공통 원인.**

8개 시도는 전부 "기존 정책에 context correction을 더하는" additive rescue.
실제로 필요한 것은 **irrelevant mapping 자체를 꺼버리는 suppression/gating**.

## 근본적 착오

- "같은 food stimulus가 context에 따라 반대 값" = late motor bias 문제가 아니라 **early routing / conjunctive representation 문제**
- 기존 food→D1→Direct→Motor = 강한 context-blind 기본 정책
- 여기에 CtxVal, D1_ctx, parallel pathway를 더해도 "항상 food 접근" 위에 약간의 보정만 얹힘
- **divergence는 생겨도 selectivity는 0.5** = 현재 회로가 풀 수 있는 해에 정확히 수렴한 것

## 5가지 수정안

### 1. Context-blind food actor를 "차단 대상"으로 바꾸기
- 기존 KC→D1/D2 food path는 context task에서 **꺼져야** 함
- CtxA active → baseline food actor OFF, bank A ON, bank B OFF
- CtxB active → baseline food actor OFF, bank B ON, bank A OFF
- context uncertain/boundary → baseline actor weakly ON or plasticity OFF

### 2. Additive parallel이 아니라 context-specific actor bank 분리
```
KC_A → D1_A / D2_A → Motor
KC_B → D1_B / D2_B → Motor
CtxA는 D1_A/D2_A bank만 disinhibit, D1_B/D2_B는 강하게 inhibit
CtxB는 반대
```
- context = "motor bias"가 아니라 **which policy is allowed to speak** 결정

### 3. Plasticity도 active bank에만
- DA는 global OK
- **inactive bank는 eligibility trace accumulation 금지**
- inactive bank는 postsyn depol clamp 또는 weight update mask
- boundary crossing 직후 몇 step은 plasticity freeze
- wrong channel이 eligible하면 trace contamination 발생

### 4. KC에서 conjunction 구조적 보장
- 가장 쉬운 방법: **KC_A, KC_B 분리** — 각자 food + 해당 context만 받기
- place→KC 실패 = conjunction coverage 부족 (KC가 food만 보거나 context만 봄)

### 5. Chicken-and-egg는 exploration scaffold로 해결
- 초기 5-20 episode는 food visible 시 소량의 forced approach
- 또는 context bank active일 때 약한 shaping reward
- 또는 boundary에서 teacher turn
- cheating이 아니라 새 channel의 첫 reward를 위한 bootstrap

## 최소 수정안 (GPT 추천)

```
CtxA/CtxB → WTA context gate
gate A ON → KC_A, D1_A, D2_A active / KC_B, D1_B, D2_B inhibited
gate B ON → 반대
baseline KC→D1/D2는 context task 동안 gate-off
DA는 global 유지
eligibility는 active gate에서만 accumulate
zone transition 후 5-10 step plasticity freeze
food visible 시 초반 bootstrap exploration 10-20%
```

## Active Dendrite 필요성

**아직 불필요.** 
- 실패 패턴 = "뉴런이 단순해서"가 아니라 회로 논리가 additive correction에 머물러 있기 때문
- context-specific hard gating + bank-specific eligibility + conjunction input structure를 먼저 넣기
- 이것으로도 0.5 근처면 그때 dendritic gating 검토

## References (GPT cited)
- Nature Neuroscience 2025: context-dependent RNN, suppression mechanism
- PFC-MD model 2024: thalamic gating for context-irrelevant neuron suppression
- Miconi: biologically plausible reward learning, recurrent exploration
- Striatal modeling: selective eligibility as key to credit assignment
