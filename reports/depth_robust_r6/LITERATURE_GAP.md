# R6 선행연구와 차이 (LITERATURE_GAP)

작성 2026-11-17. 아래는 공개 원문·공식 README에서 확인한 범위다. 논문의 성능을 이 자료에서 재현했다는 뜻이 아니다.

| 선행 | 확인한 범위 | R6가 추가로 묻는 것 | R6가 주장하지 않는 것 |
|---|---|---|---|
| Ahlmann-Eltze 등 2025 (Nat Methods 22:1657) | 유전적 개입 예측에서 평균·선형 기준선이 강함 | 출발점을 약하게 두지 않는다: 무변화, 세포주 평균, 2-블록 Ridge를 같은 입력으로 둔다 | 딥러닝이 원리적으로 불가능하다는 주장 |
| Systema 2025/2026 (Nat Biotechnol) | 공통 이동과 개입 특이 신호를 나눠 평가 | ΔRMSE와 각자 평균을 뺀 ΔPearson, retrieval을 함께 보고 | 평균 차감으로 기술 오차와 생물학을 완전히 분리했다는 주장 |
| scPRINT 2025 (Nat Commun 16:3607) | downsampling·denoising을 포함한 scRNA 사전학습 | 발현 재구성을 잘하는 것과 약물 반응 예측을 보존하는 것이 같은지 (Denoise-control) | scPRINT를 재학습하거나 이겼다는 주장 (실행하지 않음) |
| scTAT 2026 preprint | 토큰 예산·표현 규약 개선을 분류에서 보고 | 관련 연구로만 둔다 | 직접 비교 (호환 구현을 확인하지 않음) |
| data2vec 2022 (ICML) | 풍부한 view의 teacher 표현을 손상 view의 student가 예측 | count 박화를 손상으로, 실측 Δ를 항상 감독 신호로 두는 변형 | teacher–student 자체의 새로움 |
| count splitting, Neufeld 등 2024 | Poisson 가정 아래 count 분할의 독립성 | 박화본을 독립 반복으로 세지 않는다 | 박화가 실제 얕은 sequencing과 같다는 주장 |
| State (bioRxiv 2025.06.26.661135v2, 공식 README) | 세포 집합 기반 반응 모델. 로더에 binomial count `downsample`과 `downsample_cells` 옵션이 **이미 있다** | 단순 박화 증강은 새 기법이 아니다. R6의 B00이 그에 준하는 명시적 증강 대조다 | State와의 수치 비교 (부적격 판정, `DECISIONS.md` D4) |
| chemCPA (공개 코드 43e830eb) | 화학 구조 조건의 반응 autoencoder | 증강 없이 학습된 기존 모델이 저관측 입력에서 얼마나 견디는지의 참조 | 같은 학습 조건의 비교 |

## 신규성 위험

denoising, teacher–student, count downsampling 증강은 모두 알려진 기술이다. R6가 기여할 수 있는 지점은
(i) 같은 데이터·같은 학습량에서 **입력의 관측 정보**와 **반응 중심 학습 목적**을 2×2로 분리한 것,
(ii) 발현 재구성 중심 학습과 반응 보존 학습을 같은 조건에서 나란히 둔 것,
(iii) 정답을 전체 관측으로 고정하고 library size를 박화 합으로 일관되게 계산한 평가 계약이다.
실제 효과가 없으면 새 모델 이름으로 기여를 주장하지 않는다. 문헌을 조사했다고 같은 조합의 선행연구가 없다고 확정되지는 않는다.
