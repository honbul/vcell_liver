# R10 CLAIMS_AND_LIMITATIONS

(결과 수치는 [`RESULTS.md`](RESULTS.md); 이 문서는 주장의 범위만 정한다.)

## 주장할 수 있는 것 (이 자료·이 설정 안에서)

- Sci-Plex3 24 h, A549·K562·MCF7, 우리 hash QC와 train-only PCA 20 / GMM K=4 표현에서 측정한 **관측 집단의 상태 점유율과 상태 내부 분포**에 대한 기술 통계와 예측 오차.
- 같은 약물의 두 endpoint를 문맥으로 주는 **부분 측정 약물의 중간 용량 보간** 성능과, 그것이 replicate-well 관측 한계에 비해 어디에 있는지.
- 사전 고정한 2×2(D00/D10/D01/D11)에서 각 head가 이 설정의 development/test 약물에 대해 보인 차이(약물 단위 bootstrap 구간 포함).
- 합성 양성 대조에서 평가기가 구성비 변화와 상태 내부 변화를 구분한다는 것(평가 도구의 성질; 생물학적 검증 아님).

## 주장하지 않는 것

- 완전 미관측 약물(zero-shot), 새 세포주, 성인 HCC, 다른 시간점·다른 screen으로의 일반화.
- 개별 세포의 상태 전이, responder 비율, 내성 세포 생존, 세포 사멸 기여 등 인과·운명 해석. 구성비는 회수·포획·증식·사멸·상태 변화가 섞인 관측 점유율이다.
- "전체 섭동 분포 예측이 좋아졌다"는 표현은 전체 분포 지표(ED/MMD)도 함께 좋아진 경우에만 쓴다.
- 두 replicate가 독립 배양 batch라는 가정, 또는 cross-replicate 결과를 "batch 없는 생물학적 전이"로 읽는 것.
- 생성 표본 수를 늘려 생물학적 반복이 늘어났다는 해석.
- 공개 모델(CPA, sc2Flow, PerturbDiff, CellOT 등)과의 직접 비교: 동일 계약으로 재학습하지 않았으므로 하지 않는다.

## 알려진 한계

- **표현:** PCA 20차원은 표준화 HVG 2,000 분산의 약 4%만 설명한다. 모든 분포 지표는 이 저차원 공간의 지표이며, 유전자 Δmean은 같은 선형 decoder로 복원한 값이다(재구성 오차 별도 보고).
- **상태 atlas:** K=4 coarse soft partition. 상태 이름·생물학적 의미를 부여하지 않았다. K=2·8은 replicate 재현성 점수만 보고(모델 민감도 미실행).
- **QC:** hash QC threshold는 R10이 사전 고정한 값이며 원논문 규칙과의 일치는 미검증. 가공 배포본(scPerturb/pertpy) 교차확인 미실행.
- **신호 크기:** 대부분의 약물·용량에서 처리 조건과 vehicle의 구성비 차이가 replicate well 사이 차이(대부분 유한 표본 잡음)와 같은 수준이다. 강한 반응 약물은 소수다.
- **replicate 구조:** MCF7은 vehicle끼리도 replicate 간 이동이 크다. cross-replicate 계약에서 learned head는 이 이동을 배울 수 있다.
- **불확실성:** 약물 단위 bootstrap 구간은 관측된 두 replicate·plate 환경에 조건부다. development는 early stopping·선택에 쓰였으므로 development 수치는 낙관적일 수 있다. test는 freeze 후 한 번 평가했다.
- **구조 그룹:** 동일 SMILES 묶음만 적용(해당 0쌍). scaffold 수준 묶음은 하지 않았다.
