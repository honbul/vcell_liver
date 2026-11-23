# H3 연구 모델 카드

| 항목 | 정의 |
|---|---|
| 작업 | 성인 포함 배양 세포 계통의 연구용 in vitro 감수성 순위와 약물 내 반응 점수 |
| 모형 | H3a의 96→16 bilinear RNA·화학 encoder, 고정 약물 anchor, 세포별 additive head |
| 입력 | 사전 처리 basal bulk RNA의 17,871개 등록 유전자 축, 정확 parent SMILES의 ECFP4 |
| 적합 특징 | 표본별 RNA rank, 적합 계통으로만 정한 512개 변동 유전자와 정규화 기준 |
| 학습 | 정정 65,678개 원래 실측에서 분리된 적합 행, 32 epoch·384 update, fit-only assay 교정 |
| 체크포인트 | G1 H3 루트 outer-forecast의 분할별 3-seed 연구용 상태·특징·anchor, 분할과 해시는 영수증 참조 |
| 재현 경계 | 당시 등록 68개 parent 내 학습/평가 지지. 스크린 보류 시 적합 소스에 관측 있는 정확 parent만 평가 |
| 검증 | 단일 패스의 3개 스크린 × 3계통 분할, 전역 계통 단위 짝지은 95% CI, seed의 표본 SD |
| 배포 판단 | 승격 기준과 채점 결과는 아래 링크에 기록. 모델은 배양 실험 재현을 위한 연구용이다. |

동일 환자의 치료 효과, 임상 반응, 용량 선택과 비교효과는 검증 대상 범위 밖이다. 기존 직교분해와 HCC 결과에는 과거 평가 노출이 있다. [결과와 판정](RESULTS.md) · [데이터와 노출](DATA_SUPPORT_AND_EXPOSURE.md) · [동결 체크포인트 해시](score_receipt.json)
