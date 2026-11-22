# LIMORE 50계통의 정정 학습 자료 기술 평가

이전 LIMORE 외부 회차의 CTRPv2 학습표에 navitoclax 및 rapamycin의 혼합 처리 1,394행이 있었다. 학습 입력은 원 공개 스크린의 정확한 단일 약물 관측으로 정정했다. 이 기술 평가의 목적은 같은 50계통·68약물의 성적이 이 정정으로 얼마나 달라지는지 확인하고, 사용 중인 정정 V0+ 패키지의 직접 입력 지원을 검증하는 것이다. 이전 LIMORE 성적과 결론은 별도 이력으로 유지한다. [정정 근거](SOURCE_PANEL_EXTENSION.md) · [기존 동결 성적](STEP4_EXTERNAL_RESULTS.md)

- 등록된 3,400개 관측과 원래 RNA 축을 사용한다. 비교 모델의 설정은 기존 소스 검증에서 고른 settings SHA-256 `9fb42665b3d5bce1757bab31f341c256bc2f9bcf5434e6239c48c47dd3be3474`로 고정한다. 기존 정정 68약물 코어에서 약물 평균, 가산, 약물별 Ridge·ElasticNet, LightGBM, CatBoost, CellHit, Velodrome inductive·transductive, CODE-AE inductive·transductive, V0+ inductive·transductive를 다시 적합한다. 동결된 원래 예측의 같은 arm을 비교한다. Transductive arm은 보류 RNA 정렬만 적용한다.
- 구조 커널은 기존 벌점 4·잔차 배율 1이며 과거 코어와 정정 코어를 각각 적합한다. 정확히 같은 RNA·표적·경로·ECFP·Chronos·driver/CNV proxy를 사용한다. 각 코어 조건에서 구조 커널+Ridge의 고정 50:50 평균, V0++Ridge의 50:50 평균을 비교한다. 추가 약물 및 배율의 탐색은 없다.
- 별도 패키지 arm인 약물 평균·Ridge·V0+는 배포한 `single-agent-corrected-70` 모델을 CPU CLI로 그대로 실행한다. 입력을 LIMORE의 기존 68 exact parent와 기저 RNA로 제한한다. 이 arm은 Yang의 공통 RNA 축과 추가 2약물까지 포함해 적합했던 배포 모델이므로 68약물 코어 정정 효과의 비교 쌍에는 사용하지 않는다. [배포 모델 카드](V0PLUS_CORRECTED_MODEL_CARD.md)
- 모든 예상 행의 예측을 병합하고 SHA-256으로 동결한다. 그 뒤 별도 작업에서 등록된 LIMORE AA의 해시를 검증하고 채점한다. 약물 내 환자·계통 Spearman·Pearson, 계통별 Spearman·NDCG@5·Precision@5를 같은 행에서 계산한다. 전체 68약물, 정정 관련 2약물, 나머지 66약물, 사전 주석의 HCC TKI·표적·세포독성군을 보고한다. 같은 모델의 전후 차이와 정정 V0+ 및 두 결합의 정정 약물 평균·Ridge·CellHit 대비 차이에 환자·약물 교차 및 계통 단위 짝지은 95% 구간을 낸다. 별도의 새 외부 우위 주장은 이 회차에서 판정하지 않는다.
