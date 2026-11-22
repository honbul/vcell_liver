# V0+ 모델과 설정 선택

세 소스의 계통 그룹으로 외부 3-fold, 각 외부 fit 안에서 내부 2-fold를 고정한다. 설정을 고르는 스칼라는 소스 동등 평균 \(0.50\rho_{drug} + 0.25\rho_{cell} + 0.25\text{NDCG@5}_{cell}\)이다. 원 소스별 원시 AUC에 대한 두 평가 축과 Pearson·Precision@5를 모두 별도로 보고한다. 내부 학습은 fit-only 앵커 기준으로 보정한 다음 assay×drug 평균을 제거한다. 약물별 assay 동등 평균을 anchor로 더한다. 각 소스의 측정 밖 모델•약물 조합은 채점 표본으로 새로 만들지 않는다.

| 계열 | 등록 설정 |
|---|---|
| 약물 평균·가산 | fit-only anchor, 동등 가중 cell main effect Ridge 30/300 |
| 약물별 Ridge·ElasticNet | 256/512 고변이 RNA 및 원자료 표적 발현, Hallmark·PROGENy·DoRothEA 가중 서명. Ridge 10/100/300/1000, ElasticNet .002/.007/.02 |
| LightGBM·CatBoost | depth 5/7, 450/650 trees, fingerprint와 약물 코드 및 anchor, 256 고변이+서명 RNA |
| CellHit | 저자 ensemble 구현의 native assay×drug XGBoost 회귀 2멤버, depth 3/5. 그룹 검증 early stopping. 공개 서비스의 기존 checkpoint를 사용하지 않는다. |
| Velodrome | 저자 512→128 backbone 및 CORAL, assay별 head의 출력에 68개 약물 축을 추가. query RNA 사용 유무를 구분. |
| CODE-AE | 저자 private/shared autoencoder 및 MMD 사용. 원 임상 이진 head를 source 약물별 연속 반응 head로 바꾼다. query RNA 사용 유무를 구분. |
| V0+ | 768 고변이 RNA+표적 유전자, Hallmark·PROGENy·DoRothEA 서명, ECFP4 2,048비트·약물 embedding, 표적 유전자별 발현 및 Chronos 추정값을 쓰는 drug-target attention. Chronos와 driver hotspot/CNV는 fit 세포에서만 보조 감독한다. query 단계는 RNA와 약물만 받는다. |

CellHit·Velodrome·CODE-AE는 연속 AUC와 공통 패널에 맞춘 비교 구현이며, 저자 논문의 원래 코호트·수치와는 별개다. 저자 downstream checkpoint·전사체·외부 조직 정답을 재사용하지 않는다. 공개 구현은 세 저장소의 commit과 호출된 소스 해시를 기록하고 G1에서 직접 로드한다. [CellHit](https://github.com/raimondilab/CellHit/tree/9d05b0881768fedecb0842305ace51a3dabe798a) · [Velodrome](https://github.com/hosseinshn/Velodrome/tree/453dd5883337f32a3e667bf63cc53b681fdc363d) · [CODE-AE](https://github.com/XieResearchGroup/CODE-AE/tree/46b448bcadfafb4c8f552233aa8392f47d2e6b33)

V0+의 지도 손실은 source별 경향을 뺀 반응에 대한 가중 SmoothL1과 소스 안에서 약물 고정·계통 고정 순위 손실을 번갈아 적용하는 조합이다. 도메인 정렬은 source-only covariance와 query RNA를 포함한 covariance를 따로 평가한다. 모든 estimation은 지도 fit 행에 제한한다. 192/320 hidden, 70/100 epochs, 출력 배율 0.25/0.5/1.0의 두 주 후보군을 안쪽 fold에서 선택한다. 동일 192 hidden·70 epochs·배율 1.0에서 Chronos, driver, pathway, target attention, rank 정규화, 도메인 정렬을 각각 뺀 절제를 비교한다. 표준화와 변이 유전자 선택은 각 fit RNA에서만 다시 추정한다. [서명 리소스](https://github.com/saezlab/ConservedFootprints/tree/b93ffed985ba4b83e93cae9dc52a0f76f1a45755) · [DepMap 공개 데이터](https://depmap.org/portal/data_page/)

최종 설정은 세 outer fit에서 모인 inner 검증 평균으로 정한다. LIMORE 및 적격 3D 코호트 AUC 결과는 그 과정에 쓰지 않는다. 주 결과의 최소 조건은 두 외부 축에서 5개 독립 환자·계통 이상인 약물에 대해 약물 내 Spearman·Pearson이 0보다 크고, 세포 내 전체 Spearman 및 NDCG@5가 drug mean 대비 개선되는 것이다. 결과 표에 규모, paired 차이와 구간, 두 축의 엇갈림을 함께 표시한다.

소스 검증이 끝난 뒤 LIMORE V0+ 예측·채점 전에 등록하는 보조 arm은 `ridge_molecular_average`다. 같은 소스 anchor를 쓰는 약물별 Ridge와 inductive V0+ 예측의 50:50 평균이며 배율을 다시 고르지 않는다. 소스 OOF의 약물 내 Spearman .379와 Pearson .422를 근거로 추가했다. Ridge는 각각 .366과 .404였다. 세포 내 NDCG@5는 .892 대 .893이었다. 소스 OOF로 동기를 얻은 보조 arm이며 외부 점수에서 주력 모델을 소급 대체하지 않는다.
