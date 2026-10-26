# 연구 확장 V2 데이터와 endpoint

원본·가중치·cell별 예측은 서버 데이터 저장소에 두고 공개 Git에는 작은 실제 결과와 provenance만 저장한다. `dataset_registry.tsv`와 각 contract를 함께 읽는다. Source cell count, 분석 cell count, 학습 example count는 서로 다른 값이다.

| 데이터 | 실제 사용 범위와 단위 | 정답·제한 |
|---|---|---|
| GSE149614 | 개발8donor/63036cells, donor-disjoint4fold; legacy02/09 제외 | 원 broad6lineage; observational 분류. 이미 반복 관찰한 개발 OOF |
| 공식 scGPT MS | reference7844/query13468, 원3000gene | 공식 annotation 흐름의 수정 재현; 저장 notebook 수치와 차이 있음 |
| 공식 GEARS Norman | 제공91205cells→공식GO 적격89357cells/5045genes, simulation seed1, test107interventions | K562 CRISPRa, single/dual-target. 원 DE dictionary는 공식 재현에만 사용 |
| GSE264667 HepG2 | target1714/214/214 train/val/test, test13994cells, same-batch non-targeting4976controls | day7 CRISPRi; dual sgRNA는 같은 gene 설계. Hepatoblastoma 유래이며 adult HCC 환자와 구분 |
| HepG2 native common | 동일3303관측gene, primary1977, test191targets/12839cells | scGPT flags에 target이 실제 panel에 있어야 함. 1544training targets×4treated와4976controls. Parent train-derived panel을 유지한 update-budget 비교 |
| Nault/scVIDR mouse liver | source30927cells/22213genes 중 author nonimmune6types18865cells, train17791/test1074portal-treated | TCDD30μg/kg,4일마다28일 노출. Same compound의 context transfer. Per-mouse assignment 불명, biological replicate CI 없음 |
| GSE151530 HCC | HCC만50023cells,25donors/32biopsies,18667providedgenes | 알려진5lineages27212cells, malignant/unclassified22811cells는 Unknown. Gold Hepatocyte 없음. 원 freeze 후8후보 최종 평가 완료; 이후 같은 cohort를 새 blind test로 재사용하지 않음 |

HepG2는 target gene의 guide·transcript 변형을 모두 같은 split에 묶었다. 기술 배치 gem_group를 biological replicate로 취급하지 않는다. 전체2000gene task와 native common1977-primary-gene task는 population이 달라 점수를 직접 비교하지 않는다. Native common에서 제외된 target/cell coverage도 contract에 기록했다.

Native GEARS와 scGPT의 두 initialization은 같은 training cells와 fixed matched control cell IDs를 사용한다. 통계적 mean 기준선은 같은 제공 control pool의 batch-matched 평균을 쓰므로 aggregation 방식이 다르다. 동일 구조·입력·optimizer로 사전학습 효과를 분리하는 직접 대조는 pretrained vs random이다. Response run의 metrics.n_cells 평균값은 조건당 평균 cell 수이며 전체 평가 cell 수는 n_test_cells 또는 evaluator provenance를 읽는다.

외부 HCC histology는 GEO sample characteristics와 원 연구의 TableS1을 교차 검증했다. 반복 biopsy suffix는 TableS1 footnote와 환자 ID로 묶었다. Malignant truth를 Hepatocyte로 바꾸지 않았다. Primary fixed-six macroF1의 최대치는5/6이며, secondary fixed-five와 coverage를 함께 보고한다. 공개 GEO sample ID 중복은 없지만 생물학적 동일인 여부와 foundation pretraining overlap은 unknown이다.

모델별 공식 입력을 유지한다. PCA는 train-only variance/HVG/scaler/PCA를 사용하고 query는 Ensembl 우선·고유 symbol 보조 정합 후 누락 gene만 raw0으로 채운다. Geneformer는 원 library sum 기반 V2 rank/CLS/EOS/4096, scGPT는 공식 collator와 길이1200 CLS를 사용한다. 서로 다른 입력 계약의 실용 파이프라인 비교이며 순수 architecture 인과 비교가 아니다.

KG는 기존 Reactome92 gene–pathway membership의 mean train-zscore 특징으로 보존한다. Pathway activation·causal intervention의 관측 정답이 아니다. 초기 concat 실험의 음성 결과를 새로운 external validation이나 실제 perturbation 예측으로 바꾸어 해석하지 않는다.
