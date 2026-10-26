# 서버 실제 실행 결과

이 보고서는 results/*/run.json의 실제 실행 결과로 생성했다. 문헌 재현 benchmark가 아니라 GSE149614의 신규 HCC 적용이다. frozen cell embedding + supervised logistic head이며 전체 시스템을 zero-shot classifier로 부르지 않는다.

## 실제 성능과 자원

| run_id | status | n_cells | macro_f1 | balanced_accuracy | n_test | n_unknown | peak_allocated_gib | peak_reserved_gib | peak_process_gpu_gib | peak_cpu_rss_gib | wall_seconds | cli_wall_seconds | prediction_coverage | evaluation_coverage | shuffle_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geneformer_comparison | executed | 10000 | 0.9456 | 0.9463 | 1240 | 0 | 1.6721 | 2.2324 | 2.5000 | 2.3955 | 1164.5205 | 1181.5103 | 1.0000 | 1.0000 | 0.1242 |
| geneformer_pilot | executed | 2000 | 0.9210 | 0.9113 | 253 | 0 | 1.6485 | 2.2012 | 2.4688 | 1.5628 | 259.8831 | 271.9505 | 1.0000 | 1.0000 | 0.1467 |
| geneformer_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | 0.6415 | 0.7363 | 0.9453 | 1.3141 | 49.7754 | 62.6513 | 1.0000 | 1.0000 | 0.0000 |
| kg_comparison_foundation_pathway | executed | 10000 | 0.9436 | 0.9428 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | nan | 1.0000 | 1.0000 | 0.1047 |
| kg_comparison_foundation_random | executed | 10000 | 0.9531 | 0.9495 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | nan | 1.0000 | 1.0000 | 0.1050 |
| kg_comparison_pathway | executed | 10000 | 0.7842 | 0.7722 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | nan | 1.0000 | 1.0000 | 0.1001 |
| kg_pilot_foundation_pathway | executed | 2000 | 0.9288 | 0.9292 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | nan | 1.0000 | 1.0000 | 0.1600 |
| kg_pilot_foundation_random | executed | 2000 | 0.9230 | 0.9262 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | nan | 1.0000 | 1.0000 | 0.1515 |
| kg_pilot_pathway | executed | 2000 | 0.8018 | 0.7892 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | nan | 1.0000 | 1.0000 | 0.1169 |
| pca_comparison | executed | 10000 | 0.9566 | 0.9498 | 1240 | 0 | nan | nan | nan | 1.3629 | 6.4029 | 11.3206 | 1.0000 | 1.0000 | 0.0878 |
| pca_pilot | executed | 2000 | 0.9359 | 0.9257 | 253 | 0 | nan | nan | nan | 0.6781 | 1.4759 | 5.7727 | 1.0000 | 1.0000 | 0.1850 |
| pca_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | nan | nan | nan | 0.5197 | 0.6546 | 5.3255 | 1.0000 | 1.0000 | 0.2833 |
| scgpt_comparison | executed | 10000 | 0.9527 | 0.9496 | 1240 | 0 | 0.3676 | 0.4453 | 0.7207 | 2.6684 | 72.0935 | 87.8211 | 1.0000 | 1.0000 | 0.1165 |
| scgpt_pilot | executed | 2000 | 0.9327 | 0.9337 | 253 | 0 | 0.3676 | 0.4238 | 0.6992 | 1.5240 | 17.5503 | 24.9308 | 1.0000 | 1.0000 | 0.1462 |
| scgpt_smoke | executed | 64 | 0.7667 | 0.8333 | 12 | 0 | 0.2602 | 0.2793 | 0.5527 | 1.2396 | 5.8722 | 11.2782 | 1.0000 | 1.0000 | 0.0667 |
| tutorial_geneformer_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | 0.6415 | 0.7363 | 1.0039 | 1.3139 | 30.5522 | nan | 1.0000 | 1.0000 | 0.0000 |
| tutorial_kg_smoke_foundation_pathway | executed | 64 | 0.6230 | 0.6667 | 12 | 0 | nan | nan | nan | 0.6169 | 4.6391 | nan | 1.0000 | 1.0000 | 0.0667 |
| tutorial_kg_smoke_foundation_random | executed | 64 | 0.8278 | 0.8333 | 12 | 0 | nan | nan | nan | 0.6169 | 4.6391 | nan | 1.0000 | 1.0000 | 0.1778 |
| tutorial_kg_smoke_pathway | executed | 64 | 0.1786 | 0.2500 | 12 | 0 | nan | nan | nan | 0.6169 | 4.6391 | nan | 1.0000 | 1.0000 | 0.0000 |
| tutorial_pca_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | nan | nan | nan | 0.5194 | 0.2201 | nan | 1.0000 | 1.0000 | 0.2833 |
| tutorial_scgpt_smoke | executed | 64 | 0.7667 | 0.8333 | 12 | 0 | 0.2602 | 0.2793 | 0.5547 | 1.2535 | 6.2624 | nan | 1.0000 | 1.0000 | 0.0667 |

## Reactome 추가 실험 (pilot)

| run_id | macro_f1 | balanced_accuracy | selected_C | selected_class_weight | n_changed_test_predictions_vs_foundation | delta_macro_f1_vs_foundation |
| --- | --- | --- | --- | --- | --- | --- |
| kg_pilot_pathway | 0.8018 | 0.7892 | 0.1000 | nan | 55.0000 | -0.1309 |
| kg_pilot_foundation_pathway | 0.9288 | 0.9292 | 0.0100 | nan | 4.0000 | -0.0039 |
| kg_pilot_foundation_random | 0.9230 | 0.9262 | 0.1000 | nan | 8.0000 | -0.0097 |
| pca_pilot | 0.9359 | 0.9257 | 0.1000 | nan | nan | nan |
| scgpt_pilot | 0.9327 | 0.9337 | 0.0100 | balanced | nan | nan |

scGPT 대비 경로 결합 macro-F1 차이 -0.0039; 크기/유전자 참여 분포를 보존한 random 대조 대비 +0.0058. Test prediction 변경 수: 4.

## Reactome 추가 실험 (comparison)

| run_id | macro_f1 | balanced_accuracy | selected_C | selected_class_weight | n_changed_test_predictions_vs_foundation | delta_macro_f1_vs_foundation |
| --- | --- | --- | --- | --- | --- | --- |
| kg_comparison_pathway | 0.7842 | 0.7722 | 1.0000 | nan | 300.0000 | -0.1684 |
| kg_comparison_foundation_pathway | 0.9436 | 0.9428 | 0.0100 | nan | 27.0000 | -0.0091 |
| kg_comparison_foundation_random | 0.9531 | 0.9495 | 0.0100 | nan | 18.0000 | 0.0005 |
| pca_comparison | 0.9566 | 0.9498 | 0.0100 | nan | nan | nan |
| scgpt_comparison | 0.9527 | 0.9496 | 0.0100 | nan | nan | nan |

scGPT 대비 경로 결합 macro-F1 차이 -0.0091; 크기/유전자 참여 분포를 보존한 random 대조 대비 -0.0095. Test prediction 변경 수: 27.

## 가상 삭제의 실제 결과

Train hepatocyte100세포, ALB/HNF4A 독립 삭제와 발현량 유사 무작위 대조 및 no-op. token_present=False는 삭제 대상 token이 없었던 세포다. **실측 섭동 검증 전 탐색**이며 발현량/약효 예측이 아니다.

gene	condition	token_present	count	mean	median	max
ALB	independent_delete	False	35	8.514949253627232e-09	0.0	5.960464477539064e-08
ALB	independent_delete	True	65	0.0001392850509056	0.0001253485679626	0.0004571676254272
HNF4A	independent_delete	False	54	1.3245476616753471e-08	0.0	5.960464477539064e-08
HNF4A	independent_delete	True	46	8.44183175460152e-05	7.092952728271484e-06	0.0003913044929504
HNRNPAB	expression_matched_random_delete	False	57	1.6731128358004385e-08	0.0	5.960464477539064e-08
HNRNPAB	expression_matched_random_delete	True	43	8.540652519048647e-05	5.602836608886719e-06	0.0003940463066101
MT-CO1	expression_matched_random_delete	True	100	7.373273372650146e-05	2.98917293548584e-05	0.0004190802574157
none	no_op	True	100	1.4901161193847656e-08	0.0	5.960464477539064e-08


완료 확인: 실제 실행·데이터/분할/예측 정합성 110개 점검과 한국어 노트북 6개의 새 kernel Run All이 통과했다. 필수 단계의 미해결 차단은 없다. 사전학습 overlap은 unknown으로 남겼고 선택적 fine-tuning/scFoundation/GEARS는 실행하지 않았다.

## 해석과 한계

Macro-F1와 balanced accuracy는 고정된 두 test 환자에서 측정했다. 환자별/클래스별 지표와 confusion matrix는 각 results/<run_id>에 있다. 환자별 macro-F1도 전체 train class 목록을 기준으로 하므로 해당 환자에 없는 class는 0을 포함한다. test 환자가 두 명뿐이어서 일반화와 신뢰구간 해석이 제한적이며 세포 bootstrap CI를 제시하지 않는다. 64세포 smoke는 동작 확인용이다.

PCA HVG(로그 정규화 분산 상위2000), scaler, PCA, classifier는 train만 학습했다. C/weight와 경로 수는 validation으로 골랐다. 모든 모델은 동일 split/cell IDs/class schema다. 모델별 공식 입력 범위가 다르므로 아키텍처만의 비교가 아니다. 원 cohort와 사전학습 데이터의 중복은 unknown이며 pretraining-unseen 성능을 주장하지 않는다. shuffled-train-label 결과는 label feature 혼입에 대한 간단한 sanity check이며 별도 유효성 증명은 아니다.

Reactome membership은 외부 prior, gene expression은 환자 관측이다. 경로 mean-zscore는 활성화/인과관계의 측정값이 아니다. 실제 입력에 concatenate한 모델을 random 대조와 비교했으며 이득을 보장하지 않는다.

가상 삭제는 실측 섭동 검증 전 탐색. 독립 gene token 삭제 전후 CLS cosine shift이며 post-perturbation expression/logFC/약효/생존율이 아니다. 신뢰할 목표 상태가 없어 변화 크기만 보고한다. 결과는 virtual_perturbation_*와 한계 문서 참조.

GEARS, scFoundation 설치/학습, 미세조정은 이번 필수 범위에서 실행하지 않았다. GEARS는 실제 genetic intervention 데이터가 필요한 조건부 후속이다. 공식 논문 예제의 정량적 성과나 생물학적 검증을 재현했다고 주장하지 않는다.

## 재실행

```bash
cd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver
source envs/activate.sh scgpt
python scripts/02_baseline_annotation.py --config configs/pca_pilot.yaml
python scripts/03_scgpt_annotation.py --config configs/scgpt_pilot.yaml
source envs/activate.sh geneformer
python scripts/04_geneformer_annotation.py --config configs/geneformer_pilot.yaml
source envs/activate.sh scgpt
python scripts/05_kg_features_ablation.py --config configs/kg_pilot.yaml
source envs/activate.sh geneformer
python scripts/06_geneformer_virtual_perturb.py --config configs/isp_pilot.yaml
source envs/activate.sh scgpt
python scripts/08_execute_notebooks.py
python scripts/07_collect_results.py --runs results --out reports/final_summary.md
```

## 다음 한 가지 실험

두 test 환자만의 변동을 확인하기 위해 현재 고정 protocol을 유지한 leave-one-patient-out 평가를 별도 실험으로 수행한다. 현재 test 결과로 모델 설정을 바꾸지 않는다.

자원 측정 범위: model run의 wall_seconds는 checkpoint 로딩/입력 처리/embedding 구간이며 head CPU 실행은 제외한다. PCA는 전처리·분류기까지 포함한다. CLI 전체 시간은 cli_wall_seconds와 reports/stages에 별도로 있다. KG 세 head의 자원 값은 공통 feature 준비와 세 head 전체 실행 구간을 공유한다. 다른 GPU 작업과 공유한 서버에서의 관측이며 속도 우열 benchmark가 아니다. CUDA allocated와 process GPU memory를 구분한다.
