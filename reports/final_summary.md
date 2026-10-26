# 서버 실제 실행 결과

이 보고서는 results/*/run.json의 실제 실행 결과로 생성했다. 문헌 재현 benchmark가 아니라 GSE149614의 신규 HCC 적용이다. frozen cell embedding + supervised logistic head이며 전체 시스템을 zero-shot classifier로 부르지 않는다.

## 실제 성능과 자원

| run_id | status | n_cells | macro_f1 | balanced_accuracy | n_test | n_unknown | peak_allocated_gib | peak_reserved_gib | peak_process_gpu_gib | peak_cpu_rss_gib | wall_seconds | shuffle_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| geneformer_pilot | executed | 2000 | 0.9210 | 0.9113 | 253 | 0 | 1.6485 | 2.2012 | 2.4688 | 1.5628 | 259.8831 | 0.1467 |
| geneformer_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | 0.6415 | 0.7363 | 0.9453 | 1.3141 | 49.7754 | 0.0000 |
| kg_comparison_foundation_pathway | executed | 10000 | 0.9436 | 0.9428 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | 0.1047 |
| kg_comparison_foundation_random | executed | 10000 | 0.9531 | 0.9495 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | 0.1050 |
| kg_comparison_pathway | executed | 10000 | 0.7842 | 0.7722 | 1240 | 0 | nan | nan | nan | 1.5041 | 52.7630 | 0.1001 |
| kg_pilot_foundation_pathway | executed | 2000 | 0.9288 | 0.9292 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | 0.1600 |
| kg_pilot_foundation_random | executed | 2000 | 0.9230 | 0.9262 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | 0.1515 |
| kg_pilot_pathway | executed | 2000 | 0.8018 | 0.7892 | 253 | 0 | nan | nan | nan | 0.7464 | 28.7677 | 0.1169 |
| pca_comparison | executed | 10000 | 0.9566 | 0.9498 | 1240 | 0 | nan | nan | nan | 1.3629 | 6.4029 | 0.0878 |
| pca_pilot | executed | 2000 | 0.9359 | 0.9257 | 253 | 0 | nan | nan | nan | 0.6781 | 1.4759 | 0.1850 |
| pca_smoke | executed | 64 | 0.6944 | 0.7500 | 12 | 0 | nan | nan | nan | 0.5197 | 0.6546 | 0.2833 |
| scgpt_comparison | executed | 10000 | 0.9527 | 0.9496 | 1240 | 0 | 0.3676 | 0.4453 | 0.7207 | 2.6684 | 72.0935 | 0.1165 |
| scgpt_pilot | executed | 2000 | 0.9327 | 0.9337 | 253 | 0 | 0.3676 | 0.4238 | 0.6992 | 1.5240 | 17.5503 | 0.1462 |
| scgpt_smoke | executed | 64 | 0.7667 | 0.8333 | 12 | 0 | 0.2602 | 0.2793 | 0.5527 | 1.2396 | 5.8722 | 0.0667 |

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
