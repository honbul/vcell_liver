# HCC 환자 외부 cohort의 알려진 비악성 5개 세포 계열 전이 평가

**원래 고정한8후보의 추론과 단일 최종 평가를 완료했다.** Slurm12919의 terminal log는 `EXTERNAL_FINAL_COMPLETE`와 `FROZEN_EXTERNAL_BATCH_RECOVERED`로 끝나며, [복구 완료 기록](external_final/attempts/external_categorical_recovery_v1/completed.json)이 [evaluation.json](external_final/evaluation.json)의 hash를 연결한다. 오래된 Slurm handle은 현재 scheduler 조회에서 만료됐으며, 이를 실패로 해석하거나 재제출하지 않았다. [최종 검증](external_final/completion_verification.json)은 원 freeze·8bundle·예측과 지표 정합성을 확인한다. 원 [freeze](external_final/freeze.json)는 변경하지 않았다.

평가 대상은 HCC 환자25명/32biopsies/50023cells에서 저자가 annotation한 다섯 **비악성 lineage**다: T/NK17348, Myeloid4616, Endothelial2338, B1854, Fibroblast1056. Known truth는27212cells(54.40%)다. Malignant/unclassified22811cells는 Unknown으로 유지하고 알려진 lineage accuracy에서 제외하되 confusion/support로 보고한다. 이들 세포의 모델 예측을 검증된 malignant lineage나 Hepatocyte 정답으로 바꾸지 않는다.

Primary fixed-six macroF1은 absent Hepatocyte를0으로 포함해 ceiling5/6이다. 사전 정의 secondary fixed-five와 uncalibrated probability의 ECE/Brier를 known truth에서 함께 보고한다. Fixed-five 역시 hepatocyte 전이·악성 판정 검증이 아니다. Balanced accuracy는 관측된 known5class의 mean recall이므로 개발6class 값과 동일 ceiling을 갖는 숫자로 직접 비교하지 않는다.

후보는12000개 development label로 fit한 PCA/scGPT/Geneformer/CLcancer, 별도4096개 budget의 DM00/10/01/11이다. Development primary 선정은 Geneformer frozen linear이며, 외부 점수에 따라 primary나 cohort를 바꾸지 않는다.12000과4096의 조건을 분리한다. 반복 biopsy는 저자 환자 ID로 묶어25donor bootstrap5000회를 사용하며 모델 재학습 불확실성은 포함하지 않는다.

| 학습 budget | 고정 후보 | Fixed-six F1(주) | Fixed-five F1(보조) | 전체/known cells | 정답 coverage | Unknown truth | Prediction coverage |
|---|---|---:|---:|---|---:|---:|---:|
| 12000 | PCA | .809865 | .971838 | 50023/27212 | 54.40% | 22811 | 100% |
| 12000 | scGPT | .825035 | .990042 | 50023/27212 | 54.40% | 22811 | 100% |
| 12000 | Geneformer(사전 primary) | .817066 | .980479 | 50023/27212 | 54.40% | 22811 | 100% |
| 12000 | CLcancer | .821433 | .985719 | 50023/27212 | 54.40% | 22811 | 100% |
| 4096 | DM00 자연+동결NN | .817813 | .981375 | 50023/27212 | 54.40% | 22811 | 100% |
| 4096 | DM10 donor-balanced+동결NN | .800339 | .960407 | 50023/27212 | 54.40% | 22811 | 100% |
| 4096 | DM01 자연+partial | .824212 | .989054 | 50023/27212 | 54.40% | 22811 | 100% |
| 4096 | DM11 donor-balanced+partial | .819049 | .982859 | 50023/27212 | 54.40% | 22811 | 100% |

원 [전체 지표](external_final/metrics.tsv)에는 known-truth ECE/Brier도 포함한다. ScGPT ECE .02530/Brier .01274, primary Geneformer ECE .02475/Brier .02735다. 모든 후보가 uncalibrated argmax이며 외부 threshold나 calibration fitting을 하지 않았다. Prediction coverage100%는 모든 셀에 broad label을 냈다는 뜻으로, Unknown45.60%의 정답을 알고 있다는 뜻이 아니다.

| 사전 정의 비교 | Fixed-six F1 차이 | 25donor paired95% CI |
|---|---:|---|
| scGPT−PCA | +.015170 | [+.007498,+.026163] |
| Geneformer−PCA | +.007201 | [−.001279,+.017661] |
| CLcancer−PCA | +.011568 | [+.004446,+.022236] |
| CLcancer−Geneformer | +.004367 | [+.001622,+.008921] |
| 4096 Data gain | −.017473 | [−.026072,−.009224] |
| 4096 Model gain | +.006399 | [+.002927,+.011673] |
| 4096 Combined gain | +.001237 | [−.007084,+.011056] |
| 4096 Interaction | +.012311 | [+.001049,+.022824] |

이 고정 batch에서 scGPT의 점추정치가 가장 높았지만 primary는 Geneformer로 유지한다. CLcancer는 이 외부 cohort에서 base 대비 이득을 보였으나 개발 평균에서는 반대였으므로 일관된 암 특화 우위를 주장하지 않는다. Partial의 model-only 외부 이득도 개발8donor/3seed에서는 CI가0을 포함했다. 외부 확인은 seed42 한 번의 고정 refit이며 다중 비교 보정이나 재학습 불확실성을 포함하지 않는다. 결과를 보고 후보·학습량·주지표를 바꾸지 않았다.

Donor-balanced 단독 변경은 Endothelial recall .98631→.84260, B .94229→.92071로 악화됐다. 결합은 단독 sampling 효과를 상쇄했지만 combined F1 CI는0을 포함해 안정적 전체 개선이라고 부르지 않는다. [Class별 표](external_final/DM10_per_class_known_truth.tsv), [donor별 표](external_final/by_donor.tsv), [8개 쌍대 효과](external_final/paired_donor_bootstrap.tsv), [비용 범위](external_final/resource_summary.tsv)를 함께 제공한다.

원12892는 freeze 후 첫 PCA prediction 전 categorical gene metadata 처리에서 실패했다. 실제 개발64cells에서 자료형만 바꾼 네 baseline 확률 동등성을 확인했고,12919는 같은 freeze/query/model bundle을 검증해 복구를 마쳤다. 원 output의 확인된 빈 directory만 archive했으며 기존 예측·점수를 삭제하거나 새 blind test로 재발행하지 않았다. 선택에 무관한 KG 진단은 이제 release의 필수 의존성이 아니다. 상세는 [의존성 점검](external_release_dependencies.json), [복구 시작 기록](external_final/attempts/external_categorical_recovery_v1/start.json)에 있다.

Public GEO sample 중복은 확인한 manifest에서 없지만 biological identity와 foundation pretraining overlap은 unknown이다. 이 cohort의 독립 최종 batch 사용은 끝났다. 이후 결과를 이용해 모델을 변경하면 새 개발 라운드로 취급하며, 같은 cohort를 다시 새로운 blind test로 부르지 않는다.
