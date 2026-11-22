# DILI v2 구조 기반 FDA 간독성 concern 예측

최종 Murcko와 Butina AUROC의 점 추정치는 각각 v1의 .7256과 .7274에서 v2의 .7484와 .7466으로 올랐다. FDA 신규 출시 증분 155개에서는 AUROC가 .6767에서 .6651로 낮아졌고 ECE는 .3606에서 .3867로 높아졌다. 세 AUROC 차이의 paired 99% CI가 모두 0을 포함해 증분 검증을 요구하는 승격은 보류한다.

## 내부 scaffold 개발

FDA 신규 증분 155개를 모든 개발 fold에서 제외했다. 나머지 732개를 기존 Murcko·Butina 등록 폴드에 유지했고, 내부 설정에는 각 외부 개발 fold 안에서 그룹 5-fold를 사용했다. 아래 값은 개발 holdout 다섯 폴드의 평균이다. 기존 v1의 887개 전체 OOF 점수와는 학습 및 평가 범위가 다르다.

| 개발 단계 | 대표 비교 | Murcko AUROC / AUPRC | Butina AUROC / AUPRC |
|---|---|---:|---:|
| R1 재적합 대조 | v1 구조+ChemBERTa LightGBM | .7792 / .8656 | .7567 / .8546 |
| R1 표현·분류기 탐색 | ExtraTrees | .8026 / .8754 | .7943 / .8700 |
| R2 그래프 단독 | D-MPNN | .7410 / .8373 | .7704 / .8506 |
| R2 기전 다중과제 | D-MPNN + 공개 기전 | .7451 / .8398 | .7709 / .8503 |
| R3 오류 가중 및 보정 | 가중 ExtraTrees | .8028 / .8765 | .7924 / .8643 |
| R4 선택 | ExtraTrees 중심 이종 스태킹 | **.8015 / .8773** | **.7997 / .8804** |

- R1은 ECFP4, RDKit 물리화학 및 PAINS·Brenk·반응성 구조 경고, ChemBERTa, MoLFormer XL, Uni-Mol 3D 및 CLAMP를 비교했다. LightGBM, CatBoost GPU, XGBoost, ExtraTrees, 로지스틱, 일반 Tanimoto·MinMax·RBF SVM을 사용했다.
- R2는 5개 학습 fold의 실제 방향성 분자 그래프 모델을 GPU에서 적합했다. Tox21, ToxCast v4.3, ChEMBL에서 공개 데이터를 수집했다. ToxCast의 대상 엔드포인트 33개에서 135,888개 hit-call을 확보했고, 9,389개 중 8,777개 DSSTox 식별자에 구조가 일치했다. 전체 FDA 카탈로그와 connectivity·scaffold가 일치하는 기전 어세이를 제거한 뒤 12,913개 구조와 42개 기전 표지를 사용했다. 다중과제와 별도의 OOF 기전 점수를 시험했다.
- R3에서는 고분자량, 지용성, 전하, 금속, 구조 경고 및 FM 누락 그룹의 오분류를 살폈다. Platt, temperature, beta, isotonic, 입력 기반 유병률 보정을 그룹 OOF로 평가했다. nearest Tanimoto로 confidence 순서를 정했다.
- R4는 ExtraTrees·가중 ExtraTrees·XGBoost·RBF SVM·D-MPNN·기전 D-MPNN을 포함하는 비음수 결합을 개발 결과로 선택했다. 최근접 Tanimoto에 따른 보류곡선을 계산했다.

[개발 폴드](evidence/cv_fold_metrics.tsv) · [모델 선택 영수증](evidence/internal_policy.json) · [오류 그룹](evidence/subgroup_error.tsv) · [대표 오분류](evidence/largest_oof_errors.tsv) · [보정 비교](evidence/calibrator_cv.tsv) · [공개 기전 영수증](evidence/epa_target_receipt.json) · [D-MPNN 실행 영수증](evidence/mechanism_fit_replication.json)

## 동결 후 단일 채점

설정과 예측 11개를 동결한 뒤 세 분할을 한 프로세스에서 채점했다. Murcko·Butina의 각각 887개 중 양성 535개, 증분 155개 중 양성 47개였다. v2 scaffold 적합은 개발부터 최종 적합까지 신규 155개를 전역 제외했다. 기록된 v1 scaffold OOF 모델은 각 fold 학습에 들어 있던 신규 행을 사용할 수 있었다. v1과 v2는 각 분할에서 같은 평가 행에 짝지었으며, 증분 모델은 이전 출시의 라벨만 사용했다. v2 증분 학습은 702행과 681개 고유 connectivity였다. [예측 동결](evidence/forecast_freeze.json) · [이전 출시 모델](evidence/historical_fit_manifest.json) · [단일 채점 원표](evidence/final_score.json)

| 분할 | 모델 | AUROC | AUPRC | Bal.Acc | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| Murcko 887 | v1 | .7256 | .7683 | .6748 | .2024 | .0381 |
| Murcko 887 | v2 | .7484 | .8055 | .6785 | .1995 | .0796 |
| Butina 887 | v1 | .7274 | .7923 | .6715 | .2043 | .0416 |
| Butina 887 | v2 | .7466 | .7945 | .6697 | .1990 | .0727 |
| FDA 증분 155 | v1 | .6767 | .4910 | .6112 | .3261 | .3606 |
| FDA 증분 155 | v2 | .6651 | .4751 | .6366 | .3543 | .3867 |

| v2 − v1, paired 99% CI | AUROC | AUPRC | Bal.Acc | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Murcko | +.0228 [−.0139, .0569] | +.0371 [−.0040, .0711] | +.0037 [−.0364, .0409] | −.0029 [−.0156, .0114] | +.0415 [−.0144, .0809] |
| Butina | +.0191 [−.0142, .0530] | +.0022 [−.0340, .0396] | −.0017 [−.0465, .0422] | −.0053 [−.0186, .0087] | +.0311 [−.0187, .0711] |
| FDA 증분 | −.0116 [−.0976, .0746] | −.0159 [−.1392, .1145] | +.0254 [−.0498, .1077] | +.0283 [−.0018, .0572] | +.0262 [−.0153, .0596] |

각 구간은 Murcko·Butina의 지정 그룹 또는 FDA 증분의 connectivity를 단위로 짝을 유지해 10,000회 재표집했다. Brier와 ECE는 낮을수록 좋다. 역사적 개발에서 유병률 이동 모의 검증의 ECE는 보정 후보 중 prior EM이 .1375로 가장 낮았다. 최종 증분 ECE는 v1을 앞서지 못했다.

## 최근접 Tanimoto 보류

유사도가 높은 순으로 지정 비율을 남겼다. 아래는 v2의 AUROC이며, 괄호는 평가 행 수다.

| 분할 | 100% | 80% | 70% |
|---|---:|---:|---:|
| Murcko | .7484 (887) | .7708 (710) | .7877 (621) |
| Butina | .7466 (887) | .7742 (710) | .7924 (621) |
| FDA 증분 | .6651 (155) | .6484 (124) | .6356 (108) |

신규 출시에서는 보류율이 20%와 30%일 때 AUROC와 보정이 모두 악화됐다. CLI의 Tanimoto 기준 .195는 개발 자료에서 동결한 지원 범위 표지다. 오프라인에서 `CCO`는 확률 .2740과 최근접 Tanimoto .2778을 반환했다. 동결 FM 조회에 없는 `C[Si](C)(C)C`도 구조·그래프로 추론했고, 유사도 .1000에 지원 범위 밖 표지를 반환했다. 잘못된 SMILES 입력은 거부했다. [보류 곡선](evidence/risk_coverage.pdf) · [일반 CLI](evidence/offline_receipt.json) · [새 구조 CLI](evidence/offline_unregistered.json) · [입력 오류](evidence/offline_invalid.receipt) · [작업 원장](evidence/jobs.tsv) · [모델 카드](model_card.md)
