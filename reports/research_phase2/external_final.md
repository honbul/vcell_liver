# 외부 GSE151530 평가 범위와 현재 상태

**현재 고정8후보의 최종 추론이 진행 중이며 최종 평가 완료가 아니다.** 최신 Slurm·완료 artifact 관측은 [current_status.json](external_final/current_status.json)에 남긴다. 원 [freeze](external_final/freeze.json)는 변경하지 않는다. 모든 후보 추론과 hash/order 검증이 끝나야 `external_final/evaluation.json`, `metrics.tsv`,25donor paired CI 및 Unknown 포함 confusion이 생성된다.

평가 대상은 HCC 환자25명/32biopsies/50023cells에서 저자가 annotation한 다섯 **비악성 lineage**다: T/NK17348, Myeloid4616, Endothelial2338, B1854, Fibroblast1056. Known truth는27212cells(54.40%)다. Malignant/unclassified22811cells는 Unknown으로 유지하고 알려진 lineage accuracy에서 제외하되 confusion/support로 보고한다. 이들 세포의 모델 예측을 검증된 malignant lineage나 Hepatocyte 정답으로 바꾸지 않는다.

Primary fixed-six macroF1은 absent Hepatocyte를0으로 포함해 ceiling5/6이다. 사전 정의 secondary fixed-five와 uncalibrated probability의 ECE/Brier를 known truth에서 함께 보고한다. Fixed-five 역시 hepatocyte 전이·악성 판정 검증이 아니다. Balanced accuracy는 관측된 known5class의 mean recall이므로 개발6class 값과 동일 ceiling을 갖는 숫자로 직접 비교하지 않는다.

후보는12000개 development label로 fit한 PCA/scGPT/Geneformer/CLcancer, 별도4096개 budget의 DM00/10/01/11이다. Development primary 선정은 Geneformer frozen linear이며, 외부 점수에 따라 primary나 cohort를 바꾸지 않는다.12000과4096의 조건을 분리한다. 반복 biopsy는 저자 환자 ID로 묶어25donor bootstrap5000회를 사용하며 모델 재학습 불확실성은 포함하지 않는다.

원12892는 freeze 후 첫 PCA prediction 전 categorical gene metadata 처리에서 실패했다. 실제 개발64cells에서 자료형만 바꾼 네 baseline 확률 동등성을 확인했고,12919는 같은 freeze/query/model bundle을 검증해 복구한다. 원 output의 확인된 빈 directory만 archive했으며 기존 예측·점수를 삭제하거나 새 blind test로 재발행하지 않았다. 선택에 무관한 KG 진단은 이제 release의 필수 의존성이 아니다. 상세는 [의존성 점검](external_release_dependencies.json), [복구 시작 기록](external_final/attempts/external_categorical_recovery_v1/start.json)에 있다.

Public GEO sample 중복은 확인한 manifest에서 없지만 biological identity와 foundation pretraining overlap은 unknown이다. 최종 batch 이후 이 cohort로 모델을 변경하면 holdout 독립성은 소모된다.
