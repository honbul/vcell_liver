# V3 진행 점검 수정사항 처리

작업 브랜치는 `research/phase2-v2`다. 검토 기준589159e460b4d0e78f3e9fe01fc699bcbc829602 이후의 기존 수정을 먼저 확인했고, 실제48조건과 native-common 결과를 재사용했다. 추가 분석 코드 기준은 a1e2f8f이며, V2 계획은 유지한다. 원 main243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79의 기존 notebook00–05와 환경 정의는 변경하지 않았다.

**V3 보완과 외부 고정8후보의 최종 평가·결과 정합 검증을 마쳤다.** 현재 관측 시각과 상태는 [current_status.json](external_final/current_status.json), 각 작업 근거는 [Slurm registry](slurm_jobs.tsv), 단일 진입점은 [결과 인덱스](RESULTS_INDEX.md)다.

| 수정사항 | 상태 | 실제 근거와 처리 |
|---|---|---|
| FIX-01 tissue metadata | fixed | [schema alias 해소](tissue_error_analysis/metadata_schema_resolution.json), [기존 canonical schema_mapping 관측](tissue_error_analysis/canonical_schema_mapping_observation.json). 원본 site와63036행 정확히 일치. Alias가 다르면 충돌 기록 후 중단하며 값을 덮어쓰지 않음 |
| 조직·환자·sample support | completed | [조직별](tissue_error_analysis/support_by_tissue_site.tsv), [donor별](tissue_error_analysis/support_by_donor_tissue_site.tsv). Tumor8/Normal7/PVTT2/Lymph1 donor이며 각 조직의 sample 수도 동일 |
| 조직별 오류 및 UMI/B 층화 | completed | [donor×tissue×lineage error rate](tissue_error_analysis/errors_by_donor_tissue_lineage.tsv), [같은 lineage 조직 비교](tissue_error_analysis/errors_by_tissue_lineage.tsv), [같은 환자 조직 쌍](tissue_error_analysis/within_donor_tissue_comparisons.tsv), [UMI 층화](tissue_error_analysis/errors_by_donor_tissue_lineage_umi_quartile.tsv), [B-cell](tissue_error_analysis/B_errors_by_donor_tissue_site_umi_quartile.tsv). Site·UMI의 인과 효과는 주장하지 않음 |
| FIX-02 선택적 KG hard gate | fixed | [변경 전후 의존성](external_release_dependencies.json). KG12900은 실제4fold 완료됐지만 후보 선택에 불참하므로 필수 조건에서 제거. 원 freeze/query/8bundle·기존 예측은 유지 |
| FIX-03 확인실험 | completed | [48실행 provenance](factorial_confirmation/provenance.json), [4효과와 donor CI](factorial_confirmation/paired_effects.tsv), seed/donor/class·비용 표. Partial은 개발에서 선택된 후보. 개발 model gain CI는0 포함, 외부 model-only CI는 양수이며 일반적 개선으로 확대하지 않음 |
| Native common 고정 비교 | completed | [191target/1977primary-gene 비교](response_evaluation_native_common/condition_macro_metrics.tsv), [paired 비교](response_evaluation_native_common/paired_model_comparisons.tsv). Pretrained/random/GEARS 모두 실행. 전체parent panel·소규모 update budget 구분 |
| FIX-04 외부 범위 설명 | fixed | [HCC 환자 알려진5비악성 계열 전이](external_final.md). 전체50023, known27212/54.40%, Unknown22811/45.60%. Fixed-six와 fixed-five 유지; malignant/Hepatocyte 정답을 만들지 않음 |
| 외부 frozen batch 최종 추론·평가 | completed | 12919가 같은 freeze로8후보 추론·단일 평가를 완료. [원 batch 검증](external_final/completion_verification.json)은 fit/bundle/input/prediction/probability hash와 지표·CI를 대조. 원12892 실패·복구 이력 보존 |
| FIX-05 control sampling | completed | [full/sample B00·GO/ridge와 native 비교](control_sensitivity.md),20seed Monte Carlo 감도, 원 AAR2 38cells의 saved native2모델 입력 점검. 새 학습 없음, 기존191target 결과를 대체하지 않음 |
| 선택·보고 지표 및 Top20 용어 | fixed | Selection=validation mean intervention MSE, report=mean intervention RMSE. 최적 후보가 항상 같지 않으며 test RMSE 재선택 없음. Top20은 absolute observed mean Delta의 evaluator-only ranking이며 FDR DEG가 아님 |
| FIX-06 공식 MS 제한 점검 | completed | [실제 metric 재계산](official_scgpt_gap_audit/audit.json), [최적화·checkpoint 선택·metric call source 비교](official_scgpt_gap_audit/source_comparison.json), [항목별 표](official_reproduction.md). 추가 학습 없음 |
| MS F1 차이의 원인별 기여 분리 | deferred_with_reason | .67558 대 원 notebook .732 차이는 남음. Backend/loader/runtime 차이의 기여가 미규명이며 허용오차·seed 탐색으로 수치 성공을 만들지 않음 |
| scFoundation 추가 실행 | blocked | 기존 [공식2경로403 및 Figshare weights 없음](scfoundation_access.json). 반복 다운로드나 검증되지 않은 대체weights 사용 없음 |
| 기존 tutorial·Git | completed | 기존06–10 재사용. 조직/감도 예제06/07을 확인했고 외부 결과를 연결한06/08/09도 fresh-kernel에서 확인. [실행 hash](tutorial_execution.json), [보존·실제 근거 점검](v3_followup_audit.json). 새 notebook 개수·benchmark·전체 재학습 추가 없음 |

## 결과를 읽는 여섯 구획

1. **공식 예제:** MS modified reproduction은 실행됐지만 수치 불일치. GEARS Norman native 실행과 condition-macro 재평가를 구분한다.
2. **HCC 분류:** 동일12000-label inductive PCA/scGPT/Geneformer/CLcancer/scVI와 외부 reference·비라벨 query-adaptive LiverCT를 별도 표로 둔다.
3. **Data/model 2×2:** 동일4096 budget,4fold,3seed,48회. Data−.00392, Model+.00145, Combined−.00869, Interaction−.00621이며 앞3개 CI는0 포함이다.
4. **Genetic response:** Norman K562, HepG2 parent214target/2000genes, native-common191target/1977primary/3303shared를 각각 분리한다. Target당4treated training-cell budget은 parent gene panel을 이미 사용한 조건이며4cell-only acquisition 검증이 아니다.
5. **Chemical response:** Mouse TCDD 같은 노출의 celltype transfer이며 biological animal replicate·사람 약효·unseen compound/dose는 미검증이다.
6. **외부 final:** 사전 고정8후보 batch를 완료했다. Primary Geneformer fixed-six/five .81707/.98048, 고정 비교군 scGPT .82503/.99004다. Ground-truth coverage와 prediction coverage를 구분하며, fixed-six 개발 점수와 외부 fixed-six를 그대로 빼 일반화 성능 하락으로 해석하지 않는다.

정답이 신뢰할 수 있는 malignant/nonmalignant hepatocyte와 세부 HCC 상태 데이터는 후속 요구사항이다. 이를 이유로 현재 cohort를 바꾸거나 새 다운로드를 시작하지 않았다. 다음 학습 라운드는 같은 control estimator·생물학적 반복 holdout을 갖춘 단일 가설로 제한할 것을 제안하며 이번 수정에서 수행하지 않았다. 원시 count·weights·cell별 전체 예측은 서버 자산 영역에 남고, 사용자 제공 원본 PDF와 검토 문서 원본을 public Git에 추가하지 않았다.
