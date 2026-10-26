# 연구 결과 인덱스

V1 기록은 보존했다. 여기서는 V2 실험과 V3 보완 분석을 연결한다. **외부 GSE151530 최종 추론은 진행 중이며 전체 연구 완료가 아니다.** 최신 서버 관측은 [current_status.json](external_final/current_status.json), 해석은 [연구 요약](final_summary.md)을 읽는다.

| 질문 | 실제 결과·근거 |
|---|---|
| 전체 진행 상태 | [상태](status.md), [Slurm 작업](slurm_jobs.tsv), [V3 적용 범위](../../docs/RESEARCH_AMENDMENT_V3.md), [실제 근거 및 보존 점검](v3_followup_audit.json) |
| 공식 재현과 수치 차이 | [한국어 설명](official_reproduction.md), [scGPT 제한 점검](official_scgpt_gap_audit/audit.json), [Norman 실제 사용 population](official_norman_population/audit.json) |
| 같은 정보 조건의 기본 분류 | [Inductive 비교](leaderboard_annotation_inductive.tsv), [donor 쌍대 CI](annotation_uncertainty_inductive/paired_donor_bootstrap.tsv), [실제 pooled calibration](annotation_uncertainty_inductive/pooled_metrics_and_calibration.tsv) |
| 간/암 특화 비교 | [CLcancer 포함 기본 비교](leaderboard_annotation_inductive.tsv), [query-adaptive LiverCT 별도 비교](leaderboard_annotation_query_adaptive.tsv), [실제 native voting head와 coverage](liverct_native_head.md) |
| 조직 부위별 오류 | [해석](development_insights.md), [donor×tissue_site×lineage 오류](tissue_error_analysis/donor_tissue_lineage.tsv), [support](tissue_error_analysis/population_support.tsv), [pooled site 표](tissue_error_analysis/pooled_by_tissue.tsv), [원본 site 정합 증거](tissue_error_analysis/provenance.json) |
| 데이터·모델 변경의 기여 | [48조건 평균±SD](factorial_confirmation/pooled_mean_std.tsv), [Data/Model/Combined/Interaction](factorial_confirmation/paired_effects.tsv), [class별](factorial_confirmation/per_class_mean_std.tsv), [실제 비용](factorial_confirmation/resources_by_factor.tsv), [선택 가설](hypothesis_registry.tsv) |
| KG 적용 방식과 음성 결과 | [계산식](kg_feature_method.md), [fixed32 압축 대조](kg_compression_diagnostic/pooled_metrics.tsv), [donor 차이](kg_compression_diagnostic/paired_donor_effects.tsv) |
| 실측 genetic response | [task별 비교](leaderboard_genetic_response.tsv), [full HepG2](response_evaluation_hepg2/condition_macro_metrics.tsv), [matched native-common](response_evaluation_native_common/condition_macro_metrics.tsv), [pretrained/random 쌍대 CI](response_evaluation_native_common/paired_model_comparisons.tsv) |
| Control 평균화 감도 | [한국어 해석](control_sensitivity.md), [191target 원/변경 지표](control_sensitivity/aggregation_metrics.tsv), [20seed 변동](control_sensitivity/simple_predictor_resampling.tsv), [saved native 첫 target](control_sensitivity/native_first_target_metrics.tsv), [실제 checkpoint/replay 증거](control_sensitivity/native_first_target.json) |
| Mouse chemical response | [별도 비교](leaderboard_chemical_response.tsv), [실행 범위](data_card.md), [원 run](runs/C00_scvidr_mouse_TCDD_portal_seed42_v3/run.json) |
| 외부 HCC 다섯 비악성 lineage 전이 | [평가 범위·상태](external_final.md), [원 freeze](external_final/freeze.json), [release 의존성 수정](external_release_dependencies.json), [같은 batch 복구](external_final/attempts/external_categorical_recovery_v1/start.json) |
| 모델·데이터·환경·실패 | [모델](model_registry.tsv), [데이터](dataset_registry.tsv), [원 run registry](run_registry.tsv), [명시적 metadata 보완](resolved_run_contracts.json), [실패](runtime_failures/), [scFoundation 차단](scfoundation_access.json), [자원 범위](resource_scope.md) |
| 사용 예제 | [06 분류](../../notebooks/06_research_development_cv.ipynb), [07 실측 반응](../../notebooks/07_measured_perturbation_response.ipynb), [08 개선](../../notebooks/08_data_model_improvements.ipynb), [09 새 query](../../notebooks/09_new_hcc_query_annotation.ipynb), [10 mouse TCDD](../../notebooks/10_conditional_mouse_liver_chemical_response.ipynb), [fresh-kernel 실행 hash](tutorial_execution.json) |

외부 metric 파일은 모든 고정 후보의 추론과 검증이 끝난 뒤 생성된다. 현재 없는 점수를 추정해 채우지 않는다. 개별 run의 원시 예측·weights·count는 서버 자산 영역에 보존하며 Git에는 작은 집계와 경로/hash만 기록한다. Fixed-six/fixed-five는 외부에 서로 다른 정답을 만든 것이 아니라 사전 정의한 두 집계이며, Unknown malignant 세포의 lineage 정답을 보충하지 않는다.
