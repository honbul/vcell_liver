# 연구 확장 V2 진행 상태

기존 main 종료 commit243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79의 환경·결과를 보존하며 research/phase2-v2에서 진행한다. 기존 scGPT/Geneformer 환경은 변경하지 않았다. 새 research와 legacy-compatible LiverCT 환경은 별도로 설치했다.

| 요구 단계 | 실제 현재 증거 | 상태 |
|---|---|---|
| 고정 protocol | 개발8명, outer4fold, legacy02/09 제외; external GSE151530 예약 | 실행됨 |
| scGPT 공식 예제 | MS7844/13468, 10epochs batch32; F1 .6756 vs 원 노트북 .732 | 수정 재현 실행, 수치 차이 있음 |
| 기본 분류 | 전체 개발63036 세포; PCA/scGPT/Geneformer/scVI4fold | 실행됨 |
| scFoundation | 공식2개 SharePoint403, Figshare 예제에는 가중치 없음 | 구체적 차단 기록 |
| 암 특화 | CLcancer4fold; base .9471 vs CL .9458 | 실행됨, 추가 이득 없음 |
| 간 특화 | LiverCT 및 동일 적응 SCANVI head4fold; 별도 query-adaptive 표 | 실행됨 |
| 개발 오류/가설 | donor 변동·rare B 오류; hypotheses_v1.json | 사전 등록됨 |
| 데이터만 변경 | donor-balanced와 donor×class-balanced, 같은4096·head·validation/evaluation IDs | 첫 fold 실행, 둘 다 음성 |
| 모델 변경 | 같은 NN head 동결·partial·seed 수정 LoRA 첫 fold 완료; validation으로 partial 선택 | screening 실행됨 |
| LoRA 연결 | 22개 query/value 모듈, gradient/update, 초기 출력·저장복원 차이0 | 실행됨, 효능 검증 아님 |
| 전 fold·3 seed 및 2×2 | 동결24조건 완료, D1 평균 차이−.00392; partial Slurm array12844 | 적응 모델 반복 진행 중 |
| GEARS 공식 | Norman simulation seed1, hidden64, 20epochs 완료; native Pearson_DE .8665 | 수정 재현 실행됨 |
| 간 유래 실측 response | HepG2 CRISPRi heldout214targets/13994세포; B00–B03 완료, GEARS RMSE .13091 vs GO/ridge .10584 | 실행됨, GEARS 추가 이득 없음 |
| foundation response 확장 | 같은3303관측gene/191testtarget, native scGPT gradient smoke 통과; pretrained/random/GEARS job12876 | 실제 비교 진행 중 |
| chemical response | Mouse TCDD portal-treated1074cells holdout; scVIDR RMSE .07629 vs no-change .20569, native arithmetic .13854 | 실제 수정 native 경로 실행됨 |
| 외부 최종 | GSE151530 unscored; histology·환자·ontology 고정과 선택 freeze 필요 | 미완료, 점수 미관찰 |
| 사용자 tutorial | 06–10 실제 실행; PCA query2000/2000정합·probability roundtrip차이0, 모든 notebook hash 일치 | 실행됨 |

학습량4096의 개선 실험은 A-series의12000과 별도 조건이다. 같은4096 baseline을 직접 실행했고 기존 A-series 점수를 재사용하지 않는다. 첫 데이터 변경은 모두 validation과 개발 outer score에서 감소했다. 선택된 donor-balanced는 후속 대조를 위한 후보이며 확인된 개선이라 부르지 않는다.

HCC 세포 분류, Geneformer embedding shift, 실측 genetic response, chemical response는 다른 task다. 외부 final은 모델/데이터 선택을 동결하기 전까지 어떠한 모델 score도 생성하지 않는다. 모델별 pretraining overlap은 확인 불가 시 unknown으로 기록한다. 위 미완료 항목이 남아 있으므로 전체 연구 완료를 주장하지 않는다.


현재 Slurm handle은 `slurm_jobs.tsv`에 기록한다. 12844 confirmation 뒤12883 final refit, 그 뒤12892 external release/evaluation 순서다. 12876 native response 비교 뒤12896이 공통 metric과 tutorial07을 갱신한다. Completed artifact와 terminal log를 확인하기 전 같은 run을 재시작하지 않는다. External cohort는 아직 unscored이며 전체 목표는 진행 중이다.
