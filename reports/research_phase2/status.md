# 연구 확장 V2와 V3 실행·집계 상태

기존 main 종료 commit243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79의 환경·결과를 보존하며 research/phase2-v2에서 진행한다. 기존 scGPT/Geneformer 환경은 변경하지 않았다. 새 research와 legacy-compatible LiverCT 환경은 별도로 설치했다.

| 요구 단계 | 실제 현재 증거 | 상태 |
|---|---|---|
| 고정 protocol | 개발8명, outer4fold, legacy02/09 제외; external GSE151530 예약 | 실행됨 |
| scGPT 공식 예제 | MS7844/13468, 10epochs batch32; F1 .6756 vs 원 노트북 .732 | 수정 재현 실행, 수치 차이 있음 |
| 기본 분류 | 전체 개발63036 세포; PCA/scGPT/Geneformer/scVI4fold | 실행됨 |
| scFoundation | 공식2개 SharePoint403, Figshare 예제에는 가중치 없음 | 구체적 차단 기록 |
| 암 특화 | CLcancer4fold; base .9471 vs CL .9458 | 실행됨; 개발 이득 없음, 외부에서는 CLcancer 이득 |
| 간 특화 | LiverCT 및 동일 적응 SCANVI head4fold; 별도 query-adaptive 표 | 실행됨 |
| 개발 오류/가설 | donor 변동·rare B 오류; hypotheses_v1.json | 사전 등록됨 |
| 데이터만 변경 | donor-balanced와 donor×class-balanced, 같은4096·head·validation/evaluation IDs | 첫 fold 실행, 둘 다 음성 |
| 모델 변경 | 같은 NN head 동결·partial·seed 수정 LoRA 첫 fold 완료; validation으로 partial 선택 | screening 실행됨 |
| LoRA 연결 | 22개 query/value 모듈, gradient/update, 초기 출력·저장복원 차이0 | 실행됨, 효능 검증 아님 |
| 전 fold·3 seed 및 2×2 | 동결24+적응24조건 모두 완료; Data−.00392, Model+.00145, Combined−.00869 | 실행됨, 확실한 개선 없음 |
| GEARS 공식 | Norman simulation seed1, hidden64, 20epochs 완료; native Pearson_DE .8665 | 수정 재현 실행됨 |
| 간 유래 실측 response | HepG2 CRISPRi heldout214targets/13994세포; B00–B03 완료, GEARS RMSE .13091 vs GO/ridge .10584 | 실행됨, GEARS 추가 이득 없음 |
| foundation response 확장 | 공통191target/primary1977gene: pretrained .12394, random .12268, GO/ridge .11073, GEARS .13732 RMSE | 실행됨, 사전학습 추가 이득 없음 |
| chemical response | Mouse TCDD portal-treated1074cells holdout; scVIDR RMSE .07629 vs no-change .20569, native arithmetic .13854 | 실제 수정 native 경로 실행됨 |
| KG 압축 가설 | 사전 고정 real32/random32/raw128 대조4fold·seed42, full63036 평가 | 실행됨, 압축 이득 없음; 외부 선택에서 제외 |
| 외부 최종 | GSE151530 25명50023cells, 후보8개와 모든 bundle hash 동결; categorical reader 동등성 확인 후 동일 batch 복구12919 | 8후보 추론·단일 평가 및 원 artifact 검증 완료 |
| 사용자 tutorial | 06–10 실제 실행; PCA query2000/2000정합·probability roundtrip차이0, 모든 notebook hash 일치 | 실행됨 |

학습량4096의 개선 실험은 A-series의12000과 별도 조건이다. 같은4096 baseline을 직접 실행했고 기존 A-series 점수를 재사용하지 않는다. 첫 데이터 변경은 모두 validation과 개발 outer score에서 감소했다. 선택된 donor-balanced는 후속 대조를 위한 후보이며 확인된 개선이라 부르지 않는다.

HCC 세포 분류, Geneformer embedding shift, 실측 genetic response, chemical response는 다른 task다. 외부 final은 모델/데이터 선택을 동결하기 전까지 어떠한 모델 score도 생성하지 않는다. 모델별 pretraining overlap은 확인 불가 시 unknown으로 기록한다. 요청한 실험·집계와 외부 최종 평가를 마쳤다. scFoundation 가중치 접근 차단과 공식 MS 수치 차이는 명시적 제한으로 남으며, 모든 모델 실행 성공이나 HCC 치료효과 검증을 주장하지 않는다.


현재 Slurm handle은 `slurm_jobs.tsv`에 기록한다. Confirmation12844, final refit12883, native response12876 및 평가12896은 완료됐다. External12892는 freeze 후 첫 prediction 전 gene_symbol categorical 자료형 처리에서 실패했다. 실제 개발64-cell 확률 동등성 검증12918 후, 모델/데이터 선택과 freeze를 유지한12919가 같은 batch를 복구해8개 추론과 최종 평가를 끝냈다. evaluation.json·복구 완료 hash·terminal log와 별도 read-only verification이 일치한다. 현재 squeue에는 이 프로젝트의 실행 중 작업이 없다. 오래된 scheduler handle 만료를 실패로 보고 재제출하지 않았다.

외부 primary Geneformer fixed-six/five F1 .817066/.980479, 고정 scGPT .825035/.990042다. Known truth27212/50023(54.40%), Unknown22811이며 Hepatocyte 정답은 없다. [전체 결과](external_final.md), [최종 점검](completion_audit.json)을 읽는다.
