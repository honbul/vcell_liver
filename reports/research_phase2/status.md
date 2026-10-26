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
| 모델 변경 | 같은 NN head 동결 대조 완료; partial/LoRA 첫 fold 학습 | 진행 중 |
| LoRA 연결 | 22개 query/value 모듈, gradient/update, 초기 출력·저장복원 차이0 | 실행됨, 효능 검증 아님 |
| 전 fold·3 seed 및 2×2 | 초기 후보 screening 뒤 선택 조건을 반복 | 미완료 |
| GEARS 공식 | Norman simulation seed1, hidden64, 20epoch 학습 | 실행 중 |
| 간 유래 실측 response | HepG2 CRISPRi target1714/214/214, test13994세포; B00–B02 실행 | GEARS strict 추가 미완료 |
| foundation response 확장 | task/input 적합성 확인 후 B04/B05 중 가능 경로 | 미완료 |
| chemical response | 필수 genetic 경로를 지연시키지 않는 조건부 조사 | 적합성 판단 미완료 |
| 외부 최종 | GSE151530 unscored; histology·환자·ontology 고정과 선택 freeze 필요 | 미완료, 점수 미관찰 |
| 사용자 tutorial | 06 분류/공식예제, 07 실측 response 결과 읽기 | 실제 실행됨 |

학습량4096의 개선 실험은 A-series의12000과 별도 조건이다. 같은4096 baseline을 직접 실행했고 기존 A-series 점수를 재사용하지 않는다. 첫 데이터 변경은 모두 validation과 개발 outer score에서 감소했다. 선택된 donor-balanced는 후속 대조를 위한 후보이며 확인된 개선이라 부르지 않는다.

HCC 세포 분류, Geneformer embedding shift, 실측 genetic response, chemical response는 다른 task다. 외부 final은 모델/데이터 선택을 동결하기 전까지 어떠한 모델 score도 생성하지 않는다. 모델별 pretraining overlap은 확인 불가 시 unknown으로 기록한다. 위 미완료 항목이 남아 있으므로 전체 연구 완료를 주장하지 않는다.
