# 유전자 개입 뒤 RNA 반응 예측

[연구 계획](docs/REPRODUCTION_FIRST_RESEARCH_PLAN.md)에 따라 GEARS와 scGPT의 원 논문 조건을 조사하고 실제 학습·평가한다. 이후 같은 대조군과 출력 유전자를 제공하는 공통 조건에서 단순 기준선과 적응 학습을 비교한다. 이전 작업의 복구 위치는 [아카이브](archive/README.md)에 있다.

저자 자료와 실행 조건은 [GEARS 계약](protocols/reproduction/gears_norman.yaml), [scGPT 계약](protocols/reproduction/scgpt_norman.yaml), [공개 자산 영수증](manifests/public_receipt.json)에 기록했다. 원래 비공개로 보관된 split과 control 배정은 회수하지 못했다. 새 실행은 저자가 배포한 Norman 데이터를 사용하는 수정 재현이다.

[공통 데이터](docs/COMMON_NORMAN_DATA.md)는 별도로 준비했고 [분할 영수증](manifests/common_norman_v1_s26.json)을 남겼다. [연구 보고서](reports/FINAL_REPORT.md)에 논문 재현, 공통 기준선, 모델 튜닝, 샘플링 검증과 홀드아웃 결과를 정리했다. [사람 간암 자료 조사](reports/HCC_DATA_FEASIBILITY.md)에서 이번 개입 예측 조건을 충족하는 후보는 확보하지 못했다.

[실행 튜토리얼](tutorials/README.md)에서 논문 재현·튜닝·간암 자료 검토를 따라 할 수 있다. [결과 탐색 노트북](tutorials/inspect_results.ipynb)은 학습 작업을 제출하지 않는다.

[조건부 전문가 구조 약물 반응 실험](reports/moe_drug_v1/README.md)은 Tahoe-100M 0.5 µM에서 미관측 약물의 평균 RNA 반응을 예측할 때 조건부 soft-MoE가 규모를 맞춘 Dense보다 나은지 확인했다. 개선은 없었고(+0.07 %, 구간 [−0.05 %, +0.19 %]) 두 모델 모두 같은 입력의 ridge에 약 5 % 뒤져, 후보를 종료했다.

[공개 chemCPA 기반 조건부 전문가 실험](reports/chemcpa_moe_v2/RESULTS_FIXED.md)은 같은 Tahoe-100M 0.5 µM, 미관측 약물 과제를 공개 chemCPA 위에서 다시 물었다. 첫 실행의 평가자가 세포주 one-hot을 학습 loader와 다른 순서로 넣어 48개 중 43개 세포주에 다른 세포주 정보를 준 것이 확인되어, 그 수치는 [철회](reports/chemcpa_moe_v2/RESULTS.md)하고 고친 평가자로 재학습했다. 원형(P0), 규모를 맞춘 Dense 잔차(P1), 세포주 라우터 MoE(P2), 계획서의 basal 라우터 MoE(P2b)를 비교한 결과 두 MoE 변형 모두 사전 등록 승격 기준을 전부 충족하지 못해 P3는 학습하지 않았고, 시험(버그 수정 후 재평가)에서도 P1보다 나빴다(P2 −0.64 %, 구간에 0 포함; P2b −1.65 %). 네 모델 모두 무변화 예측(0.1043)과 세포주 평균·ridge(0.092)를 넘지 못했으며, 약물 중심 상관만 chemCPA 쪽이 높았다(0.25 대 ridge 0.097). 이 구성은 채택하지 않는다. [모델 비교(첫 실행, 철회)](reports/chemcpa_moe_v2/MODEL_COMPARISON.md), [데이터 카드](reports/chemcpa_moe_v2/DATA_CARD_V2.md), [PIE 외부 비교군 미실행 사유](reports/chemcpa_moe_v2/X1_PIE.md), [재현 노트북](notebooks/chemcpa_moe_v2/01_reproduce.ipynb).
