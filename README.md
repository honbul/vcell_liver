# 유전자 개입 뒤 RNA 반응 예측

[연구 계획](docs/REPRODUCTION_FIRST_RESEARCH_PLAN.md)에 따라 GEARS와 scGPT의 원 논문 조건을 조사하고 실제 학습·평가한다. 이후 같은 대조군과 출력 유전자를 제공하는 공통 조건에서 단순 기준선과 적응 학습을 비교한다. 이전 작업의 복구 위치는 [아카이브](archive/README.md)에 있다.

저자 자료와 실행 조건은 [GEARS 계약](protocols/reproduction/gears_norman.yaml), [scGPT 계약](protocols/reproduction/scgpt_norman.yaml), [공개 자산 영수증](manifests/public_receipt.json)에 기록했다. 원래 비공개로 보관된 split과 control 배정은 회수하지 못했다. 새 실행은 저자가 배포한 Norman 데이터를 사용하는 수정 재현이다.

[공통 데이터](docs/COMMON_NORMAN_DATA.md)는 별도로 준비했고 [분할 영수증](manifests/common_norman_v1_s26.json)을 남겼다. 공통 모델 학습은 두 논문 실행의 상태를 기록한 뒤 시작한다. [사람 간암 자료 조사](reports/HCC_DATA_FEASIBILITY.md)에서 이번 개입 예측 조건을 충족하는 후보는 확보하지 못했다.
