# R7 — 자료·가중치 노출 기록 (EXPOSURE)

## 1. 이 자료를 전에 본 적이 있는가

| 자료 | 이전 노출 | R7에서의 사용 |
|---|---|---|
| CMap 2020 Level 3 10 µM·24 h, 978 landmark | 연구 B·R2·R4에서 처리·학습·채점 | R4 unit 파일을 읽기 전용으로 재사용 |
| R4 약물 그룹 분할(seed 20261117) | R4가 만들었고, **R4 test는 job 25374에서 한 번 열림**(W2·W1·MLP·Ridge 점수 공개) | 분할은 그대로다. R7의 test 평가는 노출된 공개 benchmark의 사전 등록 후속 평가이며, blind test라고 부르지 않는다 |
| R4 validation 점수 | R4 개발 중 사용 | D-0(backbone 정의)의 근거로 R4 validation 기록만 인용했다. R4 test 수치는 R7 설계 선택에 쓰지 않았다(설계자는 그 수치를 이미 알고 있었다) |
| HCC 3계열 | B·R2·R4에서 채점됨 | 동결 전에는 HCC 점수를 계산하지 않는다. validation 점수는 source panel만이고, HCC validation unit의 예측은 저장만 하고 채점하지 않는다 |

## 2. R7 안에서의 노출 순서

| 자료 | 읽은 job | 시점 |
|---|---|---|
| training 약물 Δ(모든 세포주) | 기준선, D1 진단, 모든 학습 | protocol commit(`b76ae64`) 이후 |
| validation 약물 Δ | checkpoint·LR 선택, validation 신호, 수정 판단 | 개발 중 |
| 첫 pilot의 이른 validation 줄 | 취소된 job 25732–25746 로그(D-2 정정) | KG 보정 결정 뒤 두 줄만 읽음. 선택에 쓰지 않음 |
| test 약물 Δ | `final_eval` 한 번(`R7_FINAL_EVAL=1`, 동결 manifest commit, clean tree 필요) | 동결 이후 |

- 지식 그래프·표적 주석은 metadata만으로 만들었다(`kg_build`가 RNA 파일을 읽지 않음은 `leak_fixtures.json`에서 확인).
- validation·test 약물의 **표적 주석**은 예측 입력으로 쓴다. 기획서는 이를 외부 정보로 허용하며, 출처와 정보 비용을 `inference_information_contract.json`에 적었다.
- 예측 함수가 처리 후 RNA를 읽지 않음을 fixture로 확인했다. delta 배열을 noise로 바꿔도 예측 차이는 GPU 재실행 잡음 수준(≤7e-7)이었다.

## 3. 외부 코드·가중치

| 대상 | 사용 | 노출 위험 |
|---|---|---|
| XPert 코드(GSanShui/XPert `d53ff497`) | 원 학습 스크립트를 수정하지 않고 R4 split으로 재학습 | 학습 h5ad에는 train·validation unit만 있고 test unit은 없다 |
| XPert 저자 학습 가중치 | **쓰지 않음** | L1000 전체로 학습되어 R7 test 약물이 들어 있을 수 있음 |
| XPert HG drug embedding(Zenodo, 8981 약물) | 약물 입력 feature로 사용 | 사전학습 그래프(PPI·DTI·DDS)에 반응 유래 관계가 들어 있는지 저장소에서 확인할 수 없다(DDS 정의 미기록). **미확인**으로 남기며, XPert 결과 해석 시 이 점을 함께 적는다 |
| UniMol 사전학습 가중치(Hugging Face `dptech/Uni-Mol-Models`, 공개) | XPert 색인에 없는 1,199개 그룹의 UniMol feature 생성 | 분자 구조 사전학습. RNA 반응은 쓰지 않음 |
| ChEMBL 37, OmniPath, HGNC | 지식 그래프 | RNA 반응 정보 없음. 단 ChEMBL activity 일부가 세포 기반 assay일 수는 있다(transcriptomic 반응 자체는 아님) |
