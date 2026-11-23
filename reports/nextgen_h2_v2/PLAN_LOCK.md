# H2-v2 연구 및 평가 잠금

H2-v1 커밋 e90c8e2cf684b6c332e34a678239f8770cd485a7의 자료·분할을 재사용한다. 새 worktree는 `nextgen_h2_v2`, Slurm은 `ng2-h2-*`, 전용 G1 경로는 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h2_v2`다. GPU 배열은 순차 연결하며 동시 작업 최대 3개, 일감당 GPU 1개를 할당한다.

## 데이터와 분할

- v1의 1,186 exact parents, 1,001 Murcko scaffold 연결 성분, CTRPv2·GDSC2·PRISM20Q2 670,830행 및 원래의 세 outer와 3×3 nested inner를 그대로 채점한다. 분할 SHA-256은 `98873d0954444add1719c65d3e478ff6b159af8f7a72f55eb063f80a4daf3800`이다. GDSC1 명부 SHA-256은 `7346d53fdeea31e867444b3bf2206a8c87f321b247209b1d6ca980993e739768`이다. LIMORE 성인 50계통·68약물과 LICOB 성인 38공여자·76약물, 소스 미포함 완전 구조 18약물은 이전 결과 노출이 있는 기술 자료다. [v1 데이터 및 노출](../nextgen_h2_multiview/DATA_SUPPORT_AND_EXPOSURE.md)
- LINCS v1의 비간계열 서명 82,740개 명부에서 978 landmark, 10,174 BING, 12,328 전체 유전자를 추출한다. 후보는 24시간 MODZ 평균, 24시간 평균+세포주 SD/IQR, 6/24시간 분리, 24시간 중/고농도 분리다. 각 약물이 측정된 참조 세포 내에서 농도나 시간을 결합하고 참조 세포를 균등 가중한다. L1000 관측 약물의 핵심 분모는 v1 24시간 명부의 724개다. [공개 L1000 정의](https://doi.org/10.1016/j.cell.2017.10.049)
- Tahoe raw Parquet 3,388개에 공개 구조 레지스트리와 정확 부모 키를 사용한다. 허용 농도는 0.05/0.5/5 µM, 미지정·간계열 및 예약 HCC 세포는 제외한다. 바코드 해시 6% 고정 샘플을 취해 최소 유전자 200개, UMI 500–100,000을 적용하고 표본·계통당 16개 이상에서 같은 plate·계통 DMSO 대비를 계산한다. Pseudobulk 평균, 세포 내 SD/세포주 SD 및 농도 기울기를 별도 후보로 둔다. [공개 Tahoe 데이터](https://huggingface.co/datasets/tahoebio/Tahoe-100M)
- DepMap Public 25Q3에서 정확 ModelID와 default model/model-condition을 사용한다. 체세포 hotspot, 유전자 CNV, 융합, 전역 시그니처는 각각 마스크로 분리한다. 융합 파일에 default sequencing ID가 확인된 행만 관측으로 취급한다. Chronos 실측은 독립 참고 arm이며 최종 멀티오믹스 후보에서 제외한다. LIMORE/LICOB에는 DepMap 실측을 붙이지 않는다. [DepMap 데이터 설명](https://depmap.org/portal/data_page/)

## 선택과 채점

- 세 outer-fit 각각에서 inner fold를 모아 평가한다. C1과 H2-v1은 동일 행과 이전 선택값을 참조한다. 가중치, 벌점, 정규화, 표현과 절편은 inner에서 결정하고, 관측 LINCS 또는 공통 측정 지원의 within-drug Spearman을 우선한다. 후보 우승값이 양의 탐색 경계에 닿으면 경계를 최소 두 배 확장한다. 마스크 가중치 0은 PSD의 실제 경계이며 음수를 탐색할 수 없다.
- 추가 뷰는 Round 1을 고정하고 독립 가중치로 각각 시험한다. 관측 조건에서 같은 지지의 C2(마스크)·C3(핏/질의 내부 별도 값 순열)를 비교한다. 전체 행 및 표적 항암제와 암종별 영향을 기록하고, 실측 생물학 뷰 가용성만으로 선택하지 않는다. 최종 구조 전용 후보는 별도 462개 비관측 지원에서 C1과 비교한다. 구조 후보를 적합할 때 각 훈련 약물의 가상 서명은 그 약물의 골격을 보지 않은 내부 교차 적합에서 생성한다.
- 두 비선형 패밀리는 inner 교차적합 kernel OOF 예측의 잔차를 적합한다. 3/3 outer-fit에서 일관되게 개선할 때만 승격한다. 선택에 쓰지 않은 원래 outer 및 외부 결과는 모든 후보 예측과 최종 정책을 봉인한 뒤 단일 절차에서 읽는다.
- H2-v1 대비 within-drug ρ ≥.01, 이중 중심화 ρ ≥.02, NDCG@5 ≥.01 중 하나의 짝 95% 하한이 양수여야 한다. 다른 지표와 신규 GDSC1의 비열등 하한은 −.005다. 462개 구조 전용 gate는 C1 대비 within-drug ρ ≥.01 및 짝 95% 하한 양수다. 원래 스크린별 fold 내 약물 균등 macro와 2,000회 골격/세포계통 짝 구간을 재사용한다.
