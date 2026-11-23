# H2 개발 검증 잠금

기준 커밋 `5ec8ed231ee88c6ae578d0e1cb30d7591c792392`. 작업 경로는 `nextgen_h2_multiview`, 산출물 경로는 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_h123/h2_multiview`, Slurm 이름은 `ng-h2-*`다. 한 시점에 A30/A40 최대 2장을 사용한다.

## 데이터와 평가 경계

- `licob-cold-5bf53fc/inputs`의 1,186 exact parent, 670,830관측, 기저 RNA 14,421유전자를 사용한다. 등록 영수증 SHA-256 `67c11aba490282e30be9ef6d3d4d28ad192a2f5620a9e0b84c978019a0ccdb9e`. 표적은 ChEMBL 37의 exact parent에 연결된 사람 단일 단백질에 한정한다. 물리·화학 기저는 ECFP4 2,048비트와 분자량, LogP, TPSA, HBD다.
- Cell Painting cpg0004 등록 교집합 411 exact parent / 362 대표 골격이다. 기존 A549 48 h dose recode 6, 두 배치의 DMSO 기준 중앙값 캐시를 사용한다. LINCS 2020의 구조 확인 교집합은 744 / 637이며, 알려진 비간계열 Level 5 보유군은 742 / 635다. 0.5–20 µM, 6/24 h window MODZ는 734 / 627이다. 두 뷰 교집합은 이 LINCS 정의에서 378 / 332다. 데이터 영수증 SHA-256 `60602ebba5ecbc0e54e92e0a761f5e87fe3719760a5a1a2d086c4946b6b7a5ca`.
- Exact parent, connectivity, 모든 등록 구조 표현의 Murcko scaffold를 연결 성분으로 묶었다. 비환상 약물은 등록 규칙의 connectivity별 그룹이다. 전체 패널을 평가할 수 있도록 약물 축의 GroupKFold 3분할과 각 outer-fit의 nested GroupKFold 3분할을 사용한다. Outer test 약물의 반응 열은 별도 scorer 전까지 해당 fold에서 읽지 않는다. 세포는 약물 블라인드 질문에 맞춰 공유하고, 입력 RNA와 학습용 반응 행은 모든 arm에서 동일하다. 기존 joint 분할과의 차이는 재사용 문서에 기록한다. [분할과 뷰 지원](STEP0_SUPPORT.tsv)
- GDSC1 8.5에서 등록 학습의 exact parent, connectivity 또는 모든 골격 표현과 겹치거나 이전 브리지의 등록 약물/ID에 포함된 후보를 제외했다. 결과 66약물, 66골격, 713개 공유 소스 세포, 결과값을 읽기 전 39,545쌍을 별도 보류했다. 동일 세포 및 공개 스크린 이력이 있어 새 약물·스크린 기술 검증으로 해석한다. LIMORE 및 LICOB는 이전 노출 자료이며 LICOB18은 fully specified source-absent 집합이다. [GDSC 8.5](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)

## 적합과 판정

- C1의 구조 커널은 동일한 corrected RNA·pathway·TF·추정 Chronos/driver 기반을 사용한다. 소스별 감수성 방향, fit-only 양의 affine 보정, 소스×약물 내 반응 중심화와 관측 가중을 이전 구현에서 재사용한다. C1의 분자 또는 이중 중심 상호작용 커널과 벌점 4/16은 각 outer-fit의 inner 주 지표로 먼저 선택한다. C0은 같은 설정에서 표적을 제거한다.
- Round 1에서 두 뷰를 독립 비교한다. Painting의 z-score PCA, PCA whitening, truncated ZCA sphering, 분산/중복 필터에서 32/64/128/256과 cosine/RBF, 그리고 원시 표준화 대조를 비교한다. LINCS는 24 h 단순 평균·24 h MODZ·6/24 h MODZ·strict QC 최소 1/2 세포주를 978 landmark, 44개 경로, rank-signed, PCA whitening과 교차한다. 44경로는 L1000 BING의 PROGENy 14개 및 유전자 커버리지 상위 Hallmark 30개로 정하며, 978차원 landmark 비교에는 직접 측정 유전자만 쓴다. C4−C2의 레시피 공통 관측 지원에서 표현을 선택한 뒤 레시피 관측군과 전체에서 고정 순열 C3까지 평가한다. 선택 통과는 관측 지원에서 C2 대비 Spearman 증가 ≥.02, 짝 95% 하한 >0이고, C3 대비 하한도 >0이다. 머리말의 녹색 판정은 outer에서 재검증한다.
- Round 2는 통과 뷰의 0/평균 채움 부착, PSD 합, 대각 정규화 PSD, scaffold OOF 잔차 게이팅을 가중 .10/.25/.50에서 비교한다. 테스트 약물의 관측 프로파일을 fold마다 25%/50% 고정 시드로 가리고 자연 결측과 분리한다. 결합별 가중은 자연·25·50% 전체 지원의 Spearman 평균으로 고정한다. 단순 부착 대비 자연 결측 전체 지원에서 Δρ≥.02 또는 짝 95% 하한 >0이면서 ΔNDCG@5 하한 ≥−.01이면 결측 결합 후보를 채택한다. 양쪽 개별 통과 fold에서만 고정 이중 정규화 PSD 1개를 비교한다. 둘 다 실패하면 LINCS 44경로 기하 대조만 추가 정보 확인으로 진전시킨다.
- Round 3은 독립 LINCS 또는 통과한 LINCS 결합의 RNA 경로 × 약물 변화 경로 곱을 가중 .05/.20에서 비교한다. 관측군에서 기하 대비 증가 및 짝 95% 하한 >0, 그리고 C2/C3 정보 게이트 통과를 요구한다. 기존 가용 후보를 대체할 때는 동일 지원에서 해당 후보보다 Spearman이 높아야 한다. 세 라운드 후 각 outer-fit에서 동일 선택 규칙으로 단일 정책을 고정한다. 전체 소스 적합의 외부 기술 평가 설정은 inner 선정 정책의 다수결과 고정 타이브레이크로 정한다. Outer 성적으로 재선택하지 않는다.
- 각 스크린에서 fold 내부의 약물별 Spearman/Pearson을 약물 균등 평균한다. 스크린별 fold 평균 뒤 스크린을 균등 가중한다. 유효 약물은 비상수 관측 ≥5개다. NDCG@5와 Precision@5는 한 세포 내 약물 순위로 평가하고, 관측 마스크에서 세포·약물 main effect를 뺀 상호작용 Spearman/Pearson을 보조 보고한다. 약물 대비에는 골격 블록, retrieval 대비에는 계통에 공유한 2,000회 exponential multiplier 짝 95% 구간을 쓴다. 모든 점수는 동일 지원에서 비교한다.
- 배포 및 평가에서 보조 뷰가 없는 query는 정확히 C1으로 돌아간다. Round 2는 결측 자체의 성질을 진단하도록 비게이팅 arm의 원래 출력도 채점한다. 최종 예측 파일·선택 정책·오프라인 패키지 해시를 먼저 봉인한다. 단일 score 실행에서 source outer, tissue 및 외부를 함께 채점하고 score pass를 1로 기록한다.
