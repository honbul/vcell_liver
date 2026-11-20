# R11 화학 표현 비교 실행 명세

기준 코드 `research/r6-sensitivity-correction-v3`의 `1b47c0ce85c82d5d8227432b9a234a2814d22fe2`. 2026-11-20 개정안 E0–E4를 `research/r11-chemical-probe`에서 수행한다. 아래 운영값은 모델의 개발 비교 전에 지정했다.

## 과제와 자산

- Tahoe-100M 공개 미러 `2dc57900b7981cfcf5e211527169a0b006546a95`, 0.5 µM, 24시간. `sample`은 물리적 웰이다. `BARCODE_SUB_LIB_ID`는 세포 ID다. 원자료의 첫 토큰은 gene 1 / expression −2로 확인했고, 두 배열의 선두 항을 함께 제외한다. 실제 count는 비음수 정수만 사용한다. [Tahoe 원 논문](https://doi.org/10.1016/j.cell.2026.08.035) · [고정 버전 데이터 설명](https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/2dc57900b7981cfcf5e211527169a0b006546a95/README.md)
- 원본 번호 1–14의 `plate13`은 의심 보고에 따라 사전에 격리한다. 기록 상태는 `external_report_unverified_detail`이다. 나머지 주 용량 plate2·plate5·plate8·plate11을 보존한다.
- R6의 train·3개 source fold·test 화합물군을 유지한다. historical validation은 R6의 미사용 정책을 유지한다. fragment parent·중성화·비입체 연결성, 원래 SMILES와 PubChem ID를 확인하고 같은 화합물군의 모든 용량을 fit 경계에 함께 둔다. 모델 입력에는 원 entity별 canonical stereo SMILES를 사용한다. 같은 군의 반응을 하나의 정답으로 합치지 않는다.
- 화합물군당 entity, entity당 세포주, 세포주당 반복 웰에 균등 가중한다. pool 평균 중심화에는 같은 pool 안 group→entity→well 순으로 균등 가중한다. 적격성은 기술 QC와 웰×세포주당 유효 세포 20개, 물리적 control 가용성, pool당 비교 그룹 3개 이상으로 판정한다. 반응 크기·일치도는 적격성에 사용하지 않는다.
- 전체 매핑 유전자 UMI를 합산한 뒤 `log1p(10000 × gene_count / mapped_library)`를 적용한다. 각 개발 fold의 학습분과 final 재적합의 전체 source에서 처리군 well×line 분산 순으로 2,000개를 따로 고른다. 유전자 토큰 번호로 동률을 푼다. role 간 공유하는 gene-panel 선택에는 control을 입력하지 않는다.
- 각 plate의 두 DMSO 웰을 sample ID 순서로 A/B 지정한다. R-A는 A만 읽어 기저 feature·fit delta·개발 선택에 사용한다. R-B는 B만 읽어 독립 실행한다. final 측정 반응 및 반대 role 웰은 두 역할의 예측이 존재할 때 채점 코드가 읽는다. 두 role 공통 적격 행에서 seed 후 role별 점수를 평균한다.
- 원본 식별자, 제외·층화, 화학적 유사도는 메타데이터와 영수증에 기록한다. source와 final의 count 산출물을 분리한다. 원자료와 대용량 산출물은 G1에 보관한다.

## 비교

- E와 C는 각각 ECFP4(반경 2, 2,048비트, chirality off)와 고정된 ChemBERTa-77M-MTR의 마지막 CLS 384차원을 사용한다. E와 C 모두 고유 train entity·물리적 control×line에서 별도 PCA(최대 32/16)와 latent 표준화를 적합한다. 외적과 주효과 설계를 만든 후 train 열을 표준화하고 상수열을 뺀다. Ridge 5개 후보와 절편 무벌점, 학습 행수로 정규화한 군 가중치를 양 arm에 동일 적용한다. [고정 후보 모델](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR/tree/66b895cab8adebea0cb59a8effa66b2020f204ca)
- B0은 no-change, B1은 train cell-line 평균 및 축소 pool 평균, B2는 전체 ECFP4와 DMSO 평균을 입력으로 한 강한 선형 Ridge와 mean-state MLP 재적합이다. E/C/B2의 공통 residual offset은 축소 계수 2를 사전 고정한다. fit의 offset은 해당 화합물군 전체를 제외하며, query offset은 train 응답만 쓴다. pool→line→global fallback을 기록한다.
- 세 fold의 주 지표로 role별 B*와 설정을 선택한다. 동률인 B1 축소 후보는 전체 delta RMSE로 정하고, B* 동률은 저비용·단순 arm을 우선한다. B2 MLP는 두 학습률과 개발 seed 42를 사용한다. 최종 신경망은 선택 설정의 개발 best epoch 중앙값과 seed 42·43·44, CPU deterministic Torch를 사용한다.
- 약물 정보 제거 진단은 개발 fold의 plate별 동일 entity 배치 패턴을 가진 화합물군을 통째로 순환 치환한다. 같은 매핑을 모든 세포주에 쓰고 교환 불가능한 stratum도 기록한다.
- E3 열림 조건을 role별로 사전 고정한다. C가 각 개발 fold에서 E·B*·치환 C보다 낮아야 한다. 그리고 같은 entity가 두 plate에 있는 화합물군의 HepG2/C3A−SNU-423 OOF residual 차이가, 같은 plate 쌍의 다른 화합물군 199회 치환 평균보다 높은 fold가 최소 둘이어야 한다. 반복군이 3개 미만인 plate 쌍으로는 진입하지 않는다. 이는 추가 구조 연구의 운영 규칙이다. 진입 시 두 학습률의 F/M을 동일 조건으로 평가하고 유사 파라미터 수(±10%)와 실제 업데이트를 기록한다. 한 role만 열리면 구조 결과는 해당 role 탐색으로 남긴다.

## 점수와 종료

- 주 지표는 동일 pool에서 관측·예측을 각각 중심화한 약물 대비 RMSE다. 보조는 전체 delta RMSE, 별도 절대 발현 RMSE, pool 내부 entity retrieval, plate·세포주별 요약이다. Pearson/cosine의 0분산·0norm은 사유와 함께 NA로 기록한다. 공통 조건의 no-change가 retrieval에서 전 후보 동률일 때 기대 순위와 기대 top-1을 쓴다.
- E2 공식 대비는 role별 C−B*, 귀속 대비는 C−E다. E3 양 role 진입 시 F−B†가 두 번째 공식 대비다. 2,000회 Exp(1) 양수 가중에 같은 group·role·seed·arm의 행을 묶는다. 매 가중 반복마다 동일 pool에서 약물 평균을 새로 뺀다. 주 공식 대비 1개면 95%, 두 대비면 각 97.5% 근사 백분위 구간을 사용한다. Plate별 및 leave-one-plate-out 재집계와 두 간계열 슬라이스를 함께 남긴다.
- 개발 결과와 선택·코드·자료 hash를 동결한 뒤 final 처리군 count를 처음 연다. 평가용 유전자 패널·효과 판정 임계값을 final에서 선택하지 않는다. 노출 상태는 `project_seen_source / fit-held-out / protocol-frozen follow-up`이다. 실용적 크기 문턱은 미설정이다.
- CPU 작업 4개, 작업당 8 thread, GPU 2개 및 48 GPU-hour, 다운로드 20 GiB 및 신규 scratch 60 GiB 이내. G1 기존 Slurm을 사용하며 원본과 기존 결과는 별도 경로에 유지한다. 새 표현, R7, 환자 약물 순위와 DTP 실험은 계획의 기본 실행 범위 밖이다.

## 파일럿 후 사전 고정한 보조 운영값

전체 개발 모델 실행 전에 fold별 적격 화합물군 수로 개발 점수를 가중하도록 확정했다. 각 그룹은 하나의 held fold에만 등장한다. 기능 파일럿은 초기 동일 fold 가중값으로 기록을 보존한다. 또한 metadata에서 진입 가능성을 확인한 scaffold 보조 검사를 지정했다. 고정 모델로 source에 없는 비어 있지 않은 Bemis–Murcko scaffold의 test group만 채점하되, 해당 group의 모든 entity가 조건을 충족해야 한다. 적격 pool에서 평균을 다시 계산한다. 모델 재적합을 추가하지 않으며 사전 지정된 공식 대비는 동일하다.
