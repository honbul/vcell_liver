# 독립 HCC 파일럿의 비교군 학습 노출과 구조 입력

**96군 중 R17 A와 Tahoe T2의 기록된 반응 학습에 쓰인 동일 화합물은 0군이었다.** 두 이력의 동결·예측 영수증을 검증하고 실제 적합과 선택 명부에 canonical parent, tautomer, 연결성 키를 대조했다. R17은 커널+FiLM과 비교군 7개, T2는 5 µM의 비교군 10개를 검사했다. 같은 이름의 모델도 두 이력에서 별도로 추적했다. [사전 명세](HCC_PILOT_MODEL_SUPPORT_PROTOCOL.md) · [모델·분할별 원표](data/hcc_pilot_model_support/audit/model_exposure.tsv)

| 기존 학습 이력 | 검증한 모델·분할·역할 조합 | 분할별 반응 적합 parent | SNU-423 반응 적합 parent | HUH7 반응 적합 parent | 초안 화합물 일치 |
|---|---:|---:|---:|---:|---:|
| R17 A, HEPG2 6 h | 72 | 1,238–1,247 | 0 | 0 | 0/96 |
| Tahoe T2 5 µM, 최소 20세포 | 60 | 134–136 | 134–136 | 0 | 0/96 |
| Tahoe T2 5 µM, 최소 200세포 | 60 | 134–136 | 134–135 | 0 | 0/96 |

R17의 내부 선택 명부는 분할당 244–275군이며 초안과 일치가 없었다. T2의 내부 선택은 같은 outer-fit 명부의 부분집합이다. T2에는 SNU-423의 다른 화합물 반응이 들어갔다. 이 이력의 SNU-423 평가는 기존 세포주에서의 새 화합물 검증에 해당한다. HUH7은 이 두 반응 적합 명부에서 빠져 있으므로 별도 맥락 전이로 기록한다. [SNU-423 등록번호](https://www.cellosaurus.org/CVCL_0366) · [Huh-7 등록번호](https://www.cellosaurus.org/CVCL_0336)

| 비교군 자산 | 확인 범위와 평가 조건 |
|---|---|
| XPert, chemCPA, TranSiGen, PRnet, FiLM, concat 및 State Morgan 적응 | 기록된 약물 반응 적합 명부와 새 가중치 초기화 경로를 확인했다. XPert에는 정적 PPI 유전자 특징이 들어간다. 신규 파일럿에서는 새 물리 대조군과 공통 유전자축에 대해 같은 정보로 예측을 생성한 뒤 동결해야 한다. |
| ChemBERTa Ridge | 기존 약물 반응 적합과 구조 인코더를 구분한다. 고정 인코더의 정확한 사전학습 화합물 명부는 확인되지 않았다. 기존 실행은 구조-only CLS에 새 반응 회귀를 적합했다. |
| scGPT·UCE DMSO 맥락 | 기존 반응 적합에서 정확한 화합물 일치가 없었다. 공개 인코더의 표본별 사전학습 출처는 미확인이다. 기존 적응은 DMSO 세포를 임베딩하므로 파일럿 적용에는 같은 역할의 세포 단위 대조군 입력이 필요하다. |
| 공개 State ST-Tahoe, 범주형 CPA | 기록된 State Morgan 적응과 공개 ST-Tahoe는 다른 자산이다. 범주형 drug 입력으로는 미등록 parent를 직접 조건화할 수 없다. |

학습 이력 감사는 평가 화합물의 처리 RNA를 입력에 주지 않는 A 계약에 한정했다. [T2 자산과 비교군](../goal_tracks/T2_COMPARATORS.md) · [State·CPA 입력 조건](../goal_tracks/PUBLIC_COMPARATORS.md) · [R17 적합 규약](../r17_lincs/PROTOCOL.md)

별도 G1 CPU 작업은 고정된 96군 모두에서 ECFP4 2,048비트와 ChemBERTa CLS 384차원을 생성했다. 잘린 구조는 0개였고 출력 순서·유한성과 체크포인트·토크나이저·배열 SHA-256을 검증했다. 함께 내보낸 978 landmark 매핑도 기존 어댑터의 해시와 같았다. 실제 측정의 유전자축과 물리 대조군은 들어온 assay 명부로 검증해야 한다. 구조 배열은 G1에 보관했다. [구조 지원](data/hcc_pilot_model_support/features/structure_support.tsv) · [구조 영수증](data/hcc_pilot_model_support/features/receipt.json)

감사 중 두 과거 Tahoe 메타데이터 표의 SNU-423과 HepG2/C3A 표시 이름이 뒤바뀐 것을 확인했다. 등록번호로 다시 대조한 결과 각 세포주의 5개 DMSO 웰 판정과 T4의 동일 농도 교차 지원 0군은 수치가 동일했다. 코드와 정정 표에는 등록번호를 함께 남겼다. [Tahoe 표기 정정](SHIFT_RESULTS.md) · [정정 원표](data/hcc_pilot_model_support/audit/tahoe_fit_controls_corrected.tsv)

명세는 `72c9824`에서 고정했다. G1 `31442` 구조 입력과 `31444` 최종 감사 결과의 해시를 대조했다. 처음 감사에서 발생한 원표 열 이름 오류와 세포주 ID 정정 전의 감사도 원장에 기록했다. 발현 학습, 실측 채점, 예약 코호트 접근은 없었다. [감사 영수증](data/hcc_pilot_model_support/audit/receipt.json) · [전체 작업 원장](data/hcc_pilot_model_support/jobs.tsv)
