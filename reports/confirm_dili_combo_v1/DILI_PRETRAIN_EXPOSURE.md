# CLAMP 사전학습 노출 감사

동결한 공식 CLAMP 가중치와 설정의 SHA-256을 확인했다. 이 checkpoint 설정은 PubChem 계열 경로, `time_a` assay 분할, active-only 학습을 가리킨다. Seidl 등의 공개 실험은 PubChem BioAssay 2018과 ChEMBL27 유래 FS-Mol을 다룬다. 이 checkpoint 설정에 ChEMBL 학습 소스는 명시돼 있지 않다. [ICML 원논문과 부록](https://proceedings.mlr.press/v202/seidl23a.html) · [공식 데이터 문서](https://github.com/ml-jku/clamp/blob/main/data/pubchem.md) · [공식 분할 코드](https://github.com/ml-jku/clamp/blob/main/clamp/train.py)

저자 PubChem18 보관본의 2,120,811 분자, 21,002 어세이, 223,219,241 activity 행을 확인했다. FDA 등록 CID 및 원본·정규 부모 SMILES와 공개 분자 목록을 대조하고 일치 후보의 InChIKey를 재계산했다. 연결성은 InChIKey 앞 14자리, exact는 정규 charge-parent 전체 키를 사용했다. 텍스트로 일치하지 않고 조회 CID가 다른 별칭 구조는 추가 탐색 범위 밖이다. 공개 코드의 oldest 60% assay 및 activity 1 조건을 재구성했다. 마지막 포함 AID는 602432다. [보관본 및 집계 영수증](evidence/dili_exposure_receipt.json)

| FDA 범위 | N | 등록 CID | Exact 부모 | Connectivity | 초기 assay | 초기 active | Exact 부모의 초기 active |
|---|---:|---:|---:|---:|---:|---:|---:|
| 등록 scaffold | 887 | 736 (83.0%) | 415 (46.8%) | 739 (83.3%) | 610 (68.8%) | 531 (59.9%) | 325 (36.6%) |
| 등록 증분 | 155 | 89 (57.4%) | 38 (24.5%) | 89 (57.4%) | 33 (21.3%) | 26 (16.8%) | 14 (9.0%) |
| 이전 FM 증분 | 147 | 88 (59.9%) | 38 (25.9%) | 88 (59.9%) | 33 (22.4%) | 26 (17.7%) | 14 (9.5%) |
| 이전 FM scaffold | 858 | 723 (84.3%) | 408 (47.6%) | 726 (84.6%) | 601 (70.0%) | 527 (61.4%) | 322 (37.5%) |

초기 assay와 초기 active 열은 connectivity 일치 항목의 관측이다. 공개 코퍼스에는 FDA 약물 구조가 실제로 포함돼 있다. 특정 checkpoint의 최종 학습에 포함된 전체 분자 목록과 중간 학습 로그는 확인되지 않았다. 처음 보는 분자에 대한 완전 비노출 결과를 주장할 근거는 없다. 공개 코퍼스 중복 수를 FDA concern 라벨 노출 수로 해석할 수도 없다. ChEMBL·FS-Mol의 별도 실험은 위 표의 분자 및 어세이 분모에 포함하지 않았다.
