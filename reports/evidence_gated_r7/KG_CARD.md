# R7 지식 그래프 카드 (KG_CARD)

기계 판독본은 `manifests/evidence_gated_r7/kg_card.json`(파일 sha256 포함)이다. 그래프 배열·DB 덤프는 서버 `assets/kg_v1`, `assets/external`에만 있고 Git에는 없다. 만든 job은 25747이다(D-2 보정 반영본).

## 출처

| 자원 | 버전·접근 | 쓴 부분 | 이용조건 |
|---|---|---|---|
| ChEMBL | release 37 SQLite, EBI FTP `releases/chembl_37`, 2026-11-18 다운로드, sha256 `33c20374…d281`(공식 checksums.txt와 일치) | compound_structures, molecule_hierarchy, drug_mechanism, action_type, activities, assays, target_dictionary/components, component_sequences | CC BY-SA 3.0 |
| OmniPath | 웹서비스 `interactions`, datasets=omnipath,collectri, human, `license=academic`, 2026-11-18 02:07 KST, 150,705행, sha256 `13719218…e749` | 방향·부호·출처 수·참고문헌 수·curation effort | academic(하위 자원별 조건은 OmniPath 표시를 따름) |
| HGNC | complete set 2026-09-08(서버 공용 사본) | UniProt/Entrez → 승인 기호(명명 변환만) | CC0 |

쓰지 않은 것: LINCS compoundinfo의 target/moa 열, DrugBank, STITCH, GO, Reactome, TRRUST, 반응에서 만든 어떤 유사도.

## 약물 → 표적

- **연결:** 구조로만 한다.
  - pert_id의 InChIKey 전체가 같으면 `exact`, 첫 블록만 같으면 `connect`이며 flag를 남긴다.
  - InChIKey가 없는 384개 pert_id는 같은 표의 RDKit SMILES에서 계산했다(D-2).
  - 근거는 일치한 ChEMBL 분자의 parent family(염·형태 포함)에서 모은다. 이름으로는 연결하지 않는다.
- **근거 종류**(따로 보관하며, 하나의 수치로 합치지 않는다):
  - `mech`: 큐레이션된 기전. 작용 부호는 action_type의 상위 분류에서 정했다(`action_type_sign_map.tsv`). 억제·길항·분해 = −, 작용제·활성화 = +, 결합·조절 등 = 미상.
  - `actB` / `actF`: binding / functional assay. pchembl ≥ 6, 관계 '=', 유효성 경고 없음, assay confidence ≥ 7, human 표적.
  - 효능 값은 v1 입력에 쓰지 않는다.
- **연결 결과:**

| | 값 |
|---|---|
| ChEMBL에 연결된 pert_id | 7,983개 |
| 연결된 그룹 | 6,265개 |
| 근거 행 | 31,473개 |
| HGNC 기호가 없는 human 표적 accession | 70개(제외) |

- **주석 상태**(약물 그룹 수):

| 역할 | complete | partial | missing | 계 |
|---|---|---|---|---|
| train | 920 | 1,431 | 3,451 | 5,802 |
| validation | 160 | 346 | 737 | 1,243 |
| test | 172 | 293 | 778 | 1,243 |

- missing의 이유: ChEMBL에 연결되지 않음 2,023 그룹, 연결됐으나 기준을 넘는 human 근거 없음 2,943 그룹.
- 주석이 있는 그룹의 표적 유전자 수: 중앙값 3, 75% 분위 6, 최대 596. 표적 수가 많은 약물은 kinase panel 등 다표적 측정이 많다. 그래서 주석의 풍부함은 "유명하고 많이 측정된 약물"이라는 뜻을 함께 담을 수 있다(평가에서 표적 수 tertile을 따로 보고).
- 주석 있는 그룹 중 표적이 978 landmark 안에 하나라도 있는 비율은 32.6%다. 나머지 표적은 미측정 노드로 들어간다.

## 표적 → 측정 유전자 그래프

| 항목 | 값 |
|---|---|
| 노드 | 4,696개 = 표적 2,453 + 중간 노드 1,540 + landmark 978(표적 중 275개는 landmark이기도 함) |
| 관계 | 63,234개(모두 방향 있음) |
| 채널 | sig+ 19,624 / sig− 6,254 / sig? 7,892 / tf+ 22,442 / tf− 3,778 / tf? 3,244 |

- complex endpoint는 구성 단백질로 펼쳤다(flag). 유일한 HGNC 기호로 바꿀 수 없는 7,797개 endpoint 쌍은 제외했다.
- landmark 노드는 X0 값과 측정 mask를 갖는다. 그 밖의 노드는 값 0과 **미측정 mask 1**을 갖는다(낮은 발현으로 읽지 않는다).
- 나가는 관계가 있는 표적은 1,299/2,453개다. 나머지 표적은 seed는 있지만 그래프로 신호를 보내지 못한다(landmark인 표적은 자기 노드에서 바로 읽힌다).
- **재배선 대조:** 채널별로 도착점을 섞어 차수를 보존했다. 원래 edge는 15.6%만 남는다(seed 20261118, `rewired_graph.json`).

## 하지 않은 것

- 입력 그래프를 GRN 정답으로 채점하지 않는다.
- 없는 관계를 음성으로 학습하지 않는다.
- ChEMBL confidence를 세포 안 작용 확률로 해석하지 않는다.
- 출처 수를 독립 실험 수로 쓰지 않는다(log 척도 입력 feature로만 사용).
