# 데이터 지원, 입력 동등성 및 평가 노출

고정 H2 등록에는 670,830개 반응 행, 정확 parent 1,186개, 대표 Murcko 골격 1,001개, 세포 모델 1,014개와 전역 계통 991개가 있다. 소스 반응 행은 CTRPv2 285,842, GDSC2 83,995, PRISM20Q2 300,993개다. Drug-Cold의 3개 outer는 기존 H2 396/395/395 부모의 동일 행이다. 별도 Cell-Cold는 전체 확장 행을 한 번 나누고 68약물 교차스크린 평가 지지는 기존 H3-v1의 45,428행을 사용한다.

| 정보 | 등록 지원과 신규 모델 입력 |
|---|---|
| 전체 basal RNA | 정정 17,871 측정 유전자에서 분할 내 PCA 최대 512 및 변동/표적 발현. |
| H2 RNA 기하 | 14,421유전자 축은 정정 축의 부분집합. H2의 RNA 표준화 벡터의 학습 span 좌표와 query norm을 포함한다. |
| 경로·TF | H2 입력의 경로/TF 338개와 적합 계통에서 발현으로만 유도하는 130개 내외 Chronos·driver proxy. RNA를 직접 입력하는 주 비교다. |
| 분자 구조 | ECFP4 2,048비트, H2 RDKit 물성 4개. GNN에 한해 동일 SMILES의 원자·결합 adjacency 추가. |
| ChEMBL 37 | 정확 parent의 인간 타깃 1,451개 유전자 축 및 직접 기저 발현 평균·최대·표준편차와 지지 수·비율. |
| LINCS 2020 | 24 h MODZ 978 landmark와 44개 경로, 별도 측정 여부. 코어 1,186 parents 중 724개 관측, 462개 결측. |
| 생물학 교차항 | 44개 세포 활성×약물 변화량, 평균·최대 절댓값·cosine·부호 일치와 타깃 발현. |
| 사전학습 표현 | 읽기 전용 CLAMP와 Uni-Mol SHA 확인. GNN 사전학습 서브암에서만 추가한다. |

분할의 집계 및 archive 해시는 G1 H3-v2 루트 `splits/manifest.json`에 있다. 관측이 없는 LINCS는 공통 atlas에서 NaN으로 보존한다. LightGBM은 native missing 분기를 쓰고 FiLM·GNN은 관측 플래그로 branch 출력을 제어한다. Transformer는 같은 flag로 LINCS token의 key mask를 건다. 기본 분자·RNA 입력은 archive의 H2와 같은 원자료이며 정규화 방식은 각 모델의 정해진 분할 내 변환을 따른다.

H2의 상호작용 ρ 약 .2737은 LINCS 관측 지원이다. 혼합 배포 전체 지원의 상호작용 ρ는 약 .2304다. 기존 H2의 동일 행·동일 산식을 통해 두 지원을 각각 대조한다. [H2 동결 비교](../nextgen_h2_multiview/RESULTS.md) · [H3-v2 잠금](PLAN_LOCK.md)

GDSC1의 66 신규 골격은 이전 H2에 등록됐고 소스 세포와 일부 공유된다. LIMORE 50계통·68약물 및 LICOB 성인 38공여자·76약물은 과거 평가에 노출된 기술 확인이다. PUB LINCS 프로필은 H2 등록의 비간계열 MODZ를 사용하고 반응 실측은 모델 선택에 넣지 않는다. [H2 자료 등록](../nextgen_h2_multiview/DATA_SUPPORT_AND_EXPOSURE.md)

첫 R1 연동에서 정정 RNA의 H2 span을 재구성한 Gram 최대 차이는 4.81e−8이었다. Drug-Cold 0의 첫 inner fold에서 LINCS 478,242개 관측 좌표 중 94개는 표준화 절댓값이 8을 넘었다. 따라서 두 LINCS 원자료 뷰의 정규화 좌표는 모두 절단 없이 제공하며, 최초 제한형 실행은 후보 선정에서 제외했다. H2 고정 입력을 재현하려는 목적의 수정으로, 정정 반응 파일과 분할, PCA 및 구조 자산은 그대로 사용했다. [입력 loader](../../src/h3v2/atlas.py)

세 외부 입력의 동일 행 검증은 실측 개봉 전에 진행했다. GDSC1의 exact parent·connectivity·모든 골격 표현 교집합은 각각 0이다. 등록된 LICOB의 43 PDO와 성인 38공여자를 구분했다. LIMORE·GDSC1 발현은 17,871개, LICOB 발현은 14,421개 유전자의 측정 지지를 가진다. 이후 모델의 플랫폼 결측 유전자 보완은 해당 소스 적합의 유전자별 중앙 순위만 사용한다. [외부 입력 영수증](external_roster_verification.json) · [분할 명세](split_manifest.json) · [실제 feature/OOF 검사](step0_verification.json)

C1 및 H2의 16개 비교 로스터와 교차스크린 9개의 H3-v1 고정 예측을 정확 행 키로 대조했다. Drug-Cold 670,830행, Cell-Cold 670,830행, 68약물 교차스크린 45,428행을 재확인했다. 실측 개봉 없이 수행한 검사다. [비교군 로스터 영수증](reference_roster_verification.json)

외부 전처리 사전 검증에서 1,186개 parent의 소스 관측 670,830행으로 적합하고, 3개 외부 코호트의 입력 46,213행을 변환했다. 512개 RNA 주성분, 978개 LINCS landmark와 44개 경로, 정확 행과 정렬된 유한 H2 OOF·외부 기준값을 확인했다. LICOB의 3,268행은 성인 공여자 단위 집계 전 입력이다. [외부 전처리 영수증](external_atlas_preflight.json)
