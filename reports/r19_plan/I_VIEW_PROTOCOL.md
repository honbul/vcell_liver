# DEV-ARCH 입력 뷰 지원 감사

I 트랙은 R17에서 사용한 GSE92742 HEPG2 6 h의 QC 적격 RNA plate를 개발 모집단으로 쓴다. R18의 등록 물리 측정·파생 서명·별칭과 B donor 명부에 겹치는 개발 처리 웰을 제외하고, 기존 CRC32 parent 3-fold에서 뷰별 화합물 지원을 센다. 동일 화합물의 독립 모달리티 관측은 M 정보 계약으로 감사하며, R18에 같은 화합물의 ID가 나온 경우까지 배제한 민감도도 함께 낸다. 실제 RNA 행의 적합·선택과 뷰의 전처리는 감사 뒤의 별도 단계다.

| 뷰 | 고정 입력 및 매칭 |
|---|---|
| I1 | R7의 ChEMBL 37 원 SQLite를 `mode=ro&immutable=1`로 연다. InChI connectivity 기반으로 찾은 레코드를 현재 canonical parent로 똑같이 재현한 경우에만 연결한다. `pchembl_value`, assay 종류와 confidence, 표적 종류·종, relation·단위·원 action type을 보존한다. 표적 수준 수동 기전 주석은 별도 열이다. 음성·미측정은 0으로 채우지 않는다. |
| I2 | Broad cpg0004의 두 batch 중 plate-local DMSO RobustMAD에서 만들어진 median consensus를 받는다. 같은 A549 48 h, 원 dose code 6 조건의 모양 값만 주 뷰에 쓰며, 다른 시간·용량의 겹침은 별도 감사한다. 현재 BRD 구조 매칭 또는 모호하지 않은 역사적 구조 crosswalk를 사용한다. source 파일에 실린 MoA와 target 주석은 I2 값에 넣지 않는다. |
| I4 | Figshare API의 PRISM 19Q4 v4 primary replicate-collapsed LFC와 그 treatment·cell line metadata를 쓴다. 화학구조를 우선 확인하고 BRD와 서로 다른 parent를 가리키면 제외한다. PRISM 파일의 행이 세포주이므로 공식 tissue와 원·현재 cell-line alias를 검사해 전치 전에 간 유래 및 판별 불가 세포주를 제외한다. 부모 화합물별 2.5 µM 최근접 관측을 사용한다. |

## 보정과 데이터 범위

Broad의 plate script는 plate 안의 DMSO만으로 정규화하고, 선택한 consensus 경로는 동일 처리·조건의 반복을 median으로 합친다. Repository의 별도 sphering은 전 batch를 사용한다. Harmony 및 sphered 파일, 전역 특성 선택 variant를 대상에서 제외했다. 표준화·PCA·결측 처리의 파라미터를 학습하게 되는 단계에서는 해당 fold의 적합 parent에서만 산출한다. [Broad 프로파일 문서](https://github.com/broadinstitute/lincs-cell-painting/blob/061870127481dcd73c29df85ebcfddeac2ed0828/profiles/README.md) · [고정 consensus 코드](https://github.com/broadinstitute/lincs-cell-painting/blob/061870127481dcd73c29df85ebcfddeac2ed0828/consensus/scripts/nbconverted/build-consensus-signatures.py)

PRISM 공식 파일은 publisher가 처리 조건별로 ComBat을 적용한 공개 LFC다. 직접 입력에서 간 유래 세포주 값을 제외하고, publisher 전처리를 비간계열만으로 재적합한 자료라는 주장은 하지 않는다. I4는 별도의 공개 비간계열 반응 지문이 있는 경우를 평가하는 M-phenotype 정보 계약이다. [PRISM 19Q4 v4](https://doi.org/10.6084/m9.figshare.9393293.v4)

새 파일의 합계와 로컬 SHA-256을 기록한다. Cell Painting의 git-lfs SHA-256과 PRISM publisher MD5도 대조한다. R7에서 받은 ChEMBL은 원본을 복사하지 않고 읽으며 가공한 산출물은 G1에 두고 Git에는 작은 영수증만 기록한다. I1 CC BY-SA 3.0, Broad profiles CC0 1.0, PRISM CC BY 4.0이다.

주 뷰는 최소 100 parent, outer fold마다 20 parent가 있어야 적격이다. I3 진입은 적격 뷰 두 개 이상이며 동일 지원 비교를 위해 교집합의 수와 두 뷰 이상을 쓸 수 있는 수를 별도로 보고한다.

## I3 수치 검증

동일 parent의 정준 구조에서 기존 Morgan Tanimoto를 계산한다. I1 정량값은 표적 식별자·표적 종·assay type·측정 type·단위가 같은 항목에서 parent 내 median으로 정리한다. 연결 confidence는 7 이상을 쓰고, 사전 어휘는 적합 parent 둘 이상이 가진 항목으로 한정한다. 각 뷰의 표준화·결측 평균·최대 64차원 PCA를 해당 내·외 fold의 적합 parent에서만 추정한다. 선택·평가 parent의 독립 공개 뷰는 추정된 변환으로 매핑한다.

구조 커널을 항상 둔다. 각 사용 가능 뷰의 정규화 벡터로 박스별 Gram을 더하고, 양쪽 행을 현재 가용 뷰 수의 제곱근으로 나눈다. 비용함수에서 양쪽이 공유하는 뷰 수로 나누지 않는다. 이 합은 PSD 커널 각각에 같은 행·열 대각 축척을 적용한 것이므로 PSD다. 선택한 뷰가 모두 결측인 parent끼리는 구조 커널 값이 그대로다. I1+I2를 주 조합, PRISM을 포함한 세 뷰를 보조 조합으로 등록한다. 이 수치 검증에서는 RNA를 읽거나 예측 모델을 적합하지 않으며, 합성 테스트와 실제 뷰의 최소 고윳값 및 fallback 오차를 기록한다.

## 단일 뷰와 결합 대조군

I0는 Morgan 구조만 사용한다. I1·I2·I4는 구조에 해당 뷰를 하나씩 추가하고, I3는 I1+I2, 보조 I3_all은 세 뷰를 결합한다. I1 주석 대조군은 구조와 기존 assay 추출물에 연결된 ChEMBL 표적×수동 작용 주석의 이진 벡터를 사용한다. 적합 parent 두 개 이상에서 관찰된 조합을 어휘로 고정하고, 관측된 벡터를 행별 L2 정규화한다. I2 morphology-only는 같은 형태학 변환과 형태학 값만 쓴다.

Mask-only는 가용성의 두 상태가 같은 정도를 커널로 쓰며, parent shuffle은 뷰별 관측값을 fit과 select/score 각각의 관측된 parent 안에서 고정 seed 1501로 따로 섞는다. 결합에서는 단일 뷰 제거와 zero-fill+mask도 비교한다. Zero-fill은 없는 뷰의 숫자 벡터를 0으로 두고, 가용성 지시자와 뷰 블록을 같은 비중으로 더한 후 구조를 포함한 전체 블록에 고정 가중치를 준다.

사용 가능한 선택 뷰가 0개인 query의 결합 예측은 같은 분할의 I0 예측을 사용한다. 양쪽 모두 결측인 구조 Gram 검증에 더해, 예측 결과의 fallback 행 축도 따로 검증한다. 순수 mask-only와 morphology-only 대조군은 각 입력 권한 그대로 평가한다. 결과 지원은 전체와 동일 뷰 보유군, 일부 보유군, 구조 fallback군을 보고한다.

## 반응 예측과 채점 설계

주 실험은 동일한 개발 웰에서 구조만 쓰는 기준선과 M-annot 또는 M-phenotype 입력을 비교한다. 같은 parent의 비간계열 RNA를 제공하는 B는 이 실험에 포함하지 않는다. R17 계열의 위치 anchor와 FiLM을 공유하며, 강한 비교군 `B_R17`의 FiLM checkpoint와 두 결합 계수는 독립 select 정답에서 DEV 공동 규칙으로 고른다. R17 계열의 커널 규칙은 원 후보군과 선택 절차를 사용한다. I0 및 입력 뷰 모델은 같은 FiLM과 위치 anchor를 고정하고 ridge 커널만 바꾼다. Ridge 규제값은 `.1, 1, 10`, 출력 gain은 R17의 `0, .125, .25, .5, .75, 1`이다. 내분할 select에서 공통 웰 전체와 원 R17 손실로 두 설정을 고른다. 손실은 소수 9자리로 비교하고, 동률이면 gain과 규제값이 작은 순으로 고른다. 각 arm은 선택 뒤 outer-fit에서 다시 적합한다.

예측 입력 묶음은 잠근 R15 처리 캐시 인덱스와 R17 역할별 DMSO 평균의 정확한 인덱스만 읽는다. 예측 프로세스에는 원 캐시 경로와 score 정답을 전달하지 않는다. 3개 fold와 3개 역할 차례를 모두 동결한 뒤 채점 프로세스가 해시를 확인하고 score 역할의 정답을 읽는다. I0와 B_R17 대비 효과는 동일 뷰 지원과 전체 fallback을 따로 낸다. 화합물 단위 짝 bootstrap 10,000회로 두 주 지표의 양측 99% 구간을 계산한다. 동일 지원이 전체 100개 및 각 fold 20개 미만이면 결과를 기술값으로 둔다. 가용성·scaffold 층, mask/shuffle 대조, 공통 FiLM을 뺀 kernel 진단, 물리 plate 하나씩 제외한 점추정을 별도 표에 남긴다.
