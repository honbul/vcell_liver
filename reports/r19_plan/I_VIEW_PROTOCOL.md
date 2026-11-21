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
