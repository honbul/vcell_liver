# GSE90653 단일 약물 PDX 기술 평가

Hu 등의 Figure 4c에 나온 독립 HCC PDX 16개를 GEO의 GPL570 F2 비동결 표본과 모델 ID로 연결한다. 각 모델에서 한 GSM만 등록한다. 같은 PDX를 담은 GPL6801, F0/F4, cryopreserved 표본은 지원 수에 더하지 않는다. 약물은 기존 학습 구조 등록부의 sorafenib `MLDQJTXFUGDVEO-UHFFFAOYSA-N`이다. [원논문 PDF](https://lihonglab.github.io/assets/pdf/Intl%20Journal%20of%20Cancer%20-%202019%20-%20Hu%20-%20Establishment%20of%20a%20hepatocellular%20carcinoma%20patient%E2%80%90derived%20xenograft%20platform%20and%20its.pdf) · [GEO 원행렬 디렉터리](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE90nnn/GSE90653/matrix/)

- GPL570 주석에서 단일 유전자 기호만 유지한다. 같은 유전자에 대응하는 probe를 표본 안에서 중앙값으로 합치며, GEO가 제출한 log2 신호를 사용한다. 공개 행렬은 GEO 제출자가 quantile 정규화한 값이다. 이에 따른 입력 단계의 코호트 정보 공유를 기술한다.
- GDSC2·CTRPv2·PRISM 학습 RNA와 공통인 유전자에서 Broutier 기술 평가와 같은 코드로 source-only 재적합한다. 모델 계열, 설정, 출력 배율은 2D source CV에서 선정한 기존 `settings.json`을 사용한다. 비교군은 약물 평균과 source CV 최선 전통 모델이며 후보는 V0+ inductive 하나다. PDX 반응이나 GEO 발현을 학습 및 모델 선택에 사용하지 않는다. 기존 LIMORE·Broutier 결과를 다시 채점하지 않는다.
- 예측은 3개 방법 × 16개 계통으로 동결한다. 별도 채점에서만 PDF와 Figure 4c 막대 순서 파일을 읽는다. 왼쪽에서 오른쪽으로 갈수록 약물 투여 뒤 종양 성장이 작고 감수성이 높다. 그림의 순위가 사전에 공개돼 있으며, 기술 평가로 표기한다.
- 약물 내 계통 Spearman과 짝지은 환자 재표집 95% 구간(10,000회, 시드 90653)을 보고한다. Pearson은 원 수치표가 필요하고 세포 내 약물 순위 지표는 두 약물 이상이 필요하다. 이 결과는 단일 약물 PDX 전이의 기술 근거로 해석하며 LICOB 다약물 오가노이드 확인과 분리한다.
