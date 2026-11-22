# GDSC1 독립 측정의 추가 학습 및 측정 보류

GDSC1 8.5의 72시간 생존율 원 AUC를 별도 측정 소스로 등록한다. GDSC1은 Sanger·MGH의 Syto60 또는 resazurin이고 GDSC2는 Sanger의 CellTiter-Glo다. 같은 계통이 반복되고 연구기관도 일부 공유한다. GDSC2 보류의 보강은 같은 프로젝트 계열의 측정 간 전이로 해석한다. CTRPv2와 PRISM 보류는 Broad 측정으로의 전이로 구분한다. 원 자료와 적합 예측은 G1에 보관하며 내부 연구용으로 사용한다. [Sanger 측정 명세](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [자료 사용 조건](https://depmap.sanger.ac.uk/documentation/data-usage-policy/)

- GDSC1 전체 원표는 앞선 LIMORE 보호 감사에서 이미 명부에 포함했다. 동일한 정정 코어 1,014모델과 공개 DepMap 명부를 사용하며 Sanger ID가 일치하고 알려진 COSMIC ID가 충돌하지 않는 모델만 연결한다. 신규 외부 코호트 발현이나 반응은 이번 적합에 사용하지 않는다.
- 기존 검증 GDSC2와 동일한 공식 native ID와 공식 이름 또는 동의어인 경우 검증된 exact parent를 유지한다. 나머지는 IMPROVE의 native 번호·이름·구조와 공식 주석을 대조한다. 혼합물, 여러 parent, 미지정 입체 요소가 있는 신규 ID는 제외한다. 정확한 parent가 기존 등록 1,186약물에 있는 행만 연결하고 주 적합에는 원래 정정 68약물만 포함한다. 약물마다 원래 등록 세포 지지가 가장 큰 하나의 native ID×농도 범위를 반응값에 따른 선택 없이 고정한다. [공개 구조 카탈로그](https://doi.org/10.5281/zenodo.22920452) · [기존 화학 등록](LICOB_COLD_RESULTS.md)
- 기존 계통 및 소스 보류 분할을 유지한다. 12개 분할에서 정정 코어의 약물 평균, 구조 커널, Ridge를 기존 동결본으로 고정한다. 후보는 같은 학습 계통의 GDSC1 AUC를 추가한 구조 커널이다. 정정 코어의 앵커·전처리와 원래 세 소스의 척도를 보존한다. GDSC1은 해당 fold의 같은 학습 계통·약물 앵커에서 기존 기준 스크린에 별도 척도를 적합하고 소스별 약물 평균을 뺀다. 기존 벌점 4 및 배율 1을 사용한다. 주 후보는 동등 소스 가중, 보조 조건은 GDSC1 관측 가중 ×.25다. 성인 HCC 및 주요 TKI 하위군을 보고한다.
- 기존 코어만 적합한 구조 커널과 Ridge 및 약물 평균을, GDSC1을 통째로 보류하고 기존 세 계통 fold의 GDSC1 행에 적용한다. GDSC1은 이 비교에 전혀 적합되지 않는다. 약물당 보류 계통 5개 이상인 약물 내 Spearman·Pearson, 계통 내 Spearman·NDCG@5·Precision@5를 보고한다.
- 사전 승격은 보강 주 후보가 CTRPv2·PRISM 두 보류 조건에서 모두 정정 코어 대비 약물 내 Spearman·Pearson의 짝지은 95% 하한이 0을 넘고, 두 계통 순위 지표 차이의 하한이 −.005 초과이고 약물 평균 대비 점 추정치가 비음수일 때다. 두 보류 조건의 전체 등록 행이 주 판정이며 보강 지원 약물은 보조 분석이다. 간암 외부 반응을 추가로 열지 않는다. 기존·신규 분석의 15개 예측을 SHA-256으로 먼저 동결하고 별도 작업으로 채점한다.
