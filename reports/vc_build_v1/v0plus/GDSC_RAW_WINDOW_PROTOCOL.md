# GDSC2 원자료에서 복원한 생존율과 공통 농도 비교

CTRPv2–PRISM에서 확인한 농도 구간 진단을 GDSC2 8.5 원자료까지 확장한다. 기존 세 소스의 학습 코어를 그대로 사용한다. 주 분석 농도와 지원 수준은 앞서 등록한 0.1–1 µM 로그 적분 AUC 및 1 µM 생존율이다. [선행 농도 진단](DOSE_WINDOW_DIAGNOSTIC.md)

1. 등록된 GDSC2 세포·원약물 ID를 DepMap–Sanger 모델 ID로 정확히 연결한다. Glo의 library single-agent 태그 `Lx-Dy-S`만 선택하고 물리 barcode·scan·position으로 중복을 제거한다. 복수 scan, 웰 내부 충돌, 대조와 처리 태그의 중복은 제외한다. Blank `B`와 DMSO `NC-1`의 플레이트별 평균을 이용해 공개 `gdscIC50`의 정규화식을 적용하고 생존율을 0–1로 제한한다. 사전 지정 자료 QC는 `NC-1` 8웰 이상, `B` 4웰 이상, 양의 평균 범위와 Z′ ≥0.3이다. CSV의 `DURATION` 필드는 실제 값으로 기록한다. [Sanger 분석 문서](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [공개 정규화 코드](https://github.com/CancerRxGene/gdscIC50/blob/master/R/nlme_fit_prep.R)
2. 플레이트·library별로 최소 5개 농도, 등록 원래 min/max와 상대 오차 2% 이내인 곡선을 남긴다. 공통 농도 구간 양끝을 덮는 경우에만 로그 농도 선형 보간으로 공통 AUC와 1 µM 생존율을 계산한다. 동일 용량의 반복은 중앙값으로 합친다. 물리 플레이트별 지표와 세포·약물 중앙값의 해시를 기록한다. 발행 NLME AUC와의 같은 약물 내 상관은 보조 진단이다.
3. CTRPv2와 PRISM에 기존 농도 절차를 적용한다. 정확히 같은 세포·약물 지원에서 원래 AUC, 공통 AUC, 1 µM를 비교한다. 소스별 두 소스 교집합과 세 소스 공통 교집합을 각각 보고한다. 계통 안에서 중복 세포를 평균하고 약물별 30계통, 계통별 5약물, 모든 비교 지표의 유한한 변동을 요구한다.
4. 별개 물리 플레이트가 있는 GDSC2의 반복 일치도를 부가 기술한다. 날짜 및 barcode 순으로 첫 플레이트와 같은 `DURATION`의 다음 플레이트를 선택한다. 날짜가 다른 경우를 별도로 센다. 다른 날의 플레이트는 독립 환자를 의미하지 않는다. 이 회차는 자료 진단이며 외부 간암 반응은 사용하지 않는다.
