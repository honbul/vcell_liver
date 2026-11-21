# Tahoe SNU-423 scaffold 차단 재적합

P4와 사후 구조 지원에서 일부 근접 유사체에 개선이 집중됐다. 이번 DEV 후속은 평가와 같은 scaffold 성분을 적합에서 실제로 빼고, 같은 수를 일반적으로 빼는 경우와 구분한다. 기존 5 µM·24 h·최소 20세포 P4 평가 웰과 203 parent의 세 fold, 181개 Bemis–Murcko 연결 성분, 두 물리 대조군 역할, 같은 웰 전 세포주 평균을 뺀 목표를 사용한다. 화합물의 구조만 제공하는 A 계약이다.

- 유전자축은 반응에 따라 고르지 않는다. R15의 기존 978 landmark Ensembl 명부와 원 Tahoe 전체 token 주석에서 양쪽 일대일인 유전자 전부를 고정한다. 원 R12의 개발용 0.05/5 µM 집계에는 격리·예약 웰이 없음을 행 명부에서 검사한 뒤, 기존 P4 5 µM 처리 행과 물리 대조군에만 해당되는 값을 추출한다. source와 원 manifest 해시를 확인한다. RNA panel 적합, 순위, 새 유전자 선택은 없다. 이 정규화는 원 full-library log1p(CP10k)다. 새 panel이므로 이전 2,000유전자 P4 점수와 수치를 직접 대조하지 않는다.
- 각 outer fold에서 공통 P4 평가행을 고정한다. 새 적합은 (1) 같은 fold의 전체 적격 parent (2) 평가와 scaffold 연결 성분이 하나라도 같은 적합 parent를 모든 세포주에서 제외한 parent (3) 그와 같은 수의 SNU-423 적합 parent를 고정 seed `4232026 + fold`로 제거한 대조다. (3)은 같은 개수만 통제하며 화학 특성이나 손실로 매칭하지 않는다. 세 fit에서 실제 단일세포주 지원 수와 배제 명부를 기록한다. (2)의 적합 parent와 평가 parent는 성분 단위로 서로소임을 검사한다. 같은 웰에서 10세포주 이상, pool에서 3 parent 이상 조건은 각 fit 범위에서 고정한다. fit 목표는 own-role 차량을 뺀 뒤 웰 중심화하고 적합 pool 안에서 parent 가중 중심화한다. 각 parent는 SNU-423의 첫 적격 entity를 대표로 하며 동일 entity의 독립 웰을 평균한다.
- 각 fit에서 고정 λ=1·배율 1·절편 0 ECFP4 Tanimoto Ridge와 Tanimoto ≥0.70인 가장 가까운 약물 복사를 산출한다. 동률 copy donor는 숫자 parent ID가 가장 낮은 것을 고르고 임계 미만 예측은 0이다. scaffold 차단 fit에는 같은 λ·배율로 ChemBERTa cosine Ridge도 맞춘다. B0의 예측은 0이다. 기존 SNU 화학 지원 결과를 보고 0.70을 택했으며 새 score로 배율이나 기준을 고르지 않는다.
- 주 비교는 scaffold 차단 ECFP Ridge 대 같은 차단 fit의 ChemBERTa와 B0다. 두 비교 각각에서 상대 RMSE 개선 점추정 ≥0.5%, pooled R² 차이 ≥0.5%p, RMSE 차이 99% 상한 <0, R² 차이 99% 하한 >0을 요구한다. 성분 10,000회 대응 재가중에서 평가 pool을 다시 중심화한다. parent 재가중은 참고다. ECFP와 복사의 차단 대 동일 크기 제거 및 전체 적합 비교, 각 fit 범위의 최대 Tanimoto와 임계 도달은 구조 의존성 진단으로 보고한다. 이 비교가 기존 VC 우위나 외부 HCC 확인을 대신하지 않는다.

명세 고정 후 사전 추출 작업이 기존 P4 개발 행을 fit과 score로 분리한다. 예측은 자기 fold의 fit과 held 메타데이터만 받으며 반대 역할 대조군·평가 발현은 score 프로세스가 동결 검증 후 연다. 보류·예약 자료, 별도 외부 발현과 신규 다운로드는 쓰지 않는다.
