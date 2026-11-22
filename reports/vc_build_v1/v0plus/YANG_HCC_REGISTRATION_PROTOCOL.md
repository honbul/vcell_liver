# Yang 성인 HCC 오가노이드의 공개 자료 등록

저자 Mendeley Data v2에서 `RNA.rds`와 `RNA.meta.txt`를 고정한다. 논문의 발행사 S1에서 `HCC` 및 나이 18세 이상인 CaseID만 선택한다. RNA의 `_O` 접미사를 제거한 PDO ID를 S6의 첫 열과 정확히 연결하고, PDO ID의 접두사와 저자 RNA 메타데이터의 CaseID도 일치해야 한다. `_T`, `_E`, `_L`, `_AL`, `_ALO`는 기저 종양 PDO 집합에서 제외한다. 환자에서 채취한 여러 지역은 동일 환자로 묶는다. RNA는 저자가 제공한 log2(TPM+1)이며 중복 유전자 심볼을 모두 제외하고 원천 RNA의 유전자만 유지한다. [원 논문](https://doi.org/10.1016/j.ccell.2024.03.004) · [저자 발현 배포 v2](https://doi.org/10.17632/rv2w3dv9rs.2) · [발행사 S1](https://ars.els-cdn.com/content/image/1-s2.0-S1535610824000898-mmc3.xlsx) · [발행사 S6](https://ars.els-cdn.com/content/image/1-s2.0-S1535610824000898-mmc8.xlsx)

S6의 약물명과 열 위치 및 키만 읽는다. 소분자 6종의 PubChem 구조를 RDKit parent로 정규화하고 정정 68약물 코어 및 1,129약물 확장 학습의 exact parent와 대조한다. Bevacizumab은 항체이며 소분자 구조 커널의 지원 범위 밖이다. 이 교집합은 관측값의 완결성 확인 전의 등록 지원이다. 두 종류의 학습 범위에서 동일 약물·환자로 대조할 수 있는 주 비교와, 확장으로만 지원되는 약물의 보조 비교를 구분한다. 소수 약물에서 상위 다섯 개 적중률은 정보를 주지 않으므로 확인 전에 약물 지원과 해당 지표의 계산 범위를 고정한다.

공개 다운로드의 바이트 수와 SHA-256을 기록한다. Mendeley 발현은 CC BY 4.0이다. 발행사 XLSX에는 별도의 파일별 재배포 라이선스가 확인되지 않았다. RNA와 XLSX는 G1에 둔다. GSA-Human `HRA006499` 원시 파일은 접근 허가가 필요하며 저자 가공 파일과 별개다. [Mendeley 라이선스](https://data.mendeley.com/datasets/rv2w3dv9rs/2) · [GSA-Human 등록](https://ngdc.cncb.ac.cn/gsa-human/browse/HRA006499)

S6 스키마를 확인하는 초기 열 점검에서 첫 5개 PDO의 공개 `Sensitive` 플래그가 출력됐다. 이 플래그는 적합, 지표값, 약물 선정에 사용하지 않는다. 주 평가 후보인 연속 `Normalized AUC` 값은 출력·분석하지 않았다. 등록 코드가 남기는 평가용 행렬에는 실측 열이 없다. 예측 후보와 판정 기준은 해당 실측을 채점하기 전에 별도 커밋으로 고정한다.
