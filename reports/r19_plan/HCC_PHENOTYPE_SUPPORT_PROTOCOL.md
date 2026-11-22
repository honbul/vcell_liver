# SNU-423 표현형 입력 지원 감사 명세

기존 I 트랙에서 확보한 A549 Cell Painting I2와 비간계열 PRISM I4가 Tahoe의 SNU-423 5 µM·24 h 평가에서 얼마나 지원되는지 확인한다. 이전 scaffold 차단 재적합에서 해시를 고정한 203군의 평가 명부와 같은 단일세포주 지원 및 scaffold 차단 적합 명부를 유지한다. 기존 공개 처리 프로파일과 metadata만 쓰며 신규 수집은 없다.

- 원 I 트랙의 구조 정규화, 고유 BRD 구조·별칭 매칭, Cell Painting A549 48 h dose code 6, PRISM의 2.5 µM 최근접 profile을 보존한다. Tahoe group이 한 canonical parent로 유일하게 해석되는 경우에만 매칭한다. 한 canonical parent가 여러 Tahoe group에 연결되면 해당 그룹 전체의 표현형 입력을 보류한다. 해석 실패는 미지원으로 계수한다.
- PRISM의 공개 입력은 기존 tissue 검사에서 간·담관계 및 조직 미확인 세포주를 제외한 지문이다. 원 배포의 treatment별 ComBat을 라이선스와 함께 명시한다. Fit-only 전처리의 지원은 scaffold 차단 fit에서만 학습한 가변 좌표와 PCA 최대 64축으로 확인한다. 후보 assay와 출처의 측정 농도·시간을 일치한 것으로 해석하지 않는다.
- 평가 203군, 각 outer fold, scaffold 차단 fit의 뷰별 수 및 총·이중·한쪽·미지원군을 낸다. R18 코호트와 B donor 예외에 등장한 동일 parent를 제외한 민감도는 메타데이터로만 센다. 어떤 발현값도 열지 않는다.
- 후속 예측에 진입할 뷰는 전처리 후 전체 평가 100군 이상, 각 평가 fold 20군 이상 및 각 scaffold 차단 fit 20군 이상이어야 한다. 두 뷰 결합에는 전체·fold별·fit의 동시 지원에도 같은 기준을 적용한다. 한 뷰만 통과하면 그 뷰의 단독 DEV 후속만 명세할 수 있다. 후속 실험은 M-phenotype이고, 공통 지원과 전체 fallback, 결측 마스크 및 parent 순열 대조를 포함해 별도로 채점 전에 고정한다.
