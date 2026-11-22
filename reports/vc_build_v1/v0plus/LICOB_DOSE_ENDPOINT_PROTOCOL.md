# LICOB의 최대 효과 및 반응 농도 전이

이전 공개 LICOB 분석의 43개 RNA 모델·성인 38명과 입체 구조를 정정한 76약물을 그대로 사용한다. 기존 AUC 평가 이력이 있는 코호트의 후속 기술 분석이다. S7의 IC50과 S8의 Emax에 대해 아래 후보와 판정을 먼저 등록한다. GDSC2·CTRPv2·PRISM 학습원과 환자·계통이 겹치지 않는 LIMORE 성인 HCC 50계통은 별도 핵심 공개 2D 검증이다. 해당 코호트에서 실측 HCC 변이·CNA, 약물 표적 발현 및 RNA·ECFP·경로를 쓰는 비교는 이미 고정한 성적을 유지한다. [LICOB 등록과 AUC 성적](LICOB_COLD_RESULTS.md) · [LIMORE 외부 검증](LIMORE_CORRECTED_RESULTS.md) · [표적·실측 변이·CNA 비교](MEASURED_DRIVER_RESULTS.md)

## 종말점과 지원

- 공개 S5의 최고 농도, 희석 배수, 농도점을 쓴다. Emax는 최고 두 농도의 생존율 중 작은 값이다. 적합에는 두 농도가 모두 실제 2D 측정 범위 안에 있는 곡선만 쓴다. IC50은 생존율 0.5 최초 교차점이다. 동일 약물의 LICOB 최저 및 최고 농도를 모두 덮는 소스 곡선에서만 로그 농도를 선형 보간한다. 미교차는 측정 범위의 적절한 끝에 둔다. S7의 공란은 제외하며 0과 양·음의 무한대 스트링은 경계에 둔다. 양의 유한 수치는 S5 범위에 제한한 뒤 `−log10`으로 변환한다. 실제 값이 한계 밖인 경우를 낮음·높음으로 각각 보고한다. IC50은 범위 제한 순위를 주 기술값으로 해석한다. Emax와 AUC는 부호를 반대로 해서 세 지표 모두 값이 클수록 감수성이 크다. [저자 보충 방법 및 Fig. S4](https://www.ebi.ac.uk/biostudies/files/S-EPMC10949980/NIHMS1969891-supplement-Supplementary_Materials.pdf) · [GRmetrics 구현](https://rdrr.io/bioc/GRmetrics/src/R/GRfunctions.R)
- CTRPv2는 SNP 확인을 통과한 공개 응답점, GDSC2는 동일 native 세포·약물의 공개 well 값과 플레이트 정상화·Z′ 기준, PRISM은 기존 AUC를 절대오차 .005 이하로 복원하는 공개 연속 적합을 쓴다. 공통 물리 농도 내 보간만 허용한다. 관측 생존율은 곡선 요약 전에 0–1로 제한하고 반복 곡선은 소스 내 중앙값으로 합친다. AUC 대조는 같은 곡선의 측정 로그 농도 전체에서 평균한 상대 생존율이다. [CTRPv2 원자료](https://portals.broadinstitute.org/ctrp.v2.1/) · [GDSC 원자료](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [PRISM 저자 계산 코드](https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R)
- 약물마다 2D 등록 세포가 30개 이상인 정확한 parent만 맞춤 평가에 넣는다. 지표별 등록 약물 5개부터 동일 지원 회귀 적합과 외부 기술값을 보고한다. 10개 이상이면서 약물당 성인 공여자 5명 이상과 순위 변이가 있어야 공식 기준을 적용한다. 이 범위의 약물명·농도·소스 지원을 결과와 같이 남긴다. 이전 동결 AUC의 상호작용 모델과 화학 약물 평균을 약물 76개 전체의 세 정답에 그대로 대입해 기술값을 보고한다.

## 공통 지원 모델 및 판정

종말점 모델과 곡선 AUC 비교군은 해당 종말점의 **동일한 2D 소스 행**을 사용한다. 소스×약물별 평균에 대해 잔차를 만들고 소스별 약물 평균에 같은 가중을 준다. 이전 분자 표현, 커널, 벌점 4, 출력 배율 1을 유지한다. 종말점 약물 평균, AUC 분자 상호작용, 종말점 분자 상호작용을 동결한다. Emax와 IC50을 별도 보고한다. 환자당 여러 모델은 동등 평균한다. 사전 판정은 두 비교군 모두 대비 약물 내 Spearman·Pearson의 짝지은 95% 하한이 0 초과이고, 환자 내 Spearman·NDCG@5 하한이 −.005 초과다. 약물 내 구간은 공여자·약물 교차 재표집, 환자 내 구간은 공여자 재표집이다. Precision@5, AUC와 종말점의 관측 일치도는 기술 분석이다.

S5 등록 및 소스 곡선 작성에는 LICOB 발현·약물·농도 메타데이터만 전달한다. S6–S8 결과는 모델과 기존 AUC 비교군의 행과 해시를 동결한 뒤 별도 채점 단계에서 읽는다. 공개 LICOB의 이전 평가 이력을 유지하며 이번 결과로 배포 체크포인트를 교체하지 않는다.
