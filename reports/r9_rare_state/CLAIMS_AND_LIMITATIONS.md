# R9 CLAIMS_AND_LIMITATIONS

(결과 수치는 [RESULTS.md](RESULTS.md). 이 문서는 무엇을 주장할 수 있고 무엇을 주장할 수 없는지만 정리한다.)

## 자료가 지지하지 않아 하지 않는 주장

- **전문가 세부 아형(희귀 세부 상태) 인식:** GSE149614 배포 주석에 세부 아형 이름이 없다. Tier 2 결과는 저자 `res.3` 비지도 군집 번호 재현이며, 군집 자체가 시험 환자 세포를 포함해 정의되었다.
- **희귀 암 클론·기능·약물 반응:** 어떤 결과도 이를 뜻하지 않는다. 보류(abstention)는 새 아형 발견이 아니다.
- **Hepatocyte의 외부 일반화:** GSE151530에 대응 label이 없다. Malignant cells를 Hepatocyte 정답으로 쓰지 않았다.
- **봉인된 시험 결과:** HCC02·HCC09는 G1에서, GSE151530 HCC는 phase2-v2에서 이미 채점되었다. GSE149614 결과는 개발 증거, GSE151530 결과는 "R9 고정 버전의 첫 외부 적용"이다.
- **인과 분리:** N×D의 D 효과는 "여러 환자 자료에 노출된 효과"이며 기술 변동과 생물학적 변동을 인과적으로 분리하지 않는다.
- **방법 class 일반화:** 결론은 시험한 설정(PCA50, 대각 공분산 수축 0.3, K=3×클래스, 300 step, 이 allocation 규칙)에 한정한다.

## 섞지 않은 것

- 자연 희귀(Tier 2 군집), 인공 희소화(Tier 1 sparse), 환자 단독 label은 표·결론을 따로 둔다.
- 같은 embedding head 비교(BASELINES `same_PCA50_representation`)와 외부 전체 pipeline 비교(`external_full_pipeline`)는 따로 둔다. scPoli는 시험 donor RNA로 적응하는 transductive 조건이다.
- 선택용 점수(내부 donor-group OOF)는 보정에만 쓰고 일반화 결과로 보고하지 않는다.
