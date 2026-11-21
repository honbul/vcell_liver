# R19 개발 결과와 R18 확인 준비

## 가장 재현된 효과

| 질문 | 지원 | 관찰 | 판정 |
|---|---|---|---|
| HEPG2 6 h 실측으로 24 h 예측 | 132군, 독립 24 h 물리 웰 268개 | 저랭크 연산자의 단순 배율 대비 RMSE 7.33% 감소 [99% 구간 5.06, 9.58], R² +14.32%p [10.42, 18.06] | 동일 assay B에서 두 주 비교 통과 |
| 시간 전이에 약물별 6 h 입력이 기여하는가 | 전체 132군과 같은 농도 순열 적격 89군 | 정확한 약물 입력의 RMSE는 fit 평균보다 7.25%, 순열 입력보다 16.11% 감소 | DEV 후속 대조에서 기여 확인 |
| 같은 약물 외부 세포주 반응의 B 전이 | 493군, HEPG2 웰 2,324개 | R17 결합의 donor anchor 대비 ΔRMSE −0.01965 [−0.02245, −0.01687], ΔR² +3.27%p [2.71, 3.83] | 같은 L1000 B에서 기준선 위 잔차의 가치 확인 |
| 구조와 손실을 바꾸면 더 나아지는가 | A 1,862군, B 493군 | A·B의 S2와 L1은 실용 기준 미달. S3는 연산 감소와 매우 작은 오차 감소 | 승격 후보 없음 |
| 실측 보조 뷰를 추가하면 나아지는가 | 공통 345군과 fallback 전체 1,862군 | I5 kernel의 B_R17 대비 상대 RMSE 이득 0.200%와 0.046%. FiLM 증분 효과의 두 구간은 0 포함 | 승격 기준 미달 |
| 같은 화합물·용량의 24 h 반응이 다른 assay에서도 일치하는가 | HepaRG 5군과 3군, 공통 랜드마크 827개 | 시간 모델의 평균 cosine이 LINCS 내부 0.247·0.220, HepaRG 0.024·−0.017 | 좁은 기술 비교에서 교차 assay 개선 근거 미확보 |

[시간 전이](T4_RESULTS.md) · [기여 대조](T4_ATTRIBUTION_RESULTS.md) · [S/L B](SL_B_RESULTS.md) · [S/L A](SL_RESULTS.md) · [I5](I5_RESULTS.md) · [교차 assay 결과](T4_CROSS_ASSAY_RESULTS.md)

## 비교의 공정성과 한계

같은 수의 대조 웰에서 fit·select 역할을 분리한 D1은 배율이 0이 아니었던 여섯 비교군 모두의 과대 배율을 줄였다. D2의 공동 선택은 FiLM·Concat·XPert의 두 지표를 개선했다. D3의 fit-view 증강은 사전 실용 문턱에 못 미쳤다. 구조·맥락·표현형의 이득은 기존과 같은 역할에서 평가했다. [D1·D2](D12_RESULTS.md) · [D3](D3_RESULTS.md)

T4의 배율 대비 RMSE 개선과 R17 B의 donor 대비 개선은 상위 기여 20군을 각각 제외해도 3.95%, 1.77%였다. 부모와 plate를 함께 재가중한 기술 범위는 모두 개선 방향이었다. [화합물·plate 기여도](ROBUSTNESS_RESULTS.md)

동일 assay의 T1 전역 배율은 원 donor RMSE를 1.47% 줄였고, T3의 T1 대비 추가 변화는 Holm 기준 0.01을 통과했으나 양쪽 99% 상한이 실용 문턱보다 작았다. X의 기저 맥락과 donor 재가중은 약물별 반응 차이의 실용 이득을 보이지 못했다. 기존 Tahoe·LINCS의 5 µM·24 h 교차 비교에서는 셀·농도를 맞춘 6/24 h 공통 지원이 0군이었다. [T1–T3](T123_RESULTS.md) · [T 비교 지원](SHIFT_RESULTS.md) · [X A](X_RESULTS.md) · [X B](X3_RESULTS.md) · [Holm 원표](data/dev_family/primary.tsv)

## 확인 상태

R14 동결 결과를 회수하고 R17 재현을 점검했다. R18 C1–C3는 지원과 노출, 접근 명부와 동결·채점 검증까지 준비했다. R18 처리값은 개방되지 않았으므로 원 판정은 미실행이다. 위 수치는 이미 노출된 DEV와 현재 허용된 측정을 중심으로 계산했다. 성인 간세포암에서 기존 VC를 능가한다는 목표에는 독립 확인 결과가 필요하다. [R14 및 R18 지원](progress_reconciliation.md) · [R18 사전 명세](R18_CONFIRMATION_PROTOCOL.md) · [실험별 상태와 판정](experiment_matrix.tsv) · [쉬운 요약](BEGINNER_SUMMARY.md) · [다음 판정](NEXT_DECISION.md)

논문 작업용 초안은 위 결과의 방법·수치·적용 범위를 원표에 연결했다. [영문 초안](manuscript/DRAFT.md) · [주장별 근거](manuscript/CLAIMS.md) · [벡터 그림](manuscript/figure1.pdf)

공개 성인 HCC 연구 여섯 건의 표본 등록을 추가 감사했다. 처리 전사체 연구가 있었고 대부분 한두 약물의 기술 비교에 해당했다. 주요 두 연구를 합친 단일 약물은 세 개였다. [공개 HCC 자료 감사](HCC_PUBLIC_SUPPORT.md)
