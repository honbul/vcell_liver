# Paper 1 제출 전 체크리스트

기술 감사는 완료했다. 투고는 **저자 확인 및 공개 범위 승인 대기**다. 감사 대상은 r14-cross-organ-data `b80e38cd362417f79ad6ef275f383e899693d276`의 원고와 T4 v2 `aa4112e4079a481d1893f7e987ae9022838f5c32`였다. 원본 워크트리는 읽기 전용으로 참조했다. [원고](../r19_plan/manuscript/DRAFT.md) · [근거 목록](../r19_plan/manuscript/CLAIMS.md) · [리뷰 상태](../r19_plan/manuscript/REVIEW_STATUS.md) · [검토 DOCX](../r19_plan/manuscript/REVIEW.docx) · [감사 영수증](evidence/paper1_artifact_audit.json)

## 수치, 그림, 주장

| 상태 | 항목 | 제출 전 확정 내용 |
|---|---|---|
| 확인 | 원고와 Word 표 | DRAFT·CLAIMS·S1·표 1·벡터 그림 3개·DOCX의 해시를 리뷰 영수증과 대조했다. DOCX ZIP의 CRC 및 표 1의 17개 데이터 행 전체를 검사했다. 비교군명, parent/well 수, 수치, 부호, 반올림, 99% 구간과 R18 관문 표기가 TSV와 일치했다. |
| 확인 | HCC 표 1 통계 재생 | JHH-5 305 parent/309웰과 JHH-7 88/88의 원척도·중심화 비교 4개를 Slurm 33827에서 다시 계산했다. 두 집계 파일 SHA-256이 기존 영수증과 일치했다. 이 재생은 익명 parent별 오차 통계의 재집계 범위를 검증한다. |
| 확인 | 핵심 성인 HCC 해석 | 10 µM·24 h·같은 측정 donor에서 적응 gain의 상대 RMSE 감소는 JHH-5 1.720%, JHH-7 2.609%였다. 같은 gain 명세에서 shrinkage의 추가 감소는 0.014%, 0.071%로 두 99% 구간에 0이 포함됐다. 초록, Fig. 3, 표 1과 CLAIMS가 일치한다. JHH-7은 한 평가 RNA plate다. |
| 확인 | HEPG2 T4 rank-16 | 같은 6시간 입력에서 centered scalar 대비 4.164%, 대각 Ridge 대비 3.382%가 초록·표 1·Fig. 1의 해당 비교와 일치한다. 7.237%는 다른 origin/nonnegative scalar 비교다. State 대비 7.442%는 같은 6시간 실측을 받은 1-token State 재적합 결과다. 각 비교군 이름을 캡션에서 유지한다. |
| 정리 필요 | State 입력 설명 | Methods는 앞선 605/97-parent 8-donor 비교에 8개 평균을 **1 token**으로, off-position 305/88 분석에 8개 벡터를 **8 tokens**로 제시한다. Comparator adaptation summary의 State 행에는 성인 HCC 8-token만 기재돼 있다. 두 설정을 행 안에 각각 표기하고 Fig. 2의 8-donor 분석과 Fig. 3의 off-position 조건을 구분한다. |
| 정리 필요 | R18 C1에서 지표별 비교군 표기 | Results의 절대 RMSE 감소 0.02024는 전체 최선 기존 VC 대비다. 같은 문장의 R² +0.183%p와 표 1의 RMSE 감소 0.00747은 모두 R15 커널 비교의 값이다. Results에서 두 비교를 별도 문장으로 쓰고 각 comparator 이름을 명시한다. Fig. 1c와 표 1은 R15 기준을 유지한다. A 4/12, B 13/14의 C1 공동 관문은 둘 다 미통과다. |
| 정리 필요 | 초록에서 노출과 재현 범위 | JHH 조건은 기존 B1 노출 자료이며, JHH-7은 한 plate다. 이미 본 개발 코호트라는 설명이 Methods, Fig. 3과 CLAIMS에는 있다. 독립 성인 HCC 초록 해석을 막으려면 초록의 성인 HCC 결과 문장에 개발 코호트임을 직접 표기한다. |
| 확인 및 유지 | 전이 한계 | HEPG2는 hepatoblastoma 유래다. HepaRG 교차 분석은 5·3 parent의 TempO-Seq 비교로 플랫폼과 세포 맥락이 함께 바뀌었다. JHH 두 세포주 차이는 target plate와 교락했고 저용량 이득은 일관적이지 않았다. 96개 성인 HCC 파일럿은 예측만 동결됐고 실측은 없다. |
| 확인 및 유지 | 모델 범위 | 현재 제목·초록·Discussion은 같은 입력의 L1000 평가와 특정 재적합 모델을 기술한다. 최신 VC 전체 우위, 실전·임상 VC 완성, 단일 모델의 지표별 최고 결과 혼합은 확인되지 않았다. DILI·병용은 별도 종말점으로 Paper 1 결과에 포함하지 않는다. |
| 정리 필요 | 참고문헌과 그림 제출 | DOI 1–10의 서지와 Word의 17개 번호를 확인했다. 참고문헌 17의 PRISM DOI는 사용한 Figshare 버전에 고정하고, 11의 CLUE 홈페이지 링크에 정확한 L1000 릴리스 및 파일 식별자를 붙인다. 8·10은 투고 날짜에 출판 상태를 재확인한다. Figure 1–3은 원본 벡터 PDF를 제출용으로 내고, Word 내 PNG는 리뷰 미리보기로 취급한다. |

근거는 [Figure 1](../r19_plan/manuscript/figure1.pdf) · [Figure 2](../r19_plan/manuscript/figure2.pdf) · [Figure 3](../r19_plan/manuscript/figure3.pdf) · [R18 결과](../r19_plan/R18_RESULTS.md) · [1-token 출처](../r19_plan/HCC_DONOR8_STATE_PROTOCOL.md) · [8-token 출처](../r19_plan/JHH5_OFF_POSITION_PROTOCOL.md) · [State 출판사 기록](https://www.sciencedirect.com/science/article/pii/S0092867426009219) · [PRISM 보관본](https://api.figshare.com/v2/articles/9393293)에 있다.

## 저자가 확정할 행정 및 공개 항목

- [ ] **교신저자와 전 저자**가 이름·순서·소속·지속 식별자·교신 정보 및 최종 원고를 확인한다. 개념화, 방법론, 소프트웨어, 검증, 자료 관리, 초안, 검토 및 수정 등 각 사람의 CRediT 역할을 합의한다. 감사 대상과 소속기관 표기를 확인한다. [저널 가이드](https://www.nature.com/srep/author-instructions/submission-guidelines) · [CRediT](https://credit.niso.org/contributor-roles/)
- [ ] **교신저자와 연구 책임자**가 연구비 식별자, 지원자 역할, 재정·비재정 이해충돌, 공개 자료의 2차 분석에 해당하는 기관 윤리 절차, 동의 또는 적용 제외 근거를 확정한다. 미실험 파일럿을 완료된 절차로 표현하는 문장이 없는지 확인한다. [편집 정책](https://www.nature.com/srep/journal-policies/editorial-policies)
- [ ] **교신저자**가 방법과 제출 폼의 AI 보조 고지를 승인한다. 원고는 Codex의 설계·구현·실행·분석·작성 보조를 이미 기술한다. 실제 범위와 사람이 검증한 내용을 확인하고 도구를 저자로 등록하지 않는다. [Scientific Reports AI 정책](https://www.nature.com/srep/journal-policies/ai)
- [ ] **데이터·코드 책임자**가 버전 고정 공개 저장소를 정한다. 허용 범위에는 원본 다운로드 스크립트, 실행 코드·환경, 분할 manifest, 허용된 집계·그림·통계 재생 패킷을 포함할 수 있다. 재현 README와 데이터·코드 접근 문구를 맞춘다. 공개 전 익명 요약의 재식별 및 원본 라이선스 조건을 검토한다. 리뷰 ZIP은 내부 검토용으로 승인 대기 중이다. [리뷰 재생 범위](../r19_plan/manuscript/REPLAY_SCOPE.md) · [저널 공개 정책](https://www.nature.com/srep/journal-policies/editorial-policies)
- [ ] **데이터 책임자**가 CLUE 및 GEO 자산을 파일별로 귀속한다. NIH 등 공공 자금의 공개 L1000 배포는 해당 GEO 출처 인용과 전처리 차이 표기를 요청한다. CLUE 등록 사용자의 제한 파일, 키와 코드에는 별도 사용·재배포 조건이 있으며, CMap 2020 Level 3 원본처럼 출처와 공개 범위 확인이 필요한 자산은 원본 링크와 획득 절차를 제공한다. 재배포하려는 파생 데이터의 허용 여부를 확인한다. [CLUE 재배포 안내](https://clue.io/connectopedia/data_redistribution) · [CLUE 릴리스·이용 조건](https://clue.io/releases/data-dashboard)
- [ ] **데이터 책임자**가 정확한 DepMap/PRISM 릴리스, Figshare 버전과 파일별 출처를 고정한다. PRISM Repurposing 19Q4 v4에는 CC BY 4.0이 표시되어 있어 귀속, 버전, 변경 사항을 남긴다. DepMap 포털의 외부 기탁 자료는 각 자료의 별도 약관을 확인한다. LINCS Cell Painting 처리 프로필과 소스 코드는 별도 라이선스를 적용한다. [DepMap 인용 정책](https://depmap.org/portal/data_page/) · [PRISM v4 메타데이터](https://api.figshare.com/v2/articles/9393293) · [Cell Painting 자료와 코드 라이선스](https://github.com/broadinstitute/lincs-cell-painting#license)
- [ ] **교신저자**가 표·벡터 그림·보충자료·자료 접근 문구·연구비·이해충돌을 반영한 최종 PDF에서 모형 범위와 단위를 승인한다. [제출 점검](https://www.nature.com/srep/author-instructions/ready-to-submit)

## T4 v2 편집 선택

**권고는 rank-16을 Paper 1 본문에 유지하고 T4 v2를 별도 후속 노트와 도구에 두는 것이다.** Paper 1의 같은 입력과 대조군에 대한 주장과, T4 v2에서 추가한 내부 튜닝 및 Pan-LINCS 사전학습의 근거를 각각 명확히 전달하기 쉽다. 후속 결과를 본문에 한 문장 인용할 경우 기존과 같은 평가군을 사용한 등록 후속 분석이라고 표기한다. [T4 v2 결과](https://github.com/honbul/vcell_liver/blob/aa4112e4079a481d1893f7e987ae9022838f5c32/reports/tune_t4_v2/RESULTS.md)

| 선택 | 장점 | 제출 전 작업량 |
|---|---|---|
| Paper 1에 통합 | 같은 HEPG2 132 parent·268웰과 같은 6시간 입력에서 rank-16 RMSE .90779 → .89435, 상대 개선 1.48% [99% .53, 2.39], R² +1.78%p [.19, 3.41]의 후속 결과를 읽을 수 있다. | 중간 이상. 초록·Methods·Results·Discussion·표 1·Fig. 1·CLAIMS·DOCX를 다시 맞춘다. Pan-LINCS 29,827쌍의 출처·제외 규칙과 약 1.5% 개선의 절대값을 명시한다. PHH 전체 적합과 제로샷·5/10/20개 적응을 별도로 보고한다. 새 재현 범위·권리·영수증을 추가한다. |
| 별도 후속 노트와 오프라인 도구 | Paper 1은 비교 대상과 해석이 일관되고, 후속 노트는 선형 성분, 다세포주 학습과 PHH 음성 결과를 충분히 검토할 수 있다. 현재 CLI·모델 카드·채점 원표를 활용할 수 있다. | 본 원고는 낮음. 인용문 한 줄과 버전 링크를 검토한다. 후속 노트는 별도 공개 범위와 참조환경을 승인하고, 가능하면 추가 독립 plate에서 확인한다. |

PHH 전체 적합의 동등 scaffold rank-16 대비 RMSE 개선은 1.95% [99% .07, 3.90]였다. R² 차이 +2.81%p [−.88, 7.03]의 구간은 0을 포함했다. 5·10·20개 적응은 모두 동등 공통 배율보다 오차가 높았다. 보정한 별도 24시간 실측 반복은 HEPG2에서도 더 정확했다.
