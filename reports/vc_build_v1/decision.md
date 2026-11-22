# Practical HCC VC 구축 결과

**이번에 만든 것** — 88약물의 LIMORE 72시간 Activity Area 감수성 예측기 V0, 체크포인트와 오프라인 CLI, 실제 상태 프로그램 모델, 공개 외부 평가와 보정, 로컬 교차 실험 승인안.

**새 입력 실행** — 실제 기저 RNA로 100개 후보를 시험해 등록 약물 88개에 예측·구간을 반환하고 미등록 12개를 보류했다. 운영 범위는 이 공개 실험의 curve summary다.

**가설 판정** — H1 실용 효과 기준 미통과, H2의 상태 보조 학습은 방향 기준을 충족하고 5% 실용 목표에 미달, H3 현지 실측 대기. 성인 HCC 상태 출력 승격은 미통과다.

**가장 강한 비교군과 검증 범위** — LIMORE cell-cold 개발에서는 MLP가 최선 등록 비상호작용 비교군 대비 MAE를 3.83% 줄였다. 별도로 세 외부 소스에서 학습한 Direct는 LIMORE 신규 50개 계통에서 약물 평균 대비 NDCG@5를 0.03081 높였고 전체 순위 Spearman은 0.03194 낮았다.

**다음에 필요한 승인** — VCB-06의 실제 보유·인증 모델, 블라인드 명부, assay·농도와 견적을 실험실이 확인한 후 집행 승인. 0h RNA와 24h·72h 실측을 대응시켜 V1과 V2를 별도 판정한다.

## 성적표

| 항목 | 결과 | 근거 |
|---|---|---|
| VCB-00 데이터·실측 실행 | 완료. 77 성인 HCC 모델, 74 계통, 88약물, 6,776실측. 작은 checkpoint·동결·채점 확인 | [수집 감사](data_function/VCB00_LIMORE.md) |
| VCB-01 내부 및 독립 학습원 외부 | 완료. LIMORE C, D, CD 분리. GDSC2·CTRPv2·PRISM 학습 → 별개 LIMORE 50계통 × 68약물. 외부 상위 순위 이득과 전체 순위 손실 | [기능 성적](function/RESULTS.md) · [외부 및 보정](PUBLIC_EXTERNAL_RESULTS.md) |
| VCB-01 표적·경로·변이/CNV | 완료. 동등 지지의 직접 모델 대비 명확한 추가 MAE 이득 없음. 외부 표적 발현의 추가 NDCG 차이도 0을 포함 | [분자 비교](function/MOLECULAR_RESULTS.md) |
| VCB-02/03 24h 상태와 세포 고유 효과 | 완료, 승격 미통과. JHH-5/JHH-7의 교차 95약물에서 세포 차이 RMSE가 +2.14% 악화. 반복이 부족하고 기본 상태 입력은 독립 기저 발현 대리값 | [상태 성적](state/RESULTS.md) |
| VCB-04 고정 Direct/Multi-output/State-assisted | 완료. 동일 LIMORE 기능 분할 74계통에서 State-assisted MAE +1.33% [95% +0.18%, +2.51%] 개선. 상대 5% 목표 미달 | [통합 성적](JOINT_RESULTS.md) |
| VCB-05 공개 소스 이동 | 완료. 15/10/25 계통으로 적합/구간/평가 분리, 평가 약물과 겹치지 않는 5 sentinel 사용. MAE 개선 불확실, 구간이 넓음. 현지 대응 실측은 자료 대기 | [보정 성적](PUBLIC_EXTERNAL_RESULTS.md#공개-소스-보정) |
| VCB-06 로컬 bridge | 설계 완료. 24약물 실제 가명부, 12 RNA 약물, 교차 배치, 수량별 견적 요청·승인안. 지역 모델 명부와 비용은 실험실 확인 대기 | [승인안](PILOT_APPROVAL_DRAFT.md) · [후보 명부](pilot/provisional_compound_slate.tsv) |

## 사용할 수 있는 기능

V0은 공개 LIMORE assay의 후보 우선순위를 반환한다. NDCG@5는 연속 관측 순위 품질을 요약한다. H1의 MAE 5% 기준, 정밀한 외부 AA 척도, 검증된 성인 HCC 상태 예측, 실제 top-5 hit와 가격 대비 효용은 각각 해당 지표와 실측에서 확인해야 한다. 100개 후보 추론의 CPU 관측은 약 3초이며, 범위 밖 질의는 보류한다. 공개 자료의 과거 노출과 기저 대리 입력은 명시했다. OEP003191 공개 접점과 HRA004348 데이터 등록 경로를 확인했고, 환자 유래 행렬의 재사용과 교집합은 확인 대기다.

[모델 카드와 실행 예시](model_card.md) · [관측 지표 요약](metrics.json) · [공개 오가노이드 접근 감사](PUBLIC_ORGANOID_ACCESS.md)
