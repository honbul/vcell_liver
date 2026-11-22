# SNU-423의 0.5→5 µM 동일 약물 전이 지원

**물리적으로 분리된 저용량 입력은 사전 지원 기준에 미달했다.** 5 µM 평가 화합물 203군 모두에서 같은 구조 entity의 0.5 µM 등록이 있었다. 적격 세포 수와 조합을 충족하는 다른 plate donor는 201군이었지만, release의 위치 인덱스까지 달라야 한다는 조건은 3군·6개 목표 웰만 만족했다. 같은 plate의 다른 약물 대조는 그 6웰 모두에 있었다. 5 µM 평가 pool당 최소 3 parent까지 적용한 후 적격 목표는 0군이었다. [사전 명세](HCC_CROSS_DOSE_DONOR_PROTOCOL.md) · [단계별 지원 원표](data/hcc_cross_dose_donor/diagnosis/support_funnel.tsv)

| 5 µM 평가에 추가한 조건 | Fold 0 | Fold 1 | Fold 2 | 전체 parent | 전체 목표 웰 |
|---|---:|---:|---:|---:|---:|
| 원 P4 5 µM 지원 | 67 | 69 | 67 | 203 | 275 |
| 같은 0.5 µM 구조 entity | 67 | 69 | 67 | 203 | 275 |
| 세포주 지원 및 다른 plate | 66 | 68 | 67 | 201 | 273 |
| 다른 release 위치 인덱스 | 1 | 0 | 2 | 3 | 6 |
| donor plate의 다른 약물 대조 | 1 | 0 | 2 | 3 | 6 |
| 평가 pool 최소 3 parent | 0 | 0 | 0 | 0 | 0 |

전체 평가 50군, fold별 평가 10군 및 적합 30군이 실행의 사전 기준이었다. 최종 적합 지원은 fold별 0·3·0군이었다. 따라서 양쪽 발현의 추출과 배율 적합, 예측과 채점을 시작하지 않았다. 확인된 병목은 화합물 등록 수나 음성 대조 유무보다 같은 약물의 위치 배치였다. 이 감사는 release 위치 인덱스를 사용했고 실제 배양 블록의 독립성을 확인하지 못했다. 다른 위치에서의 저용량 RNA 전이 성능은 이 자료에서 판정할 수 없다. [고정 지원표](data/hcc_cross_dose_donor/audit/support.tsv) · [남은 적합 웰 연결](data/hcc_cross_dose_donor/audit/links.tsv)

명세 `6c7a796`, G1 메타데이터 감사 `31458`, 독립 조인으로 단계별 지원을 확인한 작업 `31459`다. 두 작업이 종료 코드 0으로 끝났고 원 P4 동결 및 새 지원 원표의 해시를 검증했다. 발현·봉인·예약 자료에는 접근하지 않았다. [감사 영수증](data/hcc_cross_dose_donor/audit/receipt.json) · [검산 영수증](data/hcc_cross_dose_donor/diagnosis/receipt.json) · [작업 원장](data/hcc_cross_dose_donor/jobs.tsv)
