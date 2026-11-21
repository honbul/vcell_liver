# DEV-ARCH I 뷰 확보 및 지원 결과

R18 예약 측정, 그 서명과 별칭 및 B donor 예외를 제외하는 검사를 R17 개발 처리 웰 6,773개에 적용했다. 겹치는 물리 측정·서명은 0개이며 개발 모집단은 1,862 parent다. 세 뷰 각각 전체 100 parent 및 각 outer fold 20 parent의 기준을 통과했다. [출처·명세](I_VIEW_PROTOCOL.md) · [전체 및 교집합 원표](data/i_views/modality_coverage.tsv)

| 뷰 | 매칭 | Outer fold 0 / 1 / 2 | 범위 |
|---|---:|---:|---|
| I1 ChEMBL 37 | 1,467 / 1,862 (78.8%) | 491 / 495 / 481 | R7 원본을 읽기 전용으로 사용해 정량 activity 146,492행을 확보했다. 활동별 action type이 채워진 행은 9,478개다. 별도 표적 기전 주석을 활동값이 있는 것처럼 채우지 않았다. |
| I2 Cell Painting | 361 / 1,862 (19.4%) | 117 / 132 / 112 | A549 48 h, dose code 6의 물리 plate별 DMSO 정규화 consensus. 공통 morphology feature는 1,781개다. |
| I4 PRISM | 722 / 1,862 (38.8%) | 253 / 241 / 228 | 공개 19Q4에서 간 유래·간 유래로도 주석된 26개 및 tissue를 확인 못한 10개를 제거해 542개 비간계열 세포주로 지문을 만들었다. |
| 세 뷰 공통 | 344 / 1,862 (18.5%) | 112 / 126 / 106 | 두 뷰 이상을 보유한 군은 725개다. |

R18 명부에 같은 parent가 있는 경우까지 제외하는 민감도에서 세 뷰 공통은 248 / 1,564군이고 fold별로 85 / 91 / 72군이다. PRISM 배포본의 treatment-wise ComBat은 출판자 전처리다. 최종 지문에 직접 포함되는 세포주 축이 전부 비간계열임을 별도로 확인했다. [PRISM 세포주 감사](data/i_views/PRISM_cell_line_audit.tsv)

I2·I3·I4를 `PLANNED`로 전환했다. I3의 두 조합에서 내·외 fold를 포함한 24개 PSD 검증의 최소 고윳값은 0.00347, 구조 fallback의 최대 오차는 0이었다. 적합 fold에서만 정한 표준화와 PCA를 적용한 뒤 세 뷰 공통 실사용 지원은 outer fold별 109 / 124 / 105군이며, 동일 R18 parent까지 제외하면 82 / 89 / 70군이다. 이 단계의 증거는 입력 지원과 커널 수치 성질까지다. [I3 수치 검증](data/i_views/kernel_support.tsv) · [fold 전처리 지원](data/i_views/transform_support.tsv)

새 공개 파일은 171,300,016바이트이며, G1의 전체 새 작업 디렉터리는 감사 시점에 185,288,346바이트였다. 모든 공개 원본은 SHA-256과 연결된 라이선스를 기록했다. ChEMBL의 기존 30 GB SQLite도 별도 SHA-256을 기록했다. 원본과 feature는 `g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_dev_i/coverage.bk00jUHu`에 보존했다. [공개 원본 명부](data/i_views/public_source_manifest.json) · [ChEMBL 영수증](data/i_views/I1_receipt.json) · [작업 원장](data/i_views/jobs.tsv)
