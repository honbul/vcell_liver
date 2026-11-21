# 정확한 농도에서의 시간 전이와 교차 assay 결과

**LINCS 안에서 관측한 24시간 방향 일치는 동일 약물·농도의 HepaRG에서 거의 0이었다.** 기존 HEPG2의 6·24 h와 GSE284321·GSE296254의 24 h를 정확한 canonical parent와 실측 µM으로 맞췄다. 공통 유전자 827개, 기존 T4 예측 그리고 과거 연구에 노출된 HepaRG 자료를 사용했다. [명세](T4_HEPARG_PROTOCOL.md) · [지원 감사](data/t4_heparg/audit/support.tsv)

| 24시간 HepaRG 대상 | 일치 parent / HepaRG 물리 웰 | LINCS 직접 6 h | LINCS 6→24 h 연산자 | LINCS 직접 24 h |
|---|---:|---:|---:|---:|
| GSE284321과 동일한 LINCS 내부 지원 | 5 / 21 | 0.174 | 0.247 | 0.385 |
| GSE284321 HepaRG | 5 / 21 | 0.020 | 0.024 | 0.022 |
| GSE296254와 동일한 LINCS 내부 지원 | 3 / 9 | 0.135 | 0.220 | 0.329 |
| GSE296254 HepaRG | 3 / 9 | −0.005 | −0.017 | −0.011 |

표는 같은 물리 처리 웰 반복과 대조군 역할을 parent 안에서 평균한 뒤 parent에 같은 가중치를 준 평균 cosine이다. 그룹별 3–5군을 기술 비교했으며 외부 확인의 불확실성을 추정할 규모는 아니다. GSE296254는 3군 모두 배양·plate·위치가 다른 반복을 확보했으며 세 화합물 모두 같은 outer fold에 있었다. GSE284321에는 배양 ID가 없어 독립 배양을 검증하지 못했다. PHH 또는 HUH7에서 동일한 농도로 두 LINCS 시간과 HepaRG 24 h를 모두 연결할 수 있는 parent는 0군이었다. [cosine 및 범위](data/t4_heparg/score/cosine.tsv) · [parent별 원표](data/t4_heparg/score/by_parent.tsv)

HepaRG에서 낮은 방향 일치는 HEPG2와 HepaRG의 세포 맥락 변화, L1000과 TempO-Seq의 측정법 변화, 그 상호작용을 함께 포함한다. 관측치를 시간·측정법·세포 맥락의 개별 원인으로 나눌 수 없다. HepaRG에서 지표를 고르거나 새로운 모델을 적합하지 않았으며 직접 24 h donor 역시 같은 작은 집단에서 확인했다.

## Open TG-GATEs의 PHH microarray 24시간 지원

공유 저장소의 CEL 속성 표에서 24 h human in vitro 처리 **157화합물·785배열**, control **314배열**을 확인했다. 몰 농도가 직접 보고된 처리는 146화합물·720배열이었다. 원 T4의 PHH 6·24 h 정확 농도 목록과 농도만 비교하면 14화합물·44배열이었다. 같은 이름으로 유일한 LINCS parent에 대응 가능했던 10화합물·32배열 중 같은 parent·농도에서 적격 PHH 6·24 h 쌍과 연결된 것은 0이었다. 이름은 후보 매칭에 썼으며 별칭과 질량 단위까지 포함한 전체 가능성은 이 집계에서 결정하지 않았다. 공개 표의 실험·그룹·표본·GeneChip ID로는 물리 plate의 독립 DMSO 역할을 검증할 수 없었다. 따라서 이 명부에서 PHH 교차 assay 채점은 지원이 부족했다. [감사 원표](data/t4_tggates/support.tsv) · [공식 CEL 속성 설명](https://dbarchive.biosciencedbc.jp/en/open-tggates/data-4.html)

HepaRG 감사 `31151`에서 QC read count의 문자열 `2637825.0`을 정수로 변환하다 중단했다. 같은 값의 수치 해석을 고친 재실행 `31152`와 TG-GATEs 메타데이터 감사 `31153`이 완료됐다. HepaRG 비교의 명세 커밋은 `d8a53ed`이며 예측 `31154` → 동결 `31155` → 별도 채점 `31156`은 모두 종료 코드 0이었다. 신규 다운로드와 R18 예약 발현 접근은 없었다. [동결](data/t4_heparg/freeze.json) · [채점 영수증](data/t4_heparg/score/receipt.json) · [G1 실행 원장](data/t4_heparg/jobs.tsv)
