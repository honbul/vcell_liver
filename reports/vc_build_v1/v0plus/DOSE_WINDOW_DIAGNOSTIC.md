# 동일 약물·농도에서의 CTRPv2–PRISM 실측 일치도

약물별 평균을 새로 보정했을 때도 PRISM 전이가 개선되지 않았다. 여기서는 서로 다른 약물 곡선을 요약한 AUC를 공통 물리 농도 범위에서 다시 비교한다. 이 진단은 기존 학습 코어의 같은 세포·같은 parent 약물에 한정한다. [평균 감수성 보정 결과](ANCHOR_RANK_CV_RESULTS.md)

- GDSC2·CTRPv2·PRISM 등록 자료의 약물·세포 ID와 QC를 고정한다. 0.1–1 µM에서 로그 농도를 적분한 정규화 AUC와 1 µM 생존율을 측정한다. CTRPv2는 post-QC 측정치에서 로그 농도 선형 보간을 사용한다. PRISM은 20Q2의 공개 농도표와 배포된 4파라미터 곡선을 사용한다. 두 경우 모두 생존율 범위를 0–1로 제한한다. 괄호 범위의 양끝을 포함하는 지원만 등록한다. [CTRP 공개 리소스](https://portals.broadinstitute.org/ctrp.v2.1/) · [PRISM 20Q2](https://doi.org/10.6084/m9.figshare.20564034) · [PRISM 곡선 코드](https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R)
- PRISM 곡선은 해당 출판 농도 범위에서 native AUC를 ±.005 이내로 재현한 경우에만 연결한다. 계산된 새 측정과 출판 AUC를 같은 세포·약물에서 대조하며 계통당 평균을 쓴다. 약물 내 상관은 ≥30계통, 계통 내 상관은 ≥5약물인 그룹에서 모두 같은 지원을 사용한다. GDSC2는 8.5의 등록 농도 범위에서 지원 수만 기록한다.
- 농도 표준화의 독립적인 기여를 보기 위한 자료 진단이다. 서로 다른 배양 시간·측정 기술과 CTG/바코드 생존율 차이의 영향이 함께 남는다. LIMORE와 오가노이드의 실측은 이번 입력에 포함되지 않는다.

## 결과

두 리소스에 정확히 연결되는 세포·약물 쌍은 4,194개였다. 농도 구간과 native AUC 복원을 모두 통과한 공통 지원은 **11약물, 3,389개 세포·약물 쌍, 계통 내 중복을 합친 3,359개 관측**이었다. 이 행에서 세 방법을 비교했다.

| 같은 지원의 소스 간 실측 | 약물 내 Spearman | 약물 내 Pearson | 계통 내 Spearman | 계통 내 Pearson |
|---|---:|---:|---:|---:|
| 기존 native AUC | .204 | .248 | .656 | .724 |
| 공통 0.1–1 µM AUC | .169 | .243 | .585 | .709 |
| 1 µM 생존율 | .195 | .246 | .570 | .593 |

약물 내 상관의 지원은 11개 약물, 계통 내 상관의 지원은 398개 계통이었다. 테스트한 농도 통일에서는 두 Spearman이 모두 낮아졌다. 이는 같은 농도를 쓰면 CTRPv2–PRISM의 일치가 개선된다는 가설을 지지하지 않는다. 같은 농도로 제한할 때 관측의 변동 폭, assay와 투약 시간, 곡선 근사 차이의 기여는 이 표에서 식별할 수 없다.

PRISM에서 공개 dose와 유한한 곡선 적합값이 모두 있는 원자료 4,269행 중 3,865행이 native AUC ±.005를 재현했다. 절대 오차 중앙값은 1.18 × 10⁻⁶, 90백분위는 .00154였다. GDSC2 기존 등록에서는 33약물·19,557행이 같은 농도 범위를 덮었다. 이 숫자는 기존 적합표의 농도 범위 메타데이터에서 얻었다. [해시·지원·원표](dose_window_diagnostic.json)

### GDSC2 원자료 확보

같은 8.5 릴리스의 공식 GDSC2 원자료 ZIP 122,866,072바이트를 G1에 확보했다. 내부 데이터 CSV 2,112,924,313바이트와 함께 실린 필드 설명 PDF가 ZIP CRC를 통과했다. 원자료 CSV는 Sanger 모델 ID, 플레이트 바코드와 scan, 위치, control·treatment 태그, 약물 ID, µM 농도, 강도를 포함한다. 저자 문서는 같은 물리 웰에 복수 태그가 붙을 수 있음을 명시한다. ZIP SHA-256은 `45dca6c707dcb8b80e73764adc91aeabe9c620af4a3f1f87648c80c35cd74d1b`다. 현재 확인 범위는 배포본 무결성 및 원자료 스키마이며, 이 문서의 공통 농도 실측 표는 위 두 소스를 다룬다. [Sanger 공개 다운로드](https://cog.sanger.ac.uk/cmp/download/GDSC2_public_raw_data_27Oct23.zip) · [Sanger 실험 문서](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/)

첫 Slurm 작업 `32845`은 래퍼의 인자 조립 오류로 Python 모듈을 시작하기 전에 종료했다. 인자를 수정한 `3e0e6dc`에서 동일 명세로 `32848`을 완료했다. [원장](RUN_LEDGER.tsv)

GDSC2 원자료를 플레이트 대조 웰로 정규화해 추가한 후속 결과가 있다. [세 소스 공통 농도 진단](GDSC_RAW_WINDOW_RESULTS.md)
