# 동일 약물·농도에서의 CTRPv2–PRISM 실측 일치도

약물별 평균을 새로 보정했을 때도 PRISM 전이가 개선되지 않았다. 여기서는 서로 다른 약물 곡선을 요약한 AUC를 공통 물리 농도 범위에서 다시 비교한다. 이 진단은 기존 학습 코어의 같은 세포·같은 parent 약물에 한정한다. [평균 감수성 보정 결과](ANCHOR_RANK_CV_RESULTS.md)

- GDSC2·CTRPv2·PRISM 등록 자료의 약물·세포 ID와 QC를 고정한다. 0.1–1 µM에서 로그 농도를 적분한 정규화 AUC와 1 µM 생존율을 측정한다. CTRPv2는 post-QC 측정치에서 로그 농도 선형 보간을 사용한다. PRISM은 20Q2의 공개 농도표와 배포된 4파라미터 곡선을 사용한다. 두 경우 모두 생존율 범위를 0–1로 제한한다. 괄호 범위의 양끝을 포함하는 지원만 등록한다. [CTRP 공개 리소스](https://portals.broadinstitute.org/ctrp.v2.1/) · [PRISM 20Q2](https://doi.org/10.6084/m9.figshare.20564034) · [PRISM 곡선 코드](https://github.com/broadinstitute/repurposing/blob/master/depmap_reprocessing_script_for_secondary.R)
- PRISM 곡선은 해당 출판 농도 범위에서 native AUC를 ±.005 이내로 재현한 경우에만 연결한다. 계산된 새 측정과 출판 AUC를 같은 세포·약물에서 대조하며 계통당 평균을 쓴다. 약물 내 상관은 ≥30계통, 계통 내 상관은 ≥5약물인 그룹에서 모두 같은 지원을 사용한다. GDSC2는 8.5의 등록 농도 범위에서 지원 수만 기록한다.
- 농도 표준화의 독립적인 기여를 보기 위한 자료 진단이다. 서로 다른 배양 시간·측정 기술과 CTG/바코드 생존율 차이의 영향이 함께 남는다. LIMORE와 오가노이드의 실측은 이번 입력에 포함되지 않는다.
