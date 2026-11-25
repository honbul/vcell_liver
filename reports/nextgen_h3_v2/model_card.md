# H3-v2 연구용 감수성 모델

외부 단일 모델은 hybrid, 설정 b2ca48a7e587이다. 모델 checkpoint와 fit-only 특징 정의는 G1 final 및 features에 보관했다.

## 사용 목적과 입력

생체 밖 암 모델의 약물 감수성 순위와 상호작용을 연구한다. 세포 기저 RNA, 경로·TF, 정확 parent의 구조와 ChEMBL 타깃, 관측 시 LINCS 변화를 사용한다. 모델 출력은 스크린 감수성 점수다.

## 평가 범위

독립 seed는 11711, 11729, 11743다. 신규 골격, 신규 세포, 스크린 동시 보류, GDSC1·LIMORE·LICOB에서 평가했다. GDSC1에 소스와 공유하는 세포가 있고 LIMORE와 LICOB은 기노출 코호트다. 지원과 성능 차이는 결과표에 기록했다.

환자 치료 선택, 용량 산정과 임상 효용은 평가 범위 밖이다. 결측 LINCS 지원과 생물학적 플랫폼 차이에서 불확실성이 달라진다.

[데이터와 지원](DATA_SUPPORT_AND_EXPOSURE.md) · [결과](RESULTS.md) · [동결](final_lock.json)
