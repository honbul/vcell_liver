# 세 용량 공동 구조 커널로 SNU-423의 5 µM 반응 예측

이 비교는 P4와 SNU-423의 반복·scaffold 결과를 보고 등록한 DEV 후속이다. 24 h, 최소 20세포, P4 203 parent outer fold 및 각 fold의 5 µM 적합 유전자 2,000개를 유지한다. 주 평가는 SNU-423 5 µM의 웰 내 세포주 상대 반응이다. 각 용량의 10세포주·3 parent pool 적격성, 웰 평균 및 원 pool의 parent 가중 중심화를 적용한다. 평가 parent의 모든 용량에서 처리 RNA를 제외하고 구조만 입력하는 A 계약이다.

- 각 fold의 적합 역할 A 또는 B로 0.05·0.5·5 µM 웰의 반응을 만든다. 5 µM 적합 parent별 사전순 첫 entity를 대표로 고정하고, 같은 entity·세포주가 관측된 용량만 공동 학습에 쓴다. 적합 웰을 parent×용량별 균등 평균한다. query는 원 P4의 각 entity 구조를 그대로 사용한다. 덜 관측된 parent는 있는 용량만 학습에 넣는다.
- `ECFP-5only`는 5 µM만 보는 Tanimoto Ridge이고, `ECFP-dose-joint`는 Tanimoto와 측정 용량 간 공분산의 곱을 쓰는 Ridge다. 정규화한 용량 상관은 세 용량이 모두 있는 적합 parent의 유전자 반응에서 추정한다. 용량별 적합 평균을 빼고 모든 parent와 유전자를 합해 상관을 계산한 뒤 I와 1:1로 섞는다. λ=1, 출력 배율 1, 절편 0을 모든 arm에 고정한다. 내부·외부 정답으로 설정이나 배율을 고르지 않는다. 완전 parent <20 또는 분산 0이면 I로 돌아간다. query 라벨과 위치는 predictor에 입력하지 않는다.
- 같은 fit doses·용량 상관·λ를 쓰되 기존 ChemBERTa 특징의 정규화 코사인을 구조 커널로 쓰는 `ChemBERTa-dose-joint`로 입력 정보와 모델 용량을 맞춘다. 이는 용량을 공동 적합하는 ChemBERTa 비교군이다. `ECFP-dose-shuffled`는 동일 용량·parent 지원과 원 용량 상관을 유지하면서 두 저용량 적합 반응의 parent 대응을 각 용량 안에서 고정 seed 순환 치환한다. P4의 동결 B0, 형제 복사 및 최선 기존 VC의 참고 점수도 같은 행에서 재채점한다. 기존 VC 동결 비교에서는 과거 각 모델의 학습 계약과 정보가 다를 수 있다.
- 반대 역할 DMSO로 별도 채점한다. parent macro contrast RMSE 및 pooled R², 10,000회 paired parent 지수 재표집, 재표집마다 pool 중심화를 적용한다. 주 통과 조건은 `ECFP-dose-joint` 대 5 µM 단독·같은 용량 입력 ChemBERTa·저용량 shuffle 및 P4 동결 최선 VC의 네 비교가 모두 RMSE 차이 99% 상한 < 0 및 R² 차이 99% 하한 > 0이다. 동결 최선 VC는 재표집과 지표별로 다시 구한다. B0·형제 복사와 기존 181 scaffold component 재표집은 기술 비교다. 낮은 Tanimoto(<0.7) 지원의 고정 예측 점수도 기술 분석으로 제공한다. 기존 P4 판정과 이 효과를 독립 성인 간암 확인으로 승격하지 않는다.

사전 명세 → fit-only 예측 → SHA-256 동결 → 별도 채점이다. 이전 공개 Tahoe 개발 웰 및 기존 구조 특징만 사용한다. 예약·봉인·격리 웰 및 외부 RNA는 읽지 않는다.
