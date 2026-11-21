# X3. 같은 약물의 다른 세포주 측정을 쓰는 DEV B 비교

X에서 잠근 HEPG2 10 µM·6 h 493개 화합물과 정확히 같은 물리 score 웰을 사용한다. HEPG2의 해당 fold에서 다룰 query 약물별 A375·A549·HT29·MCF7 처리 반응을 B 전용 묶음으로 읽는다. 각 donor 반응은 자체 물리 plate의 fit DMSO만 뺀 뒤 donor 세포주 안에서 평균한다. X의 보호 처리·별칭·score 역할·donor 제외 명부와 978 랜드마크 축을 재사용한다. HEPG2 처리·score DMSO 값은 예측 동결 뒤 별도 프로세스에서 읽는다. [X 물리 지원](data/x_a/access/support.tsv)

`B_cell`은 네 donor의 균등 평균이다. `X3_similarity`는 source fit 세포에서만 표준화·PCA를 추정한 DepMap 기저 RNA 좌표의 거리로 donor를 가중한다. 온도 `.5, 1, 2`를 후보로 둔다. `X3_linear`는 source fit parent의 세포주별 반응에서, 그 세포의 반응과 나머지 donor 평균 차이를 학습한다. 두 설명항은 source 세포와 donor 간 기저 좌표 차이와, 그 차이에 donor 평균 반응을 곱한 값이다. Gene별 ridge `.1, 1`을 사용한다. 각 모델의 배율은 같은 `.0, .125, .25, .5, .75, 1` 격자에서 고른다.

내부 선택의 pseudo-target은 A375다. 그 안쪽 parent는 공유 파라미터 추정에서 제외하고, A549·HT29·MCF7의 같은 약물 처리값은 B 입력에만 사용한다. A375의 독립 select DMSO를 뺀 R17 손실로 설정과 배율을 한 번 선택한다. HEPG2 외분할에서는 세포와 parent fold를 모두 fit에서 제외해 다른 네 세포주에 재적합한다. 각 방식의 held donor forecast를 같은 seed 1501로 parent 단위 순열한 값을 화합물 대응 검증에만 사용한다.

주 비교는 `X3_similarity`·`X3_linear` 각각과 `B_cell`의 짝 parent-macro RMSE 및 pooled R²다. Parent bootstrap 10,000회 99% 구간과 fold별 결과를 기록한다. 기존 DEV 운영 문턱인 RMSE 상대 개선 0.5%, R² 개선 0.5%p와 같은 방향 fold 2개를 적용한다. 기존 X의 A 입력 비교와 별도 정보 조건이며 같은 코호트의 DEV 후속 탐색이다.
