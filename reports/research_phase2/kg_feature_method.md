# KG의 유전자–경로 관계를 세포별 특징으로 바꾸는 방법

KG는 발현량을 제공하지 않는다. 외부 Reactome v92의 `유전자 member_of 경로` 연결을 제공하고, 세포별 값은 해당 세포의 관측 raw count에서 계산한다. 원 구현은 `scripts/05_kg_features_ablation.py`의 `proxies()`다. GNN이나 graph embedding을 학습한 방식은 아니다.

세포 c, 유전자 g의 count를 x라고 할 때 다음 순서를 따른다.

1. 세포별 전체 관측 library sum으로 10,000 정규화 후 log1p: `l[c,g] = log(1 + 10000*x[c,g]/sum_g x[c,g])`.
2. 학습 세포에서만 유전자별 평균 μ와 표준편차 σ를 구한다. 평가 세포에도 같은 μ/σ를 적용한다. σ가 1e-6보다 작으면 분모를 1로 둔다.
3. `z[c,g] = (l[c,g] - μ[g])/σ[g]`를 계산한다.
4. 경로 P에 속하며 데이터에서 관측되는 유전자 집합 G(P)에 대해 `f[c,P] = sum(z[c,g] for g in G(P))/len(G(P))`를 계산한다. 모든 유전자의 가중치는 같다.

예를 들어 경로 하나에 속한 세 유전자의 z-score가 한 세포에서 1.2, 0.6, −0.3이면 그 경로 특징은 0.5다. 다른 세포에서는 같은 연결을 쓰더라도 발현값이 다르므로 다른 숫자가 나온다. 이는 학습 집단 대비 상대 발현 proxy이며 실제 pathway flux, 활성화 판정 또는 인과 효과가 아니다. 저발현 유전자의 음수 값도 평균에 포함한다.

행렬로는 `세포×유전자 Z`와 `유전자×경로 W`의 곱 `F=ZW`다. Membership이 있을 때 W[g,P]=1/|G(P)|, 없으면0이다. 구현은 큰 dense Z를 만들지 않고 sparse count 행렬에 같은 연산을 적용한다.

V1에서는 관측 유전자가10–200개인 경로를 학습 특징 분산으로 정렬하고, 경로 수32/64/128을 validation에서 선택했다. 10k 비교에서는128개가 선택되어 `scGPT512차원 + pathway128차원 = 640차원`을 같은 분류 head family에 넣었다. 최종 head의 scaling과 C/class_weight도 학습/validation 범위에서 정했다. Test 라벨은 선택에 사용하지 않았다.

대조는 유전자별 연결 수와 경로별 유전자 수를 모두 보존하는 edge swap(seed42)으로 만든다. 단순히 차원이 늘어난 효과와 실제 membership 정보의 효과를 비교한다. V1의 실제10k test F1은 scGPT .9527, 실제 경로 concat .9436, matched random concat .9531이었다. 이 조건에서 KG 추가 이득은 없었다. 이는 두 legacy 환자 결과이며 V2 blind test가 아니다.

V2의 세 번째 데이터 가설은 별도 실험이다. Geneformer768차원과 같은4096개 학습 셀을 유지하고, 학습 전처리의 실제128경로를 PCA32로 압축했다. Random membership도 같은32차원으로 압축했으며4개 fold·seed42에서 비교했다. Real32 F1 .94659, raw128 .94696, random32 .94776으로 압축 가설을 지지하지 않았다. 사전 고정한 외부8개 후보 선택에는 이 탐색을 사용하지 않았다. 실제 수치는 `kg_compression_diagnostic/`에 있다.

사용자 제공 리뷰 PDF는 representation과 실제 intervention 검증을 구분하는 설계 참고자료다. 위 pathway 평균 및 압축 실험은 저장소의 구현 선택이며, 논문의 특정 KG 알고리즘을 그대로 재현했다고 주장하지 않는다.
