# 간·간암 Virtual Cell 후속 연구기획서

## R17 교정 결과 확인과 shift·context·구조·손실의 분리 검증

- 작성일: 2026-11-21, 한국시간 기준.
- 검토 저장소: [honbul/vcell_liver](https://github.com/honbul/vcell_liver).
- 검토 고정점: `main f1c087a6da365a6aebac5315ced640aa1a156692` — R17 결과·재고·R18 요청을 포함한다.
- 검토 입력: 첨부 「간암 Virtual Cell 아키텍처 설계」 PDF 39쪽, 최신 결과 문서·코드·작은 점수표, 핵심 방법의 공식 자료.
- 문서 성격: **검토 결과를 반영한 실행 제안서**. 이 문서를 작성하면서 재학습, 새 채점, 저장소 수정·push, 봉인 자료 개봉, 대용량 데이터 수집은 하지 않았다.
- 대상 독자: 연구 책임자와 후속 실행을 맡을 AI agent. 아래 실험 ID는 새 제안의 작업 식별자이며, 기존 R18 또는 이후 연구번호를 임의로 대체하지 않는다.

> 권고: **R17의 커널+FiLM을 고정 확인에 보내고, 별도 개발 트랙에서 네 축을 각각 최소한 한 번씩 제대로 검증한다.** 하나가 실패했다고 나머지를 중단하지 않는다. 반대로 여러 최신 모델을 한꺼번에 결합한 뒤 결과의 원인을 설명하지 못하는 실험도 반복하지 않는다.

빠르게 판단하려면 **0·1·20절**, 첨부 PDF의 수정 근거는 **2·3·6절**, 실행 설계는 **7–18절**, agent에게 전달할 요청은 **21절**을 읽는다.

## 0. 먼저 읽을 결론

### 0.1 지금까지 “된 것”과 “안 된 것”

| 질문 | 현재 답 | 다음 행동 |
|---|---|---|
| 모든 시도가 실패했나? | 아니다. 기존 recipe의 음성 결과와 데이터 부족, 평가 규약 문제, 제한적 양성 결과가 섞여 있다. 특히 R17 결합은 양성이다. | 연구 전체가 아니라 개별 주장과 조건을 판정한다. |
| 지금 가장 유망한 후보가 있나? | **R17 커널+FiLM**. 구조 입력 A와 동일 약물 외부 RNA 전이 B에서 기존 비교군보다 좋아졌다. | 기존 요청 R18의 주 후보로 유지한다. |
| 간암 반응 예측을 입증했나? | 아니다. R17은 이미 본 HEPG2 6 h 개발자료의 교정 분석이다. 성인 HCC 일반화 근거가 아니다. | 시간·용량·세포주가 다른 확인과 독립 HCC 자료 적격성을 구분한다. |
| 외부 전이는 크기만 맞추면 해결되나? | 아니다. 기존 외부 예측은 방향 정보도 약하다. 스칼라만 바꾸는 경로의 개선 여지는 제한적이다. | 크기, 방향, 시간, 플랫폼을 분리해 검증한다. |
| shift 다음의 context·구조·손실도 할 것인가? | **한다.** 각각 독립된 최소 실험과 종료 기준을 둔다. | 8절의 필수 최소 행렬을 완료한다. |
| 첨부 PDF 전체를 그대로 실행해도 되나? | 권하지 않는다. 유용한 아이디어와 과도한 성공 가정·자료 계약 혼동이 함께 있다. | 다중뷰·잔차화·저차원 출력·대조 손실은 유지하되 식별 가능한 작은 비교로 바꾼다. |
| 지금 대형 flow와 환자 스크리닝까지 바로 갈 것인가? | 아니다. 분포 생성의 추가 필요성, 독립 반복, 입력 자산을 먼저 확인한다. | 조건 충족 시만 별도 확장한다. 환자 기저 RNA만으로 약물효과 검증을 주장하지 않는다. |

### 0.2 이번 계획의 성공 정의

성공은 “모델 이름을 많이 시험했다”가 아니다. 아래 중 무엇을 얻었는지 명확해지는 것이다.

1. **후보 확인:** R17 결합의 이득이 다른 적격 측정에서도 유지되는가?
2. **병목 식별:** 더 좋은 입력, 맥락 정보, 출력 기저, 손실 중 무엇이 실제로 새로운 약물 신호를 더하는가?
3. **적용 범위 확정:** 구조만 있는 약물, 외부 실측이 있는 약물, 특정 간 세포주 중 어디에서만 유효한가?
4. **데이터 요구 확정:** 현 공개자료로 답할 수 없는 질문은 무엇이며, 어떤 추가 측정이 필요한가?

통계적으로 구별되지 않는 결과는 곧바로 “효과 없음이 증명됐다”가 아니다. 효과크기와 구간, 검정력, 사전 정한 실용적 최소 개선을 함께 보고한다.

## 1. 진행 내용 검토: 최신 결과가 바꾼 판단

### 1.1 연구 흐름 요약

| 단계 | 확인된 내용 | 이번 계획에 주는 의미 |
|---|---|---|
| R6·R8 등 초기 후속 연구 | 증강·단순 보정의 효과와, 추가 복잡한 모듈의 증분 가치가 서로 달랐다. 특정 recipe 미채택을 모든 관련 원리의 실패로 확대할 수 없다. | 복잡한 모듈은 항상 강한 단순 대조와 비교한다. |
| R12 | shift·context·구조·손실 및 결합을 포함한 42개 행렬 항목 완료. 공식 개선 기준을 만족한 최종 축은 없었다. 일부 matched 비교의 작은 개선은 있었다. | 네 축을 “아직 한 번도 안 했다”고 쓰지 않는다. 새 실험은 바뀐 관측·정보 계약을 명시한다. |
| R13 | 297개 설정과 출력 배율 탐색에서도 일반적 우위가 확립되지 않았다. 지정 외부 A549 전이는 악화했다. | 같은 자료·같은 가설의 대규모 격자 확대를 반복하지 않는다. |
| Tahoe 위치 감사, T2→P1/P4 | 위치 정보와 공유 웰 구조를 공정하게 맞추면 초기 이득 상당 부분이 줄었다. 일부 성과는 소수 약물에 집중됐다. | 위치·공유 웰·약물군별 기여를 필수로 보고한다. |
| T3/P3 HepaRG | 외부 LINCS 약물 실측이 있는 일부 조건에서 작은 개선을 확인했다. | 전이 조건의 양성 경로는 보존한다. 구조-only 또는 HCC 전체 성공과 구분한다. |
| R14 | 실행 완료 기록과 산출물 해시는 있으나, 최신 Git에도 외부 효과크기·CI 원표 일부가 없다. | 첫 작업은 기존 동결 결과의 공개·대조다. 재실험부터 하지 않는다. |
| R15/R16 | 배율 공정 비교, 6 h 결합, CMap 24 h 확인, 반복 측정 평가까지 완료됐다. 내부 선택의 DMSO 공유 문제가 후속으로 발견됐다. | 이전 보고서의 “R16 결합·확인 미완료” 상태를 더 이상 사용하지 않는다. |
| **R17** | fit/select/score DMSO를 분리한 교정 후, 커널+FiLM/concat 결합이 A/B 모두 양성이다. | 가장 먼저 확인할 후보가 생겼다. |
| **R18** | 검토 커밋에는 확인 요청서만 있다. R18 protocol·freeze·score·RESULTS는 확인되지 않았다. | 완료로 보고하지 않는다. 서버의 별도 진행 여부는 실행 agent가 원장으로 확인한다. |

근거: 저장소 자료 [G1]–[G8]. 서로 다른 농도·표적·정규화·통계 기준의 수치를 한 순위표로 합치지 않는다.

### 1.2 R17의 실제 양성 신호

R17의 A는 **약물 구조 입력**, B는 **동일 약물의 비간계열 실측 RNA 입력 허용**이다. 주 평가 행은 GSE92742 HEPG2 6 h, 978 landmark 유전자이며 A 1,862 parent, B 1,815 parent다.

| 모델 | A RMSE | A R² | B RMSE | B R² |
|---|---:|---:|---:|---:|
| B0 | 0.89649 | 0.00% | 0.89821 | 0.00% |
| R15 커널 | 0.88883 | 2.04% | 0.88147 | 4.50% |
| 배율 적용 기존 VC 최선 | 0.88796, FiLM | 2.04% | 0.89140, concat | 2.74% |
| **커널+FiLM** | **0.88392** | **3.22%** | **0.87381** | **5.91%** |
| 커널+concat | 0.88393 | 3.26% | 0.87379 | 5.92% |

Δ는 후보−비교군이다. 아래 구간은 parent 10,000회 짝 재표집의 양측 99% 구간이다. R² 차이는 %p다.

| 비교 | A ΔRMSE / ΔR² | B ΔRMSE / ΔR² | 판정 |
|---|---|---|---|
| R15−배율 적용 기존 VC 최선 | +0.00087 [−0.00043, +0.00207] / −0.001 [−0.36, +0.39] | −0.00993 [−0.01099, −0.00886] / +1.76 [+1.49, +2.04] | A 미통과, B 통과 |
| 커널+FiLM−R15 | −0.00491 [−0.00565, −0.00419] / +1.18 [+0.99, +1.36] | −0.00765 [−0.00893, −0.00634] / +1.41 [+1.03, +1.78] | A/B 통과 |
| 커널+FiLM−기존 VC 최선 | −0.00404 [−0.00541, −0.00284] / +1.17 [+0.82, +1.55] | −0.01759 [−0.01894, −0.01629] / +3.17 [+2.84, +3.51] | A/B 통과 |

B의 결합은 단순 전이 B_cell×g도 이겼다. FiLM 결합−B_cell의 ΔRMSE는 −0.01216 [−0.01344, −0.01095], ΔR²는 +2.66 [+2.37, +2.96] %p였다. [G1]

**해석:** 유망한 조합을 찾았다. 그러나 R² 3.22%·5.91%는 “RNA 반응을 충분히 정확하게 복원한다”는 수준의 입증이 아니다. 이미 본 개발자료에서의 상대적 개선과 적용 가능한 예측 품질을 분리한다.

### 1.3 왜 R16과 R17에서 결론이 달라졌나

R15/R16의 핵심 문제는 평가 약물의 처리 정답이 직접 학습에 들어갔다는 것이 아니다. **학습과 내부 모델 선택이 같은 물리 DMSO 평균을 사용**했다. 그 평균의 우연한 잡음을 따라가는 후보가 내부 검증에서 유리해질 수 있는데, 최종 평가는 다른 DMSO 평균으로 수행됐다.

R17은 41개 plate의 529개 물리 DMSO 웰을 세 몫으로 분리하고 fit/select/score를 ABC, BCA, CAB로 교대했다. 신경망 단독의 선택 배율은 R16의 1에서 R17의 0.25로 줄었고, 결합 신경망 계수는 0.125–0.25였다. 이 관찰은 공유 대조군 잡음에 의한 선택 편향 설명과 부합한다. [G1, G2]

단, 동시에 DMSO 몫당 표본수와 재적합도 달라졌다. 변화 전체를 “누수 하나를 제거한 순수 인과 효과”라고 단정하지 않는다. 이번 D1 실험은 **같은 DMSO 수에서 공유 여부만 바꿔** 이 설명을 검증한다.

### 1.4 아직 해결되지 않은 부분

- **평균 반응 대조:** R17에는 B0/B_pos라는 drug-free 기준선은 있지만, fit-only 평균 처리반응·학습된 context-only·B의 blend_anchor 자체 점수가 충분히 포함되지 않았다.
- **모델 선택 대칭성:** 커널은 규칙×gain 공동 선택, 기존 VC는 gain=1에서 lr/epoch를 고른 뒤 gain을 고른다. 새 개발 실험에서는 대칭적으로 맞춘다. R18 원규약은 몰래 바꾸지 않는다.
- **공유 plate 불확실성:** parent bootstrap은 현재 plate 집합에서의 약물군 불확실성을 나타낸다. 새 plate·새 assay 일반화까지 보증하지 않는다.
- **사영 진단의 사용 제한:** R17에서 score DMSO 방향을 뺀 사영 결과는 사후 진단이다. 실제 허용 입력만으로 달성한 예측 성능이 아니다.
- **반복 측정 상한:** 교차 plate 반복이 B0보다 나쁘거나 불안정한 경우가 있다. 관측 반복 점수를 참 생물학적 상한으로 취급하지 않는다.
- **자료 노출:** CMap 24 h, MIX-seq, Emerald Bay 등을 “전혀 안 본 외부자료”로 다시 명명할 수 없다.

## 2. 첨부 PDF에서 유지할 것과 수정할 것

### 2.1 유지할 설계 원리

1. 공통 맥락 반응과 약물 특이 반응을 분리한다.
2. 교차 플랫폼의 단위·시점·용량 차이를 진단한다.
3. 구조 외 생물활성 정보는 실제 가용성을 명시한 별도 조건으로 시험한다.
4. 출력 기저, 손실, 입력의 효과를 독립 비교한 뒤 작은 상호작용 실험으로 결합한다.
5. 기전·경로 점수를 원 유전자 오차와 함께 보되, 이를 인과적 작동기전 입증으로 과장하지 않는다.
6. 실패·보류·자산 부족을 구분하고 원표와 판정을 함께 남긴다.

### 2.2 반드시 고칠 주장과 실행 지시

| PDF 위치·주장 | 문제 | 보완안 |
|---|---|---|
| 1–4쪽: 기존 구조/신경망/MSE 경로를 일괄 실패로 분류 | 특정 recipe 결과와 방법 전체의 불가능성을 혼동한다. R17 결합 양성과도 맞지 않는다. | R17 결합·공정한 MSE를 기본 비교군으로 유지한다. |
| 5–8쪽: Top-1 22.9%, R² −3.42이면 방향은 성공했고 주로 2배 크기 문제 | 검색률은 후보군·차원·공통 모드에 의존한다. 기존 pooled cosine도 낮다. | 3절의 실제 좌표계 진단으로 교체한다. |
| 6쪽: 양의 유전자별 보정은 방향 검색을 거의 보존 | 양의 전역 스칼라와 유전자별 대각 변환은 다르다. 후자는 각도·순위를 바꿀 수 있다. | scalar/diagonal을 분리하고 방향 비열등성을 직접 시험한다. |
| 6쪽: L1000 L5 MODZ가 해당 전이 실패의 원인 | 현재 R15 경로는 L3 log2 차이×ln2다. | 사용된 레벨·변환을 코드에서 고정하고, L5는 별도 데이터 계약으로만 평가한다. |
| 7쪽: 모의실험에서 label-free 보정으로 R² 양수 | 구성한 시뮬레이션은 실제 처리효과의 식별 근거가 아니다. | 알고리즘 sanity test로만 사용한다. 성공 확률·실데이터 효과로 인용하지 않는다. |
| 8–11쪽: 여섯 방법을 모두 target-label-free로 묶음 | DMSO-only, target genetic labels, target treated atlas, source의 동일 약물 실측은 다른 정보다. | 4절의 입력 허용표로 재분류한다. |
| 13–16쪽: 다중뷰가 최대 성능을 결정론적으로 제공 | 성능 보장은 없다. coverage·결측·노출이 결과를 좌우한다. | 공통 지원·전체 지원·결측 전용 평가와 약물 셔플 대조를 둔다. |
| 16–21쪽: MSE를 음성 대조로 규정하고 norm 회복을 성공 기준으로 강제 | 잡음이 큰 문제의 적절한 조건부 평균은 관측 반응보다 작을 수 있다. 큰 norm은 신호 회복의 증거가 아니다. | 공정한 MSE를 유지하고 실제 held-out 오차·약물 특이성으로 판정한다. |
| 17–20쪽: VCC 2026 지표와 우승 구조를 그대로 화학 de novo 목표로 전용 | 공식 과제는 CRISPRi의 새 맥락 전이이며 대부분 표적은 다른 맥락에서 관측됐다. | 전이 설계의 참고로만 사용한다. 화학 신규 parent 성능을 따로 검증한다. |
| 20–21쪽: pDE head와 400개 합성세포로 DEG/fid/jac 최적화 | 반복이 없는 DE 라벨과 합성세포 수 조절은 생물학적 반복을 만들어 주지 않는다. | 독립 물리 반복이 있을 때만 DE 학습. 합성분포는 별도 가설과 실제 분포 정답이 있을 때만 시험한다. |
| 21쪽: PCA/EBMF/cNMF의 높은 VE를 예측 성공으로 연결 | 재구성 가능성과 새 약물 예측 가능성은 다르다. 데이터 기저도 test에 맞추면 누수다. | fit-only basis, held-out 재구성 진단, 동일 예측기 성능을 분리한다. |
| 22–24쪽: 공개 자원의 수·ID·역할을 묶음 | GSE176150와 MoABox는 별개이며, LIMORE·환자 기저 RNA는 약물 후 RNA 검증셋이 아니다. | 6절 데이터 재고와 공식 출처로 교체한다. |
| 25–26쪽: 1.49 GB 및 8×48 GB만으로 첫날 전체 트랙 가동 가능 | 이는 일부 입력 텐서의 하한이다. 커널, 중간 복사, 원자료, optimizer·activation·캐시는 별도다. GPU 메모리도 자동 합산되지 않는다. | 실측 pilot 후 작업당 RAM/VRAM/시간·디스크 상한을 정한다. |
| 27–30쪽: R²>0.05, PDS≥0.75, DEG 중앙값 등 외부 절대 문턱 | 현재 자료의 신호·지표 정의·정보 계약과 연결되지 않는다. | 공정한 기준선 대비 효과와 사전 실용 기준, 독립 확인으로 대체한다. |
| 28쪽: dark-chemistry가 안 되면 Boltz 표적 30→100개 자동 확대 | 효과가 없는데 비용만 키울 수 있다. | 아주 작은 pilot에서 증분정보·가용성·비용을 확인한 후에만 확장한다. |
| 29–30쪽: 6주 안에 환자·오가노이드 검증 완료 | 관측 기저 RNA와 viability만으로 약물 ΔRNA 예측을 검증할 수 없다. 통제 접근도 있다. | 6주는 자원 가용 시 개발 목표로만 제시한다. 임상 적용·환자 치료 창 주장은 제외한다. |

### 2.3 문헌 검토의 범위와 신뢰도

PDF의 189개 참고문헌을 전부 독립 검증했다는 뜻은 아니다. 이번에는 설계 의사결정을 바꾸는 핵심 원문·공식 코드·기탁 정보를 확인했다. 공식 저장소 존재, 학습 데이터 확인, checkpoint 가용, 실제 로컬 실행 가능, 우리 평가 통과는 **각각 다른 단계**다.

PDF의 일부 참고번호는 주장과 직접 맞지 않는다. 예를 들어 pChEMBL 필터의 [113]은 다른 질환 바이오마커 논문, VICReg의 [130]은 개인별 참조유전체 논문, MoABox의 [150]은 signature search 논문을 가리킨다. 그 번호를 그대로 근거로 재사용하지 않는다. 이번 부록에는 확인한 1차 출처와 미확인 범위를 별도로 남겼다.

## 3. 외부 전이 실패를 다시 정의하는 수학적 진단

### 3.1 현재 수치가 의미하는 것

R15 외부 기술표에서 최소 20세포, LINCS-supported 간 두 세포주 층은 161개 공통 유전자·108개 지원 약물군의 제한된 평가다. R15-B의 Top-1은 약 22.9%, 평균 row cosine은 약 0.056, **pooled energy cosine**은 0.1536247927, prediction/observed RMS 비는 2.0106242845였다. 서로 다른 cosine 정의를 섞지 않는다. [G9]

같은 pool 중심화·같은 유전자·같은 에너지 가중 공간에서 실측 벡터를 y, 예측을 p라 하면:

$$
R_0^2=1-\frac{\lVert y-p\rVert^2}{\lVert y\rVert^2}=2\rho c-c^2,
\qquad \rho=\frac{\langle y,p\rangle}{\lVert y\rVert\lVert p\rVert},\quad c=\frac{\lVert p\rVert}{\lVert y\rVert}.
$$

이 정의는 임의의 평균 Pearson이나 다른 macro-RMSE에 그대로 대입하는 공식이 아니다. 기존 수치에 대입하면:

| 진단 | 계산값 | 해석 |
|---|---:|---|
| 현재 R² | −3.42485 | 방향과 크기가 함께 문제다. |
| norm만 맞춘 c=1의 R² | −0.69275 | 크기를 1로 맞추는 것만으로 양수가 되지 않는다. |
| 같은 frozen vector의 oracle 양의 scalar | a*=ρ/c≈0.07641 | target 정답을 봐야 얻는 사후 진단값이다. 실제 선택값이 아니다. |
| 그 scalar family의 최대 R² | ρ²≈0.02360 | 현재 벡터를 전역 scalar만 바꾸는 경로의 상한은 약 2.36%다. |

**주의:** 마지막 값은 새로운 실험 성과가 아니며 새 모델·유전자별 변환·시간 변환의 상한도 아니다. 기존 고정 벡터의 전역 scalar family에 대한 대수적 진단이다. 이 숫자를 보고 target용 gain을 고르거나 R18에 넣지 않는다.

### 3.2 두 가지 선행 unit test

1. **각도 보존:** 공통 양의 scalar는 같은 후보 집합의 cosine 순위를 보존한다. 유전자별 양의 diagonal, centering, clipping, 비선형 복원은 일반적으로 보존하지 않는다.
2. **중심화 불변성:** 평가가 같은 pool의 공통 벡터를 제거하는 C 연산이면, 공통 μ에 대해 `C(p−γμ)=C(p)`다. 고정 diagonal D가 pool 내에서 같아도 `C[D(p−γμ)]=D C(p)`다. 따라서 단순 공통 source 평균을 사후에 빼는 γ sweep은 현재 contrast 지표에서 아무것도 바꾸지 않을 수 있다. 코드가 이 조건에 해당하면 해당 sweep은 실행하지 않고 `NOT_APPLICABLE_INVARIANT`로 기록한다.

약물마다 source 가중치가 다르거나 학습 목표를 바꾸는 centering은 별개의 가설이다. 절대 DMSO-delta 개선과 pool-centered 약물 대비 개선도 따로 보고한다.

### 3.3 새 가설

“방향은 맞으므로 분산만 보정한다”가 아니라 다음을 분리한다.

- H-shift: 허용된 대조군·source 데이터에서 정한 보정이 target 처리 정답 없이도 오차를 줄이는가?
- H-time: 같은 약물·세포·플랫폼의 시간 간 대응에서 학습한 연산자가 단순 배율보다 낫고, 다른 시간쌍에도 재현되는가?
- H-context: 약물별 맥락 교호작용이 공통 평균과 동일 약물 단순 전이를 넘어서는가?
- H-information: 외부 생물활성 정보가 같은 coverage에서 구조 이상의 정보를 주는가?
- H-core: 같은 입력에서 구조·출력 기저·손실이 더 많은 약물 특이 신호를 학습하게 하는가?

타깃 DMSO 분산은 처리효과 gain이나 방향을 직접 식별하지 못한다. 분산 보정은 검증할 가정이지 정답 복원 공식이 아니다.

## 4. 과제와 입력 권한: 섞어서 비교하지 않는다

### 4.1 시나리오 등록표

| 계약 | query 약물에 허용되는 정보 | 금지/주의 | 주장 가능한 범위 |
|---|---|---|---|
| **A, 기존 의미 유지** | 구조, 농도·시간, 허용된 사전 맥락·fit DMSO | query parent의 처리 RNA는 모든 학습 데이터셋·시간·용량에서 제외 | 구조 기반 새 약물 예측 |
| **B, 기존 의미 유지** | A + 동일 약물의 비간계열 실측 RNA, 지정 시간·용량 범위 | target 처리 RNA 및 그 파생 특징은 금지. source query 실측은 입력 전용이며 target label fit로 재사용 금지 | 동일 약물의 다른 맥락/플랫폼 전이 |
| **M-annot** | A + 이미 알려진 표적·MoA·정량 bioassay | 구조-only A라고 부르지 않음. annotation의 출처·시점·query 포함 여부 기록 | RNA가 없지만 약리 주석은 있는 약물 |
| **M-phenotype** | M-annot + 외부 Cell Painting 또는 허용된 생존율 지문 | query target의 최종 평가 endpoint를 특징으로 사용 금지 | 일부 실험 정보가 있는 약물 |
| **A-proxy** | 추론 시 구조만으로 만든 예측 형태학/표적 embedding | 사전학습의 query RNA·실측 morphology/bioassay/viability supervision을 감사. 불명확하면 엄격 A 표에서 제외 | 노출을 감사한 pretrained 구조 표현 |
| **G-context** | target 기저 RNA + target 유전 섭동 정보, source 약물 데이터 | DMSO-only라고 부르지 않음. 유전 개입의 종·세포주·표적 대응 확인 | 유전 정보가 있는 맥락으로의 약물 전이 |

B에 M 정보를 추가하면 `B+M`이라고 명시한다. M의 성능을 A나 B의 개선으로 조용히 합치지 않는다. “zero-shot”은 **어느 세포주·약물·assay의 어떤 라벨을 안 봤는지**를 항상 붙인다.

기존 내부 A/B는 평가 query parent의 target 처리 RNA를 금지하지만, target context의 다른 train-parent 처리 RNA까지 전부 금지하는 과제는 아니다. **완전 target-context-zero-shot**이면 그 context의 모든 treated RNA를 추가로 차단한 별도 계약을 등록한다. `G-context`도 source의 동일 query 약물 RNA를 쓰지 않으면 `A+G`, 쓰면 `B+G`로 나눈다.

### 4.2 전 데이터 공통 원칙

- parent canonicalization, salt/tautomer/stereochemistry 정책을 고정한다. 실패한 구조를 임의 다른 약물로 매핑하지 않는다.
- A의 outer-held parent는 모든 원자료·파생 사본·다른 시간/농도/세포주에서도 반응 학습에서 제외한다. scaffold 분할은 별도 고정 민감도 분석이다.
- B의 동일 약물 source RNA는 명시적 예외다. 다른 모델에도 같은 정보 권한과 support를 제공한다.
- pretrained encoder의 query parent 비구조 supervision 노출을 확인 불가능하면 `PRETRAIN_EXPOSURE_UNKNOWN`; 엄격한 unseen-parent 확증 주장에서 제외한다. 비지도 학습에서 구조만 본 경우, 실제 Cell Painting/bioassay/viability를 본 경우, 처리 RNA를 본 경우를 따로 기록한다.
- 공통 지원 paired 비교와 배치 전체 fallback 비교를 모두 낸다. 좋은 뷰가 있는 약물만 남겨 전체 성능처럼 보고하지 않는다.
- target 처리 후 viability, DE, 최종 예측 오차로 평가 집단을 제거하지 않는다. 원본 QC로 정의한 전체 집단을 주 분석으로 유지하고 독성 층은 사전 정의된 보조 분석으로 보고한다.

## 5. 두 실행 레인: 기존 후보 확인과 새 가설 개발을 분리

| 레인 | 목적 | 바꾸면 안 되는 것 | 결과 지위 |
|---|---|---|---|
| **CONFIRM-R18** | R17 절차가 다른 측정에서도 작동하는가 | 기존 요청의 후보·격자·선택·대조군 역할·지표·판정 | 과거 노출을 명시한 고정 절차 확인 |
| **DEV-ARCH** | 새 입력·맥락·구조·손실·보정 중 무엇이 유효한가 | 개발자료/확인자료 경계, 공정 비교, 모든 변경 기록 | 탐색·원인 규명·후보 승격 |

R18을 진행하면서 DEV 후보를 C1 성능으로 골라서는 안 된다. 그렇게 했다면 C1은 그 후보에 대해 개발자료다. R18 실패는 해당 고정 recipe의 확인 실패이지 모든 DEV 가설의 종료 명령이 아니다. R18 성공도 DEV의 새 모델을 대신 검증하지 않는다.

실행 전에 **C1–C3 전체의 처리 웰·파생 서명·동일 물리 측정의 별칭을 R18-reserved manifest로 등록**하고 DEV의 source 적합·선택·특징 사전학습에서 차단한다. 특히 C2의 미사용 상태를 DEV가 먼저 소진하지 않는다. 기존 B의 허용 source donor 입력 예외도 별도 열로 관리한다. R18 후 자료를 개발에 풀면, 이후 새 후보에 대해서는 개발자료로 이력을 갱신한다.

## 6. 데이터 재고와 실제 가능한 역할

### 6.1 이미 보유한 자료: 가장 먼저 활용

| 자료 | 최신 확인 상태 | 이번 역할 | 제한 |
|---|---|---|---|
| GSE92742 HEPG2 6 h | R15–R17 반복 개발, A 1,862/B 1,815 parent | DEV의 감사·경량 기전 검증 | 새로운 blind test 아님 |
| CMap HEPG2 24 h | R16 A 502/B 500군. A 477 same-well B1/R8, 1 다른 웰, 24 해당 세포주 미노출 parent | R18 C1 | 과거에 한 번만 본 자료가 아님 |
| HEPG2 24 h 저용량 | GSE70138/CMap에 원장 미일치 후보가 있음 | R18 C2 | release 중복·물리 웰·3역할 DMSO 감사 후 최종 n 확정 |
| GSE92742 HUH7 3 µM | 6 h 68군, 24 h 67군. 다른 plate·위치 반복은 7/6군 | R18 C3 기술 확인 | 두 시간을 합쳐 135개 독립 약물로 세지 않음 |
| CMap JHH5/JHH7 | 현재 규약의 복제 품질 통과 0군 | 기술 참고만 | 표준 QC를 낮춰 주 확인군으로 승격하지 않음 |
| Tahoe SNU-423/HepG2/C3A | 기존 개발·위치 감사·동일 위치 반복 사용 이력 | 기존 DEV 구간의 contrast·context 진단 | 다른 세포주가 같은 well을 공유. 세포 수가 독립 실험 n 아님 |
| MIX-seq exp 3/10 | R14 사용. 세포주별 지원 3/3/3/6군, 동일 조건 교차실험 반복 0 | 기존 외부 원표 복구·기술 비교 | 신규 HCC 확증 자료라고 재사용 불가 |
| Emerald Bay | R14 사용. 각 간 세포주 15군·35행, 대비 적격 13군·26행 | 시간 변화·생존율 보조 기술 분석 | 처리 조건당 1물리 웰. 5일 생존율은 24 h ΔRNA 정답 아님 |
| HepaRG GSE284321/GSE296254 등 | R14/T3/P3 사용 이력, 일부 양성 | 전이 DEV 또는 이력 표시 확인 | 정상 간독성 모델의 결과를 HCC로 확장하지 않음 |
| GSE264667 HepG2/Jurkat | 이미 보유, 유전 개입 열람 이력 | G-context 가설의 가용성 감사 | chemical test가 아님 |
| LIMORE | HCC 기저 RNA·90약물 민감도; 약물 후 RNA는 로컬 재고에서 미확인 | M-phenotype 또는 별도 viability 과제 | ΔRNA 학습/검증 정답으로 취급하지 않음 |
| 환자 HCC 기저 scRNA | 여러 GEO 코호트 보유 | baseline 분포·OOD·세포상태 기술 분석 | 약물 개입 정답이 없으면 효과 검증 불가 |

위 수치는 [G4]의 공개 재고를 기준으로 한다. 물리 웰, parent, 조건 행, cell 수를 각각 별도 열로 유지한다. `원장 미일치`는 전 세계의 미사용·미노출을 뜻하지 않는다.

### 6.2 추가 자료 후보: 수집 전에 목적과 권한을 확정

| 자료 | 확인한 1차 근거·역할 | 착수 조건/우선순위 |
|---|---|---|
| Novartis MoABox, Zenodo 14291446 | U2OS DRUG-seq 별도 대규모 저장소. 공식 파일 목록과 DRUG-seq 경로 확인 | 후보 1순위 bridge. 먼저 metadata·약물 중복·DMSO·반복·용량/시간을 확인. 전역 batch-corrected/PCA 파일을 fold-fit 결과처럼 쓰지 않음 |
| GSE176150 | U2OS 24 h, 14개 내부코드 약물, 8농도. 위 MoABox와 별개 | 소규모 normalization/처리 검증. 구조 ID 해소 전 cold-drug 학습 자산으로 집계하지 않음 |
| Open TG-GATEs | PHH 포함 microarray/CEL, 시간·용량·실험/개체 ID metadata 제공 | PHH 시간·간독성 보조. L1000·scRNA와 같은 raw-count NB likelihood를 강제하지 않음 |
| E-MTAB-12668 | 원논문 기탁 확인. APAP/CSA/VPA의 PHH/HepaRG, 단회와 반복 노출이 섞임 | 좁은 간독성·시간 가설. 대규모 신규 약물 학습셋 아님. 파일 가용성 추가 확인 필요 |
| JUMP/LINCS Cell Painting | 형태학 후보. query 약물·농도·세포주 매칭 필요 | 승인된 처리 feature부터. 원시 이미지 전체 다운로드를 기본으로 하지 않음 |
| Chem-PerturBridge | 여러 화학 섭동 자원의 통합 후보 | 실제 배포 버전·라이선스·원자료 중복·parent split을 확인한 뒤 조건부 도입 |
| X-Atlas/Pisces HepG2 | 공개 범위와 full atlas 범위가 다를 수 있음 | 실제 공개 파일의 표적·NTC·라이선스만 집계. 논문 전체 세포 수를 로컬 가용량으로 쓰지 않음 |
| LICOB/PDHO 등 오가노이드 | 기저 omics·약물 민감도와 약물 후 RNA 여부를 별도 확인해야 함 | controlled access, 신규 신청, 임상/환자 데이터 활용은 별도 승인. 현 계획의 필수 완료조건 아님 |

새 자료의 수집은 기존 R18의 “새 다운로드 없음” 규칙과 분리한다. 이 기획서 작성은 다운로드 승인이 아니다. 실행 지시를 받은 agent도 대용량·통제 접근·새 비용이 필요한 항목은 수집 목록과 목적·용량을 보고하고 승인받는다.

## 7. 공통 평가 계약

### 7.1 세 가지 DMSO 역할

- **fit:** 입력 control, 학습 delta, 위치 donor, control-derived feature 산출에만 사용.
- **select:** 내부 검증 정답과 gain/epoch/구조 선택에만 사용.
- **score:** 예측 동결 후 평가 정답에만 사용. 특징, scale, gene weighting, gene selection, 사영 방향 선택에 사용하지 않는다.

세 역할은 **물리 웰 단위로 불일치**해야 한다. 같은 well의 세포를 나눠 세 역할이라고 부르지 않는다. R18은 기존 R17 CRC32 배정·순환 규칙을 그대로 사용한다. DEV의 새 자료도 역할 지원을 먼저 감사하고, 불가능하면 `BLOCKED_CONTROL_SUPPORT` 또는 규약을 명시한 별도 약한 분석으로 남긴다.

유일한 의도된 공유 예외는 D1의 공유 선택 진단군이다. 이 군은 편향 경로를 시험하는 대조이며 배포 후보·정식 확인 모델로 채택하지 않는다. manifest에 예외 이유를 명시한다.

여러 역할 회전·seed·동일 웰의 여러 세포주는 독립 biological replicate가 아니다. role 3회 평가 row 수를 물리 웰 수로 쓰지 않는다.

### 7.2 분할·전처리·선택

1. R18: 원 outer 3-fold 및 inner validation 규칙 그대로.
2. DEV 기본: 기존 개발 parent outer 3-fold, outer-fit 안에서 고정 inner 분할. 새 grid는 축별 최대 8설정, 기본 공통 gain grid `0, .125, .25, .5, .75, 1`.
3. **DEV는 모든 후보와 비교군의 checkpoint/lr×gain을 같은 목적함수로 공동 선택**한다. 비교군의 예산을 후보보다 불리하게 줄이지 않는다.
4. gene selection, PCA/EBMF, 표준화, target/MoA 압축, imputation, view weight, clipping 한계는 inner-fit/outer-fit에서만 추정한다.
5. target DMSO를 사용하는 상황은 transductive-control 입력 허용으로 명시한다. target 처리 label-free와 모든 target 데이터 미사용은 다르다.
6. 1 seed의 선별 실행 후, 승격 후보와 해당 비교군만 고정 3 seed로 확인한다. seed 간 구간을 생물학적 표본 구간으로 대신하지 않는다.
7. 모든 후보는 같은 적격 row·gene panel·normalization에서 짝 비교한다. 패널을 바꾸는 실험은 별도 버전이다.

### 7.3 기준선 묶음

| ID | 기준선 | 필요한 이유 |
|---|---|---|
| B0/B_pos | 기존 무변화/위치 기준선 | 무반응 및 실험 설계 신호와 비교 |
| B_mean | parent 균등 fit-only 평균 반응, gain도 내부 선택 | 약물 정보 없이 공통 반응만 맞춘 효과 제거 |
| B_resmean | 공유 anchor + fit-only 평균 잔차 | 새 잔차 모델의 이득이 단순 offset인지 확인 |
| B_context | 약물 특징을 제거한 capacity-matched 모델 | 맥락·모델 용량만으로 생기는 개선 분리 |
| B_cell×g | B의 동일 약물 단순 전이 | 복잡한 transfer가 단순 측정 전이보다 나은지 확인 |
| B_blend_anchor | B_pos+B_cell×g, 화학/신경망 잔차 없음 | R17 결합 잔차가 anchor 위에 실제로 기여하는지 확인 |
| B_R15 | R15 커널, 같은 새 선택 계약 적용 | 커널 기여 측정 |
| B_R17 | 커널+FiLM; concat은 보조 | 현재 유망 후보를 넘어서는가 확인 |
| 기존 VC 최선 | 기존 7종, 명시된 같은 입력·선택·배율 | 약한 비교군을 골라 승리하는 것 방지 |

R18 원 주판정표는 그대로 두고, 추가 평균/context 기준선은 DEV 또는 별도 사전등록 보조표에 낸다. 전체 VC·출판 모델·프로젝트 내부 기준을 구분한다. 단순히 공개 모델의 이름을 사용했다고 원 논문 조건 전체를 재현했다고 쓰지 않는다.

### 7.4 지표와 통계

- **주 지표:** 해당 계약의 parent 균등 RMSE와 pooled R². R18은 원 정의 그대로. DEV에서도 절대 DMSO-delta와 pool-centered 약물 대비를 혼합하지 않고 별도 표로 낸다.
- **약물 특이성:** within-pool contrast RMSE/cosine, full vs context-only/parent-shuffle, 같은 candidate pool의 Top-1/Top-5 및 chance/tie 처리.
- **맥락 특이성:** 같은 약물에 대한 cell×drug 교호작용, baseline-only와 cell-label/context shuffle 대조. 공유 well을 cluster로 취급한다.
- **기전 보조:** 사전 정한 유전자·경로의 부호/효과크기. 전체 평가를 지지하는 보조 근거이지 약물 작용기전 인과 증명이 아니다.
- **DE/분포:** 적격 독립 반복과 raw 또는 적절한 관측 모델이 있을 때만. 합성세포를 실제 반복으로 계산하지 않는다.
- **불확실성:** R18은 10,000회 parent 짝 bootstrap, 양측 99%, 원 공동 기준. DEV도 효과크기·99% 구간을 제시하고 plate/batch 삭제 민감도와 가능한 parent+plate 군집 분석을 별도 추가한다.
- **다중 탐색:** DEV의 여러 arm에서 가장 좋은 구간만 골라 확증 성공이라고 하지 않는다. 네 축의 계약별 주 비교 전체를 하나의 family로 사전 등록한다(A/B 모두면 최대 8개). 통계적 주장 시 family-wise α=0.01의 Holm 보정을 적용한다. 각 RMSE/R² 공동 가설은 두 방향성 검정의 큰 p값을 사용한다. 99% 개별 구간과 다중 보정 판정은 다른 열이다. 그 외 표는 탐색으로 표시한다.
- **최선 비교군:** 동일 bootstrap 표본에서 비교군 최선을 다시 정하는 기존 규칙을 유지하거나, inner에서 비교군을 고정하는 규칙을 새로 등록한다. 두 정의를 섞지 않는다.

`cell-eval2`는 원래 유전 섭동 과제의 target-gene 제거·정규화·패널 규칙을 갖는다. 화학 약물에 임의로 표적 유전자를 하나 지정해 제거하지 않는다. 도입 시 commit·preset·preprocessing·raw/normalized score·지원 수를 고정하고, 기존 주 지표의 대체가 아니라 별도 호환성 분석으로 시작한다. [L1, L2]

네 축의 기본 주 비교는 **T3−T1, X2−X1, S2−S0, L1−L0**로 제안한다. 적격성이 없거나 같은 함수여서 비교가 무효이면 채점 전에 사유를 기록한다. 나머지 arm으로 주 비교를 사후 교체하지 않는다. 보조 arm이 유망하면 새 후보·새 확인으로 승격한다. LINCS 주 endpoint는 원 DMSO-delta, Tahoe 주 endpoint는 기존 pool-centered 대비로 고정하고 두 결과를 합산하지 않는다.

DEV의 한쪽 검정은 개선 방향을 `−ΔRMSE`, `+ΔR²`로 정한다. 기본 p-value 방법은 parent별로 후보/고정 비교군의 예측 묶음을 확률 1/2로 교환하는 10,000회 paired randomization으로 제안한다. 같은 parent의 role·well·seed 묶음은 함께 교환하고, 매번 지표를 다시 계산하며 `(1+더 극단적인 null 횟수)/(1+반복수)`를 사용한다. 이는 parent 블록의 귀무 교환가능성을 가정하므로 plate 공유 민감도를 함께 보고한다. 이 가정이 맞지 않으면 정식 p값 주장을 보류하고, source 단계에서 타당한 군집 검정법을 명세한 뒤 진행한다. 단순 bootstrap 구간에서 p값을 자동 역산하지 않는다. R18에는 이 새 DEV 검정을 소급 적용하지 않는다.

### 7.5 운영상 승격·종료 기준

아래 수치는 문헌의 보장값이 아니라 **새 DEV의 제안 운영 기준**이다. 학습 전 원 데이터의 분산·source pilot 검정력과 자원에 근거해 한 번 확정하고, target 결과를 본 뒤 낮추지 않는다.

- **DEV_PROMOTE:** 동일 지원의 강한 기준선 대비 상대 macro-RMSE 개선 점추정 ≥0.5%, R² 개선 ≥0.5%p, 3 outer fold 중 2개 이상에서 같은 방향, 그리고 해당 가설의 특이 대조를 통과. 효과크기·99% 구간을 함께 보고한다. 이는 독립 확인 성공이 아니라 다음 예산을 받을 후보라는 뜻이다.
- **NO_USEFUL_GAIN:** 공정성 감사가 끝났고 개선 구간의 유리한 끝도 사전 최소 유용 개선에 못 미치면 현재 recipe 종료.
- **INCONCLUSIVE:** 구간이 넓어 유용한 개선과 무효를 모두 포함하면 판정 불충분. 같은 격자를 늘리기보다 정보량·독립 반복 문제를 제시한다.
- **BLOCKED/NOT_APPLICABLE:** 데이터·자산·반복·수학적 불변성 때문에 가설을 검정하지 못함. 음성 실험으로 세지 않는다.
- **CONFIRMED:** 사용하지 않은 적격 확인자료에서 동결한 후보가 등록 기준을 충족했을 때만 사용. 과거 노출 확인은 그 한계를 제목에 붙인다.

R18의 원 통과 기준에는 위 DEV 0.5%/0.5%p 문턱을 소급 추가하지 않는다. R18 원 판정과 실용적 크기의 해석을 나란히 제시한다.

## 8. 필수 최소 실험 행렬: 네 축을 모두 다룬다

| 작업군 | 필수 ID | 검증할 한 문장 | 다른 축 실패 시 계속하나? |
|---|---|---|---|
| 감사·기준선 | D0–D3 | 모델 개선이 공유 대조군·선택 규약·공통 평균 때문인가? | 공통 선행조건 |
| 기존 후보 확인 | C1–C3 | R17 절차가 다른 적격 측정에서도 유지되는가? | DEV와 별개 |
| shift/시간 | T0–T4 | source/control-only 보정이 raw transfer보다 유효한가? | **계속** |
| context | X0–X3 | 기저 맥락과 약물의 교호작용이 약물별 정확도를 높이는가? | **계속** |
| 구조/출력 | S0–S3 | 같은 입력에서 저차원·잔차 구조가 예측을 개선하는가? | **계속** |
| 손실 | L0–L3 | 같은 모델에서 약물 대비·방향 손실이 실제 신호를 더하는가? | **계속** |
| 입력 정보 확장 | I0–I4 | 가용한 bioactivity view가 구조 이상의 정보를 주는가? | 자산이 있는 경우 계속 |
| 결합 검증 | J1 | 독립적으로 살아남은 두 변경이 함께 쓸 때도 기여하는가? | 지정 후보가 있을 때 |
| 생성/유전/대규모 확장 | E1–E5 | 필수 입력·추가 효용이 입증된 경우만 확장 가능한가? | 조건부, 미실행 사유 기록 |

**완료의 의미:** 각 필수 ID에 실행 결과 또는 구체적인 `BLOCKED/NOT_APPLICABLE` 증거가 있어야 한다. T축 하나가 실패했다고 X/S/L의 미실행을 전체 완료로 처리하지 않는다. 동시에 자산이 없거나 수학적으로 효과가 없는 arm을 억지로 학습하지 않는다.

## 9. 단계 0 — 감사와 원인 규명 D0–D3

### D0. 기존 결과 회수·불변 계약·단순 기준선

**질문:** 이미 한 실험을 다시 하지 않고, 비교의 출발점을 신뢰할 수 있게 만들 수 있는가?

실행 agent의 첫 작업:

1. 작업 시작 시 Git HEAD·작업트리·진행 중 job을 읽기 전용으로 확인한다. R18이 이미 진행됐다면 중복 제출하지 않는다.
2. R14의 기존 동결 `metrics.tsv`, paired table, support, freeze/score receipt를 서버의 기록된 경로에서 회수해 해시를 대조한다. 원 경로 접근 권한이 없으면 요청하고 멈춘다. 결과를 재생성해 기존 receipt인 것처럼 만들지 않는다.
3. R17의 점수·선택·역할 배정·gene axis·물리 웰/parent 수를 재집계하는 CPU-only 검증을 수행한다. 기존 결과를 덮어쓰지 않는다.
4. B_mean/B_resmean/B_context/B_blend_anchor의 새 DEV 명세를 등록한다.
5. 3절의 cosine/검색에 대한 양의 scalar 불변성, contrast의 공통벡터 이동 불변성, RMSE/R²의 analytic 변화, shape/NaN/gene-order 검사를 unit test로 만든다.

**산출물:** `progress_reconciliation.md`, `baseline_contract.yaml`, `exposure_manifest.tsv`, `support.tsv`, `tests/`의 통과 기록. 결과 회수와 새 모델 실험을 원장에 별도 표시한다.

### D1. 동일 DMSO 표본수에서 선택 공유 편향 검증

**가설:** 내부 선택이 학습 DMSO와 같을 때 큰 신경망 gain이 유리해지고, 독립 score DMSO에서 성능이 악화한다.

- 같은 처리 행·parent split·random seed·학습 후보·gain grid를 사용한다.
- plate별 세 역할에서 같은 m개의 물리 control well을 선택한다. m은 사용 가능한 세 몫의 최소 수로 정하고, selection 전에 ID 기반으로 고정한다.
- 공통 학습 입력은 A, 공통 score truth는 C다. **공유 선택 A vs 독립 선택 B만** 바꾼다.
- position/B_cell 계수·anchor·학습 labels는 두 군에 공통으로 사전 고정한다. R17의 position gain도 select truth에 의해 바뀌므로, 이를 그대로 달리 선택하면 공통 학습 대조가 아니다. D1의 주 비교는 동일 checkpoint bank에서 checkpoint/출력 gain의 선택만 바꾼다. 전체 pipeline의 선택까지 달리하는 비교는 별도 보조 실험으로 둔다.
- 역할을 순환하되 하나의 parent로 묶어 비교한다. 원 R16/R17의 서로 다른 control 표본수를 그대로 비교하는 실험과 구별한다.
- 가능하면 공통 checkpoint 예측을 재사용해 선택 절차의 효과만 비교한다. 최종 재적합이 필요하면 두 군의 seed·학습량을 맞춘다.

**보고:** 선택 gain/epoch 변화, 내부→외부 성능 차이, paired RMSE/R², 모델별 vehicle 연관성. 공유 선택이 나빠지지 않으면 가설 지지 안 됨으로 남긴다. R17 변화 전체의 단일 원인이라고 단정하지 않는다.

### D2. checkpoint×gain 공동 선택의 공정성

**가설:** 기존 VC의 checkpoint를 gain=1에서 먼저 고르면, 최종 작은 gain에서 좋은 checkpoint를 놓칠 수 있다.

- 기존 순차 선택과 모든 후보에 동일한 공동 선택을 적용한 DEV-v1을 비교한다.
- 같은 실제 유전자별 예측 배열로 선택 손실을 계산한다. scalar score를 prediction으로 취급하면 즉시 실패 처리한다.
- source/inner 선택 단계에서만 변경하고 score label을 사용하지 않는다.
- 이 변경만으로 순위가 바뀌면 구조 효과로 해석하지 않는다.

**다음 단계 기준선:** DEV의 T/X/S/L/I는 D2의 사전 등록된 공정 선택 구현을 공통으로 사용한다. R18은 그대로 유지한다.

### D3. score-control 없이 vehicle nuisance를 줄일 수 있는가

**가설:** R17 사후 사영에서 보인 잡음 일부는 학습용 독립 control 변동만으로도 줄일 수 있다.

- D3a: 현재 적합.
- D3b: fit DMSO의 물리 웰 subset을 바꾸는 control augmentation. 같은 처리 반응에 여러 control view를 주되 label 생성·가중을 명세한다.
- D3c: 충분한 source/fit control 반복이 있을 때만, 그 control의 변동으로 저차원 nuisance basis를 학습하여 신경망 잔차를 부분 축소한다. rank `{1,2}`, 제거 강도 `{0,.25,.5}` 등 최대 6설정. target score-control 사영은 금지한다.

D3c는 실제 약물 신호까지 제거할 수 있다. RMSE뿐 아니라 contrast·기전 부호·Full−Shuffle를 보존하는지 본다. 독립 control 지원이 부족하면 D3c만 보류하고 다른 축은 진행한다. 기존 R17 score-control 사영표는 기대 성능이 아니라 진단 참고로만 인용한다.

## 10. CONFIRM-R18 — 기존 요청을 유지한 확인

### 10.1 동결 항목

주 후보는 **커널+FiLM 하나**, concat은 보조다. R17의 물리 DMSO 3역할·회전, parent 분할, 학습/선택 grid, 혼합 선택 목적함수와 기존 VC의 순차 선택 절차, anchor, 결합 정의, 기존 VC 7종, 99% 공동 기준을 유지한다. 각 코호트의 가중치는 새로 적합한다. R15–R17의 결과를 warm-start하거나 외부 score에 맞춰 계수를 고르지 않는다. [G5]

### 10.2 코호트와 노출 문구

| ID | 자료 | 사전 지위와 필수 문구 |
|---|---|---|
| C1 | CMap HEPG2 24 h, 5–20 µM, release 비중복 행 | 주 확인. **R16 및 과거 B1/R8 노출이 있는 측정 집합의 사전 고정 절차 확인**이라고 적는다. 완전 blind 외부확증이라고 쓰지 않는다. |
| C2 | HEPG2 24 h, 0.04–3.33 µM의 미사용 적격 측정 | 용량별 보조 확인. 과거 parent/원 측정/파생 파일 노출을 모두 감사한다. 100 parent 미만 층은 요청서대로 기술 분석. |
| C3 | HUH7 3 µM, 6 h 및 24 h | 각각 68/67군 수준의 HCC 기술 확인. HEPG2-only 전이를 주 기술값, HUH7 train-fold 추가를 보조로 사전 지정한다. |

실행 전 release 중복, physical well, QC, 3역할 DMSO, B의 source support, 위치 baseline을 감사한다. 미리 기대한 n과 다르면 **채점 전에** 지원표와 명세를 고정한다. 새 처리 label을 보고 유리한 농도만 고르지 않는다.

### 10.3 결과 판정

원 요청의 B0, B_pos, R15, 배율 적용 최선 VC, B의 B_cell×g 비교를 모두 낸다. ΔRMSE 99% 상한 <0 및 ΔR² 99% 하한 >0을 공동 요구한다.

- C1 A/B 모두 등록 비교 통과: 원 문구 `확인됨` + “과거 노출 자료의 절차 확인” 한정.
- B만 통과: `전이 조건에서만 확인됨`.
- 그 밖: `확인되지 않음`.
- C3는 별도 기술 확인이며, C1 통과를 “성인 HCC 검증 완료”로 바꾸지 않는다.

화합물 기여 상위 5/10/20 점유율·제외 민감도, 같은/다른 plate·위치 반복, gain 분포도 함께 낸다. 반복 이득의 분모가 0 이하이면 도달률을 산출하지 않는다.

## 11. Shift·시간·용량 실험 T0–T4

### 11.1 공통 설계

**질문:** source와 target의 관측 조건 차이를 보정하면, target 처리 정답을 보지 않고도 약물별 반응 예측이 좋아지는가?

선택은 source의 pseudo-target domain에서만 한다. domain은 cell×assay×time×dose 규약으로 정의하고, 가능하면 최소 3개의 독립 source domain으로 leave-one-domain-out 검증한다. parent도 분리해, 동일 약물의 학습 라벨 재사용과 B의 허용된 source 입력을 구분한다. target에서는 fit DMSO만 사용한다.

기존 Tahoe 외부 성적은 이미 본 개발·진단 근거다. 이 자료에서 새 보정 후보를 고른 결과를 독립 외부확증이라고 부르지 않는다.

**보정기 자체도 cross-fit한다.** 각 outer pseudo-target domain/parent의 평가에서는 a, D의 hyperparameter, 시간 연산자를 그 outer 정답 없이 inner source에서 선택·적합한다. Base prediction이 OOF라도 같은 OOF 정답으로 a를 맞춘 뒤 그 자료에서 이득을 평가하면 낙관 편향이다. 최종 target용 보정기만 전체 허용 source 개발자료의 OOF 예측으로 재적합하고 동결한다.

**p의 정의:** T1–T3는 모델의 전체 예측 효과 delta를 보정한다. 이미 delta 출력이면 그대로 쓰고, absolute expression 출력이면 허용된 target fit-control을 빼 `delta_hat`를 만든다. 보정 후 absolute 출력은 `target_fit_control + a·D·delta_hat`다. 이미 delta인 값에서 control을 다시 빼지 않는다. B도 anchor를 포함한 전체 효과를 기본 대상으로 하며, residual-only 보정은 별도 arm으로 이름·anchor를 고정한다. 3절의 oracle 상한은 기존 전체 pool-centered contrast 벡터에 대한 것이므로 residual-only gain의 상한으로 전용하지 않는다.

### 11.2 최소 calibration 비교

| ID | 변경 | 선택·추정에 쓰는 자료 | 꼭 비교할 대상 |
|---|---|---|---|
| T0 | 변환 없는 동결 예측 및 불변성 검증 | 기존 예측, label-free unit test | B0/B_mean/B_R17와 해당 전이 baseline |
| T1 | 비음수 전역 scalar `a·p` | source OOF 예측과 source pseudo-target 정답 | T0 |
| T2 | control 분산의 shrink·floor·clip을 적용한 gene-wise diagonal `D·p` | source control와 target **fit** control; 규칙은 source에서 선택 | T0와 T1 |
| T3 | `a·D·p` | T2 변환 후 source OOF에서 a 선택 | T1과 T2 |
| T4 | 시간/용량을 정합한 source 전이와 작은 시간 연산자 | 동일 source cell·compound·정확 용량의 paired 시간 | identity·scalar·직접 같은 시간 donor |

T1의 예: source OOF 에너지 기준으로 `a=clip(sum(p*y)/sum(p*p),0,2)`를 적합하고, 그 선택 목적·상한을 source 명세에 고정한다. 분모가 0이면 a=0으로 처리한다. 적합에 쓰인 좌표계·gene weighting·centering도 고정한다. 이것은 공통 residual gain grid와 다른 calibration estimator다. target oracle 0.0764를 복사하지 않는다.

T2의 예:

$$
\log D_{gg}=\operatorname{clip}\!\left[\frac{\alpha}{2}\log\frac{v_{t,g}+\tau_g}{v_{s,g}+\tau_g},\,-b,b\right].
$$

`α`, variance floor/prior `τ`, clip `b`는 source-held-domain으로만 결정한다. 작은 분모에 raw ratio를 적용하지 않는다. T2의 기본 실행 지원은 해당 집계 수준에서 target-fit의 독립 control ≥12웰·≥3plate로 제안하고, source에서 안정성을 확인해 사전 확정한다. 이는 보편적 충분조건이 아니다. 지원이 안 되면 gene-wise 분산 가설을 보류하고 T1은 계속한다.

scRNA 개별 세포의 분산과 L1000 물리 well의 분산을 그대로 나누지 않는다. aggregation 수준·cell 수/depth·유전자 단위가 맞는 control pseudobulk 분산으로 정의한다. normalization 자체를 바꾸는 효과는 별도 패널에서 보고한다.

### 11.3 시간과 플랫폼을 분리

최소한 다음 비교를 지원 가능한 동일 parent subset에서 낸다.

1. L1000 6 h → L1000 24 h: 시간 변화.
2. L1000 24 h → target 24 h: 같은 시간의 플랫폼/세포 변화.
3. L1000 6 h → target 24 h: 실제 결합 변화.
4. 동일 플랫폼·시간에서 exact/nearest dose donor vs log-dose 가중 donor.

서로 다른 지원집단의 오차를 단순 빼서 “시간이 실패의 50% 원인”이라고 말하지 않는다. cell·dose·compound·gene panel을 동시에 맞출 수 없는 축은 원인 분리가 불완전하다고 명시한다.

시간 연산자는 fit parent에서만 적합한 저랭크 ridge로 시작한다. rank `{4,16,32}`와 ridge 두 값 이내, 총 6설정을 기본으로 한다. 직접 24 h donor가 있다면 그것이 필수 비교군이다. PHH microarray를 쓸 경우 별도 플랫폼 연산자로 시작하며, 48/72 h 반복 노출을 단회 노출 시간축에 섞지 않는다.

**종료:** 주 비교 T3−T1의 미통과와 diagonal 전체의 종료를 구분한다. T2/T3 모두 T1 대비 유용한 개선이 구간상 배제되면 해당 diagonal recipe를 종료한다. T2만 유망하면 보조 탐색 결과로 새 확인을 요청하고, 구간이 넓으면 `INCONCLUSIVE`다. source 시간 연산자도 identity/scalar 대비 유용한 이득이 배제되면 종료한다. 단순 비유의성을 무효 입증으로 바꾸지 않으며, 이 결정이 X/S/L 종료를 뜻하지 않는다.

## 12. Context 실험 X0–X3

**가설:** 세포 이름을 기억하거나 control의 plate 잡음을 따라가는 것을 넘어서, 기저 상태가 약물별 반응 차이를 설명한다.

HEPG2 단일 세포주 안에서 잘 된 것만으로 이 가설을 검증할 수 없다. 기존 보유 LINCS의 적격 다중 세포주에서 시작하며, 가능하면 최소 4 source cell context를 확보한 뒤 leave-one-cell-context-out을 수행한다. 세포주 수·독립 plate·공통 약물 support가 부족하면 일반화 주장은 보류한다.

| ID | 입력/모델 | 비교 목적 |
|---|---|---|
| X0 | 현재 R17의 fit-control PCA와 FiLM, 같은 공정 선택 | 현재 맥락 baseline |
| X1 | 약물+cell ID 또는 약물+전역 평균 control, capacity matched | control 정보가 단순 세포주 식별 이상인지 |
| X2 | pre-treatment 기저 RNA/fit-control PCA를 통한 조건화, cell ID 제거 | 실제 새 세포주에서도 쓰는 맥락 표현인지 |
| X3 | B에서 단순 donor 평균 vs control 유사도 가중 vs source-fit basal 선형 보정 | 같은 약물 transfer의 맥락 적응이 단순 전이보다 나은지 |

X0/X2의 차이는 실제 입력 provenance로 명세한다. 둘이 같은 입력·함수라면 중복 실행하지 않는다. 새로운 기저 RNA나 기능성 맥락 변수가 없으면 X2를 “새 방법”으로 이름만 바꾸지 않는다.

X1의 새 세포주 ID는 학습된 embedding이 없으므로 `unknown/OOV = fit 평균 embedding` 또는 고정 unknown token 규칙을 학습 전에 정한다. held-out 세포주의 처리 label로 ID embedding을 새로 적합하지 않는다. X3의 query별 donor 통계·기울기는 B에서 허용한 source 측정으로 계산하는 입력 변환임을 기록하고, 공유 hyperparameter는 다른 fit parent에서 고정한다.

필수 대조:

- 약물과 context를 각각 제거한 재학습 모델.
- parent 및 허용된 context block 내 shuffle. 추론 시 0으로 만드는 OOD 분석은 별도다.
- 동일 약물에 대한 cell×drug interaction 오차. source 공통 약물효과를 맞춘 것과 cell-specific 차이를 맞춘 것을 분리한다.
- source basal-only vs basal+허용된 CNV/단백질 요약. 추가 뷰는 학습·시험 모두 같은 가용성 표를 가진 경우에만 보조로 사용한다.

Rhaister-O류는 X3의 **B 기준선**으로 고려할 수 있다. treatment별 source 반응을 필요로 하므로 모든 곳에서 RNA가 없는 A의 해법으로 재분류하지 않는다. Target cell DMSO-only와 query drug response-free는 다르다. [L4]

Hoshida·EMT·세포주기·경로 상태를 쓰면 **처리 전** 자료에서 계산하고 기준을 fit-only로 고정한다. 처리 후 cell-cycle나 DE를 context로 주지 않는다. 돌연변이·HBV·TERT 정보는 검증된 Cellosaurus/원자료 accession으로 고정하며 미측정을 wild-type으로 채우지 않는다.

Tahoe의 같은 well에 있는 여러 세포주를 train/test에 나누는 것만으로 독립 context 검증이라고 하지 않는다. primary context 판정은 물리 측정이 독립인 source 자료에서 하고, Tahoe는 별도 well/parent 통제 규약의 기술 검증으로 둔다.

## 13. 구조·출력 공간 실험 S0–S3

**가설:** 허용된 입력에 신호가 있을 때, 소형 잔차 구조나 출력 표현이 같은 예산으로 그 신호를 더 잘 학습한다.

입력·anchor·split·학습 손실·선택·seed를 고정하고 한 요소만 바꾼다. 아래 기본 구조 실험은 다중뷰 도입 성공을 기다리지 않고 기존 A/B 입력에서 수행한다.

**S/L의 주 채택 비교는 동일 DEV 규약으로 재적합한 B_R17 전체 결합이다.** anchor와 kernel을 유지한 채 신경망 branch만 교체하고, blend 계수는 모든 군에 같은 inner 선택 규약으로 재선택한다. component-only 성능은 기여를 설명하는 진단표다. 단독 head가 좋아져도 전체 결합이 강한 기준선보다 나쁘면 주 후보로 채택하지 않는다.

| ID | 구조/출력 | 비교 목적 |
|---|---|---|
| S0 | 현재 FiLM residual + full-gene vector head | 공정 MSE 기준 |
| S1 | parameter/계산량을 맞춘 concat 또는 dense residual | FiLM 조건화의 구조상 증분효과 |
| S2 | 동일 backbone의 방향·크기 분리 출력 | vector head 대비 parameterization 효과 |
| S3 | S0 backbone + fit-only PCA decoder, rank `{32,64,128}` 중 지원 가능한 값 | 출력 압축이 예측을 개선하는가 |

S2는 `p=m·u/(||u||+ε)`, `m=softplus(h)`처럼 비음수 magnitude와 direction을 분리하되, 처음에는 같은 MSE로 학습한다. angular loss까지 동시에 바꾸면 구조 효과가 아니다. magnitude를 관측 norm에 강제 맞추지 않는다. 별도의 L2와 결합할 때는 J1의 2×2로 분리한다.

S3는 basis를 inner-fit/outer-fit에서만 적합하고 차원을 내부 예측 성능으로 선택한다. 같은 held-out truth를 basis에 사영한 재구성치는 **표현의 사후 진단**으로만 둔다. 그 결과로 rank를 다시 고르지 않는다. 높은 VE는 예측 성공 기준이 아니다.

정적 graph basis는 기존 구현과 자산이 있으면 저비용 비교군으로 남길 수 있다. “반드시 4배 VE로 이겨야 한다”는 식의 외부 논문 수치 문턱은 두지 않는다. 간 경로만으로 주 출력을 제한하지 않는다.

**보고:** matched 모델 크기·실제 학습량·추론량, full-gene 오차, contrast, norm, chemistry shuffle, context shuffle. latent가 퍼졌다는 것만으로 약물 생물학을 배웠다고 판단하지 않는다.

## 14. 손실 실험 L0–L3

**가설:** 같은 입력과 구조에서 약물 간 차이 또는 방향을 직접 학습하면, 공통 평균에만 의존하는 해를 넘어선다.

기본 비교는 S0의 같은 backbone과 vector head, 같은 MSE 기준선에서 시작한다. 기존 R17 FiLM 적합이 호출하는 L0는 MSE 계열이다. 선택 목적함수는 학습 손실과 별개로 동일하게 유지한다. R12의 과거 유사 실험과 다른 점은 교정된 control 역할·공정 선택·현재 LINCS/B 입력 계약이다.

| ID | 학습 목표 | 가설 |
|---|---|---|
| L0 | parent 균등 MSE | 유효한 기준선. 음성 대조로 낙인찍지 않음 |
| L1 | MSE + matched drug-contrast loss | 약물 간 상대 반응이 추가로 학습되는가 |
| L2 | MSE + 안정화한 방향 손실 | 크기와 별개로 약물 특이 방향이 개선되는가 |
| L3 | 같은 가중 규약의 Huber/SmoothL1 | 소수 강한 반응·outlier가 지배하는 현상을 줄이는가 |

L1의 pair는 같은 cell·assay·time·dose·허용 plate/block의 서로 다른 parent로 정의한다. 비교할 약물이 충분하지 않은 pool은 mask하고 support를 보고한다. pool 중심화가 다른 biology를 지우는지 절대오차와 함께 본다. PDF의 Bison 전체 모델을 재현한 것이 아니라 **matched-contrast 원리를 시험한 경량 ablation**으로 명명한다.

완전한 pool 내 모든 쌍의 pair-MSE는 centered residual MSE와 비례할 수 있다. 따라서 이 항은 없는 정보를 만들어내는 것이 아니라 약물 대비를 더 강조하는 **목적함수 재가중 가설**이다. 기존 contrast loss와 같은 함수이면 중복 실험하지 않는다.

예시 목표:

$$
\mathcal L=\mathcal L_{\mathrm{MSE}}+\lambda_c\frac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}\left\| (\hat y_i-\hat y_j)-(y_i-y_j)\right\|^2.
$$

L2의 cosine 항은 거의 영벡터인 반응에서 불안정하다. norm ε와 신뢰도 가중은 train/source에서 정하고, 유효 방향 support·영벡터·NaN 처리수를 모두 낸다. 검증·시험의 실제 효과가 작은 행을 임의 제거하지 않는다.

각 추가 손실 가중은 `{0.01,0.1,1}` 등 사전 작은 grid 하나로 제한한다. 특정 논문의 가중 10 또는 γ=1.25를 최적값으로 가정하지 않는다. L3의 전환 크기는 fit residual scale로 정한 작은 후보군에서 고른다.

**판정:** 대비·방향만 좋아지고 원 주 오차가 악화하면 “방향 목적의 trade-off”로 보고한다. 이를 전체 예측기 개선으로 채택하지 않는다. Full−Shuffle가 없고 norm만 커지면 비붕괴 성공이 아니다. VICReg 등의 강제 분산 항은 필수 확장이 아니며 추가 검증 이유가 있을 때만 별도 등록한다.

## 15. 입력 정보 확장 I0–I4

**가설:** 구조로 설명하기 어려운 활동 차이가 있을 때, 이미 알려진 표적·기전·형태학·민감도 정보가 약물별 잔차를 더 잘 예측하게 한다.

이 트랙은 “SMILES만 있는 신규 화학”과 “RNA는 없지만 다른 실험 정보는 있는 약물”을 분리한다. 먼저 기존 저장소·승인된 작은 feature로 coverage를 조사한다. 가용성 문턱은 결과를 보기 전에 정하고, 예를 들어 공통 지원 ≥100 parent·outer fold별 ≥20 parent를 기본 실행 적격성으로 제안한다. 이 수가 통계적 충분조건은 아니다.

첫 I-arm은 anchor와 FiLM branch를 유지하고 kernel의 입력 뷰 하나만 변경한다. 전체 결합 대 B_R17을 같은 coverage에서 주 비교하며, kernel-only 비교는 진단이다. 신경망 입력까지 동시에 확장하는 모델은 이후 별도 arm으로 등록한다.

| ID | 추가 정보/변경 | 필수 paired 비교 |
|---|---|---|
| I0 | 구조-only kernel/FiLM, 기존 인코더 고정 | 공통 기준 |
| I1 | 표적·MoA·정량 bioassay 단일 뷰 | I0, annotation-only, mask-only, parent shuffle |
| I2 | Cell Painting 단일 뷰 | 같은 화합물 I0, morphology-only, mask-only, parent shuffle |
| I3 | I1/I2 중 가용한 뷰의 결측-aware 결합 | 각 단일뷰, zero-fill+mask, leave-one-view-out |
| I4 | 승인·기보유된 PRISM/GDSC/LIMORE 지문 중 사전 지정 한 뷰 | I0/I3의 동일 support, target-endpoint 차단, shuffle |

I1에서는 assay type·relation·단위·표적 종·단일단백질/복합체·confidence를 함께 저장한다. pChEMBL 값이 있다고 모든 assay가 비교 가능한 것은 아니다. 여러 관측을 합치는 규칙은 fit-only이며, 미측정은 “비활성 0”과 구분한다. 작용 방향 annotation과 affinity 강도도 별도 열이다.

I2에서는 source cell·시간·농도가 target RNA와 다름을 표기한다. U2OS 형태학을 HCC의 실측 표현형처럼 설명하지 않는다. Plate correction이 전체 query 실측을 사용했는지 감사한다.

I3의 안전한 출발점은 PSD가 보장되는 masked feature 결합이다. 예를 들어 각 정규화 뷰 φ_v에 대해:

$$
\psi(d)=\bigoplus_v \frac{m_v(d)\sqrt{w_v}\,\phi_v(d)}{\sqrt{\sum_u m_u(d)w_u+\epsilon}},\qquad K(d,d')=\psi(d)^\top\psi(d').
$$

이 구현은 저자의 UnDetect와 동일하다고 주장하지 않는 **결측-aware baseline**이다. 구조 뷰를 항상 fallback으로 둔다. 원 공식 kernel을 도입한다면 정의·대칭성·최소 고유값·정규화·수치 안정화까지 확인한다. 쌍마다 공유뷰 수로 나눈 행렬이 자동으로 PSD라고 가정하지 않는다.

I4의 query HCC 민감도를 이미 입력했다면 같은 수치로 항암 민감도 외부검증을 하지 않는다. RNA prediction과 viability translation은 별도 endpoint다.

모든 I-arm에 다음 4개 표가 있어야 한다.

1. 동일 coverage에서의 paired 효과.
2. 전체 대상에서 구조-only fallback까지 포함한 실제 성능.
3. 뷰 없음/부분/완전 가용 strata와 unknown-parent/scaffold strata.
4. 가용성 mask만 사용한 성능과 값 shuffle 성능.

실측 뷰가 없으면 실패가 아니라 `BLOCKED_VIEW_COVERAGE`다. 정보 이득이 없다는 결과가 나왔다고 바로 대형 인코더를 사전학습하거나 포켓 수를 늘리지 않는다.

## 16. 제한된 결합 J1과 조건부 확장 E1–E5

### 16.1 J1: 2×2로 결합 원인을 확인

처음부터 모든 축을 이어 붙이지 않는다. DEV 규약에서 승격된 **두 변경만** 선택해 다음을 비교한다.

| | 두 번째 변경 없음 | 두 번째 변경 있음 |
|---|---|---|
| 첫 번째 변경 없음 | 공통 기준 | 단독 2 |
| 첫 번째 변경 있음 | 단독 1 | 결합 |

첫 기본 후보는 `방향·크기 분리 S2 × 방향 손실 L2`, 또는 `유효 입력 I × matched loss L1` 중 source/inner에서 사전 지정한 한 쌍이다. R17 커널·anchor의 정의는 그대로 둔다. 각 단독 효과와 difference-in-differences를 보고한다. 결합만 좋아졌다면 상호작용 증거와 모든 단독 결과를 공개한다.

최종 ensemble/stacking은 outer-held label로 계수를 정하지 않는다. fit 안의 OOF prediction으로만 weight를 적합한다. 과거 외부 점수를 본 뒤 weight를 맞추면 그 자료는 개발자료로 기록한다.

### 16.2 E1: DE/반응 확률 head — 독립 반복이 있을 때

- 실제 독립 처리·대조 반복과 batch 구조를 확보한 조건만 사용한다. 기본 적격성은 조건당 ≥3 독립 물리 반복으로 제안하되 설계행렬의 estimability도 확인한다.
- count 기반 자료는 적합한 pseudobulk 통계 모형, 연속 L1000·microarray는 그 관측척도에 맞는 모형을 사용한다.
- 학습용 DE label은 fit 자료에서만 만든다. 예측 pDE는 생물학적 진리의 확률과 동일하다고 가정하지 않는다.
- vector head vs 추가 activity/direction/magnitude head를 matched 비교한다. Brier/PR-AUC·부호·효과크기와 기본 RMSE/R²를 함께 본다.
- 반복이 없으면 continuous soft response score로 다른 문제를 정의하거나 보류한다. 이를 원 PIE 재현이라고 부르지 않는다.

### 16.3 E2: 잔차 flow/생성기 — 평균 이상의 필요성이 있을 때

과거 R3의 평균 고정 생성기·pilot 결과를 먼저 재검토한다. 새 자료/새 신호 없이 같은 생성기를 반복하지 않는다.

진입 조건은 (a) 독립 well의 단일세포 분포 정답, (b) 평균을 맞춘 control shift/Gaussian baseline으로 설명되지 않는 재현 가능한 분포 차이, (c) 고정 평가 통계·예산이다.

평균 예측을 모든 군에 같게 고정하고 control shift, Gaussian/empirical residual, 소형 control-anchored residual flow를 비교한다. 세포 split은 같은 well 내부 변동을 나타낼 뿐 독립 생물학적 반복이 아니다. 합성세포 400개를 고정하는 이유가 없다면 데이터/평가 목적에 맞게 미리 정하되, 생성 수·분산을 target DEG 수에 맞춰 최적화하지 않는다.

### 16.4 E3: 공개 transfer 모델과 genetic bridge — 입력 계약 충족 시

| 후보 | 진입 조건 | 최소 비교 | 현재 계획상 지위 |
|---|---|---|---|
| AetherCell | 공식 checkpoint·license·학습 노출, 실제 10,085-gene 맥락 및 모드별 control/gene order 확보 | frozen model vs 같은 지원 B_R17/T1 | 조건부 benchmark. 세포 이름이나 161유전자만으로 대체 불가 |
| Rhaister-O | treatment-specific source support와 baseline 맥락 | X3의 simple transfer 대비 | B의 경량 기준선으로 우선 고려 |
| CREST | 원문·공식 코드·reference atlas 계약 확인 | 같은 atlas의 NN/weighted mean | 이번 검토에서 독립 원문/자산 확인 불충분; 필수 경로 아님 |
| GRAFT류 | target genetic perturbation과 source genetic/chemical bridge 확보 | basal-only / genetic-context / genetic-context shuffle | G-context 별도 트랙. SNU-423 자산이 없으면 그 맥락은 실행 불가 |
| quantile/MMD 정렬 | 실제 matched bridge, 관측모형과 semantics 검증 | identity/scalar/저랭크 선형 | 전체 marginal distribution만 맞추는 대형 정렬은 후순위 |

공식 코드가 안 읽히거나 가중치/라이선스/데이터가 불명확하면 agent는 임의 구현을 원 방법 재현으로 보고하지 않는다. 소량 target chemical 라벨을 쓰는 Adapt는 DMSO-only 실험과 별도다.

CREST류에서 target-platform의 다른 약물 treated atlas를 쓰면 `target-platform-treated-reference` 정보 계약을 추가한다. query parent·동일 physical well·금지된 파생 서명은 제외하고, 같은 atlas를 허용한 단순 비교군을 둔다. DMSO-only 또는 완전 target-context-zero-shot 표에 섞지 않는다. Genetic bridge는 A+G/B+G를 구분한다.

### 16.5 E4: 구조 유도 proxy embedding·Boltz — 작은 pilot부터

InfoAlign/PhenMol/CLAMP 등은 checkpoint의 사전학습 노출과 목적 과제를 먼저 감사한다. InfoAlign 공식 자료에는 형태학뿐 아니라 LINCS가 포함되므로 “추론 때 SMILES만 준다”는 이유로 엄격 A에 자동 포함하지 않는다. [L5]

Boltz-2는 **10–20약물×3–5표적**으로 실행성·메모리·시간·실패율을 먼저 측정한다. 표적 패널은 source 생물학과 사전 목적에 따라 고정한다. 공개 구조 기반 인코더 및 간단 target/MoA 대비 증분정보가 없으면 확대하지 않는다. 학습 데이터에서 동일 scaffold/target의 친화도 노출도 기록한다.

단위 검사: 공식 `affinity_pred_value`는 `log10(IC50 in µM)`이며 pIC50와 강도 방향이 반대다. 같은 IC50 정의일 때 `pIC50=6−value`다. binding probability와 동일한 값이 아니며, 둘 모두 세포 내 표적 점유·작용 방향·ΔRNA를 직접 알려주는 실측치가 아니다. 외부 MSA 서버 전송이 필요한 실행은 대상 데이터와 사용 권한을 먼저 확인한다. [L7]

### 16.6 E5: 프로그램 출력·경로 보조·다중 panel

- EBMF 또는 cNMF는 PCA가 유망하고 프로그램 해석에 별도 목적이 있을 때 한 가지씩 추가한다. cNMF를 signed Δ에 그대로 넣지 않는다. 비음수 발현에서 프로그램을 적합한 뒤 사용량 차이를 예측하거나 signed factorization을 쓴다.
- 경로 보조 손실은 같은 backbone의 `없음/있음/유전자 대응 셔플` 3군으로 시험한다. PROGENy/CollecTRI/TXG-MAPr의 버전·유전자 coverage·가중·스케일을 고정한다. regulon은 서로 중첩될 수 있으며 ULM/MLM 선택을 생물학적 진실로 설명하지 않는다.
- gene-query decoder는 충분한 학습 panel overlap이 있을 때만 별도 시험한다. 측정 안 된 gene을 0정답으로 채우지 않는다. 공통 panel 성능과 native panel 성능을 분리한다.
- raw count NB decoder를 L1000 L3 연속값이나 PHH microarray에 그대로 적용하지 않는다.

조건부 E군이 모두 실행되지 않아도 필수 최소 검증이 미완료인 것은 아니다. 각 항목의 자산·정보 계약·추가 효용 부족을 문서화해야 한다. 반대로 “논문이 최신이므로 필수”라는 이유만으로 예산을 늘리지 않는다.

## 17. 일정·계산 예산·수집 한도

### 17.1 제안 예산

아래는 **예약된 자원이나 실측 실행시간이 아니라 승인용 상한 제안**이다. 실제 GPU 종류·가용 슬롯·기존 작업·스케줄러 정책을 먼저 확인한다. 8×A40이 실제 사용 가능하다고 가정해 시작하지 않는다.

| 묶음 | 제안 GPU-hour 상한 | 실행 방식 |
|---|---:|---|
| D0–D3 감사·기준선·원인 규명 | 16 | 대부분 CPU, 필요한 소형 재학습만 GPU |
| R18 고정 확인 | 32 | 기존 규약·job 여부 확인 후, fold/role별 독립 작업 |
| T/X shift·context | 32 | 경량 closed-form/선형부터, 최대 소형 FiLM |
| S/L 구조·손실 | 48 | 축별 ≤8설정, 1 seed 선별 후 지정 후보만 3 seed |
| I 입력 뷰 | 32 | 기보유/승인된 처리 feature, kernel·소형 잔차 |
| J1 및 승격 후보 확인 준비 | 32 | 지정 2×2, 선택된 후보만 |
| **기본 개발·확인 상한 합계** | **192** | 실제 자산·지원·pilot에 따라 미사용분은 남김 |

E군은 위 예산에 자동 포함하지 않는다. 예를 들어 소형 flow 추가 ≤96 GPU-hour, 구조 proxy/affinity pilot ≤32 GPU-hour 등을 **별도 승인안**으로 올린다. 실패한 arm의 남은 예산을 미등록 대형 탐색으로 전환하지 않는다.

처음은 CPU/1 GPU smoke test로 시작하고, 기본 동시 실행은 최대 2 GPU를 제안한다. 공유 cluster 정책과 자원이 허용될 때 독립 fold/seed 중심으로 늘리되 최대 8개를 넘지 않는다. 작은 모델에 DDP를 쓰는 것보다 독립 작업이 효율적인지는 실측 후 결정한다.

### 17.2 메모리·시간 측정

- 150,000행의 float32 dense kernel 하나는 약 **90 GB**다. 원자료 텐서가 작아도 커널·복사·분해가 메모리를 지배할 수 있다.
- 8×48 GB는 일반 DDP에서 하나의 384 GB 단일 장치 메모리가 아니다.
- 프로세스별 peak RSS, GPU peak allocated/reserved, wall time, GPU-hour, 입력/중간/출력 디스크를 기록한다.
- 큰 kernel은 block 계산·Nyström·feature-space ridge 등 근사 전략을 **명세 후** 사용하고 dense small-subset과 오차를 검증한다.
- 새 수집은 metadata와 필요한 처리 feature를 우선한다. 예상 신규 저장량이 50 GB를 넘거나 원시 수억 cell/이미지·controlled access가 필요하면 별도 데이터·비용 승인 없이 진행하지 않는다.

### 17.3 권장 일정

| 순서 | 완료 조건 | 예상 일정의 의미 |
|---|---|---|
| 착수 | HEAD/job/권한·노출·자산 재고, R14 결과 회수 여부 | 첫 보고. 원장 없이 모델 학습 시작 금지 |
| 1차 | D0–D2 및 R18 지원 감사·명세 동결 | 실제 지원 확정 뒤 자원 배정 |
| 2차 | R18 예측 동결, T/X/S/L 최소 비교를 병렬 진행 | 축 하나의 음성 때문에 다른 축 취소 금지 |
| 3차 | 필수 arm 결과·기여 대조·불확실성, 가능한 I 뷰 | 단일 요소 결과 정리 |
| 4차 | 지정 J1, 3 seed 확인, 최종 외부 자료 권한 점검 | 새 후보의 확인 가능/불가를 구분 |
| 후속 | E군 또는 새 HCC 측정 | 별도 근거·승인·예산이 있을 때 |

기보유 자산과 queue가 원활하면 2–4주 안에 기본 의사결정 패키지를 목표로 할 수 있다. 이는 보장 일정이 아니다. 6주를 “환자 수준 검증 완료”의 마감으로 약속하지 않는다. 결과가 나올 때마다 다음 네 문장으로 보고한다: **무엇을 물었나 / 무엇이 나왔나 / 어디까지 말할 수 있나 / 다음에 무엇을 하거나 안 하나.**

## 18. AI agent 구현·산출물 계약

### 18.1 기존 코드의 활용

기존 R17 control 역할 배정과 freeze/score 분리를 우선 재사용한다. 원본 R15–R17 출력과 잠금 artifact를 수정하지 않는다.

- 현재 계약 확인 대상: `src/r17_lincs/vehicles.py`, `predict.py`, `score.py`.
- 공정 선택 연결 대상: `src/r16_lincs/predict.py`, `src/r15_lincs/estimators.py`.
- contrast 불변성 확인 대상: `src/r13_optimization/objectives.py`, `src/r11_chemical_probe/metrics.py`.
- 기존 손실/구조 재사용 대상: `src/r12_multiaxis/network.py`.

위는 저장소 내 경로다. 새 실험 코드를 넣을 실제 폴더명은 기존 번호 충돌·진행 작업을 확인한 뒤 정한다. **아래 산출물 목록과 명세 키는 요구사항이며, 이미 존재하는 CLI나 완료된 파일이라는 뜻이 아니다.** 실행 agent는 실제 지원되는 명령을 구현·검증하고 그 명령을 원장에 남긴다.

### 18.2 최소 manifest 필드

각 run에 다음을 저장한다.

| 범주 | 필수 필드 |
|---|---|
| 식별 | run_id, lane, experiment_id, status, git_commit, protocol_hash, config_hash, seed |
| 입력 | dataset/version/source, data_hash, exposure_class, information_contract, original_assay_level |
| 물리 단위 | parent_id, raw_compound_id, cell_id, dose/unit, time, plate_id, well_id, role, biological/technical replicate |
| 특징 | gene_axis_hash, normalization, modality/missingness, pretraining_provenance, source query response permissions |
| 선택 | outer/inner IDs, fitted transforms, checkpoint/lr/gain, selection objective, tie rule |
| 예측 | row_ids, shape, finite check, prediction_hash, freeze_time, query-label absence check |
| 자원 | CPU/GPU model, peak memory, wall time, GPU-hour, job_id, exit status |
| 판정 | comparator, endpoint, eligible n, effect, CI level, multiple-testing label, reason, next action |

query RNA label을 predictor 프로세스에서 읽지 않는지 파일 접근·객체 경로를 검증한다. `score-only` 디렉터리에는 oracle gain/사영·DE reference를 넣고, fit/select에서 import할 수 없게 한다.

### 18.3 필수 테스트

1. 같은 parent가 금지된 fit/select/score 데이터에 중복되지 않음. B의 입력 예외는 명시적으로 검사.
2. 동일 물리 control well이 세 역할에 중복되지 않음.
3. 같은 처리 well의 파생 cell/파일을 독립 반복으로 세지 않음.
4. 모든 prediction이 `[n_rows,n_genes]`, finite, 동일 gene order이며 row hash가 맞음.
5. rank/PCA/normalization/mean/gain이 허용된 fold에서만 적합됨.
6. 동일 pool의 공통 shift에 대한 contrast 불변성, 비영 예측의 공통 양의 scalar에 대한 cosine/검색 불변성, RMSE/R²의 analytic 변화가 각각 맞음. scalar=0, norm threshold, tie 처리도 검사.
7. B0, constant prediction, perfect prediction, shuffled prediction에서 지표의 예상 동작이 맞음.
8. zero effect/empty DE/결측 view/unknown compound/zero norm의 처리를 검사.
9. 공통 subset paired 결과와 전체 fallback 결과의 분모가 서로 혼동되지 않음.
10. 새 코드가 R18 고정 주 분석을 바꾸지 않음. 바뀌면 새 protocol/version으로 분리.

`torch.use_deterministic_algorithms(..., warn_only=True)`는 완전 결정론 보증이 아니다. deterministic 설정·경고·비결정 연산·허용 오차를 기록하고, 동일 seed 재실행의 예측 해시 또는 수치 일치 허용치를 검사한다. library/driver/모델 asset commit을 함께 고정한다.

### 18.4 반드시 공개할 작은 산출물

- `PROTOCOL.md`: 질문, 허용 입력, 지원 조건, 주 비교, grid, 지표, 종료 기준.
- `experiment_matrix.tsv`: 필수/조건부 ID별 `PLANNED/RUNNING/COMPLETED/BLOCKED/NOT_APPLICABLE/FAILED_INFRA`.
- `support.tsv`, `exposure_manifest.tsv`, `modality_coverage.tsv`.
- `selection.tsv`, `points.tsv`, `paired.tsv`, `robustness.tsv`.
- `freeze.json`, `receipt.json`, `resource_usage.tsv`.
- `RESULTS.md`: 아래 18.5의 판정문과 모든 실패/보류 이유.
- `BEGINNER_SUMMARY.md`: 식·논문 이름 없이도 이해되는 실험별 한 문단.
- `NEXT_DECISION.md`: 다음 실험/중단/추가 데이터 요구와 그 근거.

대용량 prediction/checkpoint는 등록된 artifact 위치와 해시를 남기되, 작은 점수·CI·지원 원표는 결과 보고와 동시에 검토 가능하게 한다. R14처럼 “completed” ledger만 있고 효과를 검토할 수 없는 상태를 반복하지 않는다. 저장소 반영은 사용자의 실행·반영 권한 범위에서 수행한다.

### 18.5 실험별 결과 문장 템플릿

> **[ID / 상태]** 우리는 [구체적 병목] 때문에 [변경 하나]가 [동일 정보의 비교군]보다 나을 것으로 가정했다. [자료·허용 입력·독립 단위 n]에서 [효과크기와 구간]이 나왔다. [특이 대조]는 [결과]였다. 따라서 [가설 지지/미지지/불충분/실행 불가]이며, [적용 범위]까지만 말할 수 있다. 다음은 [확인/중단/명시된 추가 자료]다.

`좋아지는 경향`, `잠재력이 있다`, `다양한 요소가 작용했다`로 판정을 대신하지 않는다. 통과하지 못한 공동 기준이 있으면 정확히 어느 기준인지 적는다.

## 19. 왜 비슷한 음성 실험을 반복했으며, 이번에는 무엇이 다른가

| 반복 문제 | 이전 결과에서 보인 신호 | 이번 차단 장치 |
|---|---|---|
| 예측하려는 과제가 바뀌는데 같은 성공/실패로 요약 | 절대 발현·DMSO delta·약물 대비·동일약물 전이가 혼재 | 정보 계약 A/B/M/G와 endpoint를 고정 |
| 실제 약물 정보보다 맥락·plate 신호가 큼 | 위치 통제 뒤 이득 축소, 단순 평균/전이가 강함 | mean/context/position/anchor·shuffle 대조 |
| 선택 규약이 배포/평가 조건과 다름 | R16→R17에서 gain·순위 변화 | 물리 DMSO 3역할, D1 paired 인과대조 |
| 후보와 비교군의 튜닝 기회가 다름 | gain·checkpoint 선택 방식 비대칭 | DEV의 공동 선택과 동일 예산 |
| 관측 가능성이 낮은 것을 모델 크기로 보상 | HCC QC 지원 부족, cross-plate 반복 불안정 | 지원·검정력 선행, 소규모 층은 기술 분석 |
| 여러 부품을 동시에 바꿔 원인을 모름 | 복잡한 통합안의 사후 설명 증가 | 단일 요소 비교, 제한된 2×2 |
| 좋아 보이는 외부자료를 반복 재사용 | B1/R8/CMap, R14/MIX-seq/Emerald Bay 이력 | 노출 원장, 새 검증이라는 명칭 제한 |
| 성공 문턱을 남의 과제에서 가져옴 | VCC·다른 패널의 절대 지표 목표 | 강한 기준선·같은 지원·사전 실용 효과 기준 |
| 실행과 결과 공개를 혼동 | R14 원표 미게시 | receipt+원표+쉬운 판정문 동시 납품 |

이 계획은 “더 복잡한 가설을 쓰면 성공한다”는 해결책이 아니다. **변경이 실제로 새로운 정보를 주는지 먼저 식별하고, 그렇지 않으면 그 경로만 닫을 수 있게 만드는 것**이 핵심이다.

## 20. 가설 통과 가능성과 사용자가 결정할 것

정확한 통과 확률을 숫자로 추정할 근거는 없다. 아래는 현재 증거에 따른 상대적 우선순위다.

| 방향 | 현재 근거 | 권고 |
|---|---|---|
| R17 결합의 다른 측정 확인 | 교정된 내부 A/B 양성, 그러나 과거 외부 B_cell 대비는 미통과 | **가장 먼저 확인할 가치가 큼.** 성공을 전제로 하지는 않음 |
| 공정 선택·control 잡음·평균 대조 | 실제 순위 변화와 누락 대조가 관찰됨 | **정보 가치가 매우 높고 비용이 낮음** |
| source scalar 보정 | 기존 진폭 과대는 관찰됨 | 해볼 가치 있음. 기존 frozen vector 방향 한계 때문에 큰 성능 도약 기대는 낮음 |
| control-variance만으로 플랫폼 보정 | 처리효과 scale과 control variance의 연결은 미검증 | 작은 source-domain 검증만 우선. 낮거나 불확실한 사전 신뢰 |
| 실측 표적/형태학 다중뷰 | 추가 생물활성 정보가 있을 가능성 | coverage·노출이 해결되면 유망. 구조-only 성공과 구분 |
| context·구조·손실 단독 변경 | R12 음성이지만 새 계약·새 target에서는 다시 물을 이유가 있음 | 필수 최소 ablation은 진행. 대형 탐색 확대는 결과를 보고 결정 |
| full flow·환자 therapeutic window | 현재 독립 분포/치료 정답 근거가 부족 | 단기 필수 목표에서 제외 |

권장 의사결정은 다음 세 가지다.

1. **기본 계획 채택:** R18 고정 확인 + DEV 네 축 최소 실험 + 기존 결과 회수. 이 파일의 기본안이다.
2. **추가 데이터/예산 승인 범위:** 기보유 자산을 넘는 MoABox·형태학·checkpoint 수집, 192 GPU-hour 초과, E군은 별도 목록으로 승인한다.
3. **최종 목표의 우선순위:** 단기에는 “허용 정보가 있는 간 맥락의 약물 반응 예측”을 검증한다. “SMILES-only 신규 약물의 성인 HCC·환자 예측”은 더 강한 별도 목표로 유지하되 이미 달성한 것처럼 표현하지 않는다.

독립 성인 HCC 확인자료가 끝내 부족하면, 필요한 추가 실험은 모델 이름이 아니라 **독립 HCC 맥락·물리 반복·무작위 위치·시간/용량 대응**으로 정의한다. 새 실제 측정 기획은 별도 승인 후 만들며, 여기서는 확보되지 않은 환자/오가노이드 정답을 있는 것처럼 가정하지 않는다.

## 21. 실행 agent에게 전달할 요청문

아래 요청은 실제 실행 권한을 부여받은 agent에게 전달하는 용도다. 현재 문서 작성만으로 실행이 시작된 것은 아니다.

> GS / vcell_liver 후속 연구를 이 기획서에 따라 진행하라. 먼저 현재 HEAD, AGENTS.md, 진행 중 작업, 최신 R18 상태를 확인하고 중복 실행하지 말라. 원 R15–R17과 봉인·격리·예약 자료는 보존하라.
>
> 첫 보고는 (1) 검토 기준 이후 변경, (2) R14 기존 동결 원표 회수 가능 여부, (3) R18 C1–C3의 노출/QC/물리 DMSO 3역할 지원, (4) 실제 자원·필수 자산, (5) 실행 가능한 D/T/X/S/L/I ID 목록으로 작성하라.
>
> CONFIRM-R18은 기존 요청의 주 후보 커널+FiLM과 R17 선택·격자·평가 규약을 그대로 유지하라. C1의 B1/R8/R16 노출 이력을 명시하고 완전 blind 외부평가라고 부르지 말라. 지원 감사 → 명세 고정 → 예측 동결 → 별도 채점 순서를 지켜라.
>
> DEV는 물리 DMSO fit/select/score를 분리하고 checkpoint×gain의 공정 선택을 사용하라. fit-only mean, context-only, B의 blend_anchor를 추가하여 약물 잔차의 추가 가치를 확인하라. D1의 동일 control 표본수 비교를 통해 R17 변화의 가능한 원인을 분리하라.
>
> shift T, context X, 구조 S, 손실 L의 필수 최소 비교를 모두 처리하라. T가 실패해도 X/S/L을 중단하지 말라. 각 축의 입력·backbone·손실·출력·평가 중 바뀌는 요소를 하나로 제한하라. 자료가 부족하거나 변환이 불변이면 구체적 증거와 함께 BLOCKED/NOT_APPLICABLE로 기록하라.
>
> 외부 처리 정답으로 gain·γ·gene scale·donor·경로·rank를 고르지 말라. Target oracle와 score-control 사영은 score-only 진단이다. 기존 R15-B의 oracle scalar 0.0764를 실사용 설정으로 복사하지 말라.
>
> 입력 정보 A/B/M/G를 섞지 말라. 신규 multimodal view는 동일 coverage 비교, 전체 fallback, mask-only, parent-shuffle를 포함하라. InfoAlign 등의 사전학습 노출을 감사하고, 불명확한 encoder를 엄격 unseen-parent 확증 표에 넣지 말라.
>
> 독립 반복 없는 DE, 합성세포 수를 이용한 유의성 강화, 과도한 norm 강제, target outcome을 본 뒤 cohort 제거를 하지 말라. R17 결합을 필수 강한 기준선으로 유지하라.
>
> 기본 예산은 제안 상한 192 GPU-hour이며, 실제 승인·자원 정책을 우선한다. 기본 동시 작업 2 GPU에서 시작하라. E군, 대용량 다운로드, controlled access, 새 비용, 봉인 자료 사용, 저장소 push는 해당 권한을 확인한 뒤 진행하라. 부족한 권한은 우회하지 말고 요청하라.
>
> 완료 보고는 모든 ID의 상태·효과·99% 구간·지원·노출·비용·다음 결정을 포함하라. 성공과 미통과뿐 아니라 불충분·실행 불가도 정확히 구분하라. 작은 원표·선택표·freeze/receipt를 결과 문서와 함께 제공하라. 각 실험을 “가설 → 결과 → 된 것/안 된 것 → 다음 결정” 네 문장으로 쉽게 설명하라.

## 22. 최종 완료 체크리스트

- [ ] 검토 이후 Git 변경과 R18 중복 실행 여부 확인.
- [ ] R14 외부 원표 회수·해시 대조 또는 접근 blocker 기록.
- [ ] R17 핵심 점수·선택·DMSO 역할 계약 대조.
- [ ] R18 C1–C3 지원/노출 명세와 원 규약 보존.
- [ ] B_mean/B_context/B_blend_anchor 및 공정 선택 대조.
- [ ] DMSO 공유/분리의 동일 표본수 비교.
- [ ] T0–T4, X0–X3, S0–S3, L0–L3 각각 결과 또는 구체적 비실행 근거.
- [ ] I0–I4 coverage·정보 계약·mask/shuffle 대조 또는 자산 blocker.
- [ ] J1 및 E1–E5 진입 여부와 이유.
- [ ] 모든 후보의 같은 row/gene/정규화 및 예산 공정성.
- [ ] parent와 plate/well의 독립성·반복 수 표기.
- [ ] target-treated/score-control이 선택·변환으로 역류하지 않는 테스트.
- [ ] 효과크기·CI·다중검정 지위·실용성·소수 parent 기여 보고.
- [ ] raw/contrast/검색/기전/DE/분포 endpoint를 분리.
- [ ] R18 확인과 DEV 후보 개발을 혼동하지 않은 최종 판정.
- [ ] 결과·선택·지원 원표 및 해시·자원 원장과 쉬운 요약 제공.

## 부록 A. 검토한 저장소 근거

아래 링크는 모두 검토한 `f1c087a6da365a6aebac5315ced640aa1a156692`에 고정했다. 이 문서는 보고된 CI를 코드·작은 원표와 대조한 검토다. 서버의 전체 원자료·checkpoint를 독립 재실행한 전량 재현 감사는 아니다.

- **[G1] R17 결과와 수치:** [RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/RESULTS.md), [points](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/data/score/points.tsv), [paired](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/data/score/paired.tsv), [selection](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/data/score/selection.tsv).
- **[G2] R17 계약과 구현:** [PROTOCOL](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/PROTOCOL.md), [vehicles.py](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r17_lincs/vehicles.py), [predict.py](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r17_lincs/predict.py), [score.py](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r17_lincs/score.py).
- **[G3] 완료된 R16:** [RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r16_lincs/RESULTS.md), [CONFIRMATION](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r16_lincs/CONFIRMATION.md), [선택 구현](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r16_lincs/predict.py).
- **[G4] 최신 자료 재고:** [INVENTORY](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/INVENTORY.md), [시간·용량별 미기록 QC](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r17_lincs/data/inventory/qc_no_recorded_overlap_conditions.tsv).
- **[G5] 기존 R18 요청:** [CLAUDE_REQUEST_R18_CONFIRMATION_2026-11-21.md](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/claude_review_20261121/requests/CLAUDE_REQUEST_R18_CONFIRMATION_2026-11-21.md).
- **[G6] 선행 다축/확대 실험:** [R12 RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r12_multiaxis/RESULTS.md), [R12 matrix](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r12_multiaxis/experiment_matrix.tsv), [R13 RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r13_optimization/RESULTS.md).
- **[G7] R14 완료 기록·자료:** [execution ledger](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r14_cross_organ/data/execution_v1_jobs.json), [DATA](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r14_cross_organ/DATA.md).
- **[G8] 위치·외부·세포주 특이성:** [position](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/tahoe_position/RESULTS.md), [P1–P3](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/p123/RESULTS.md), [P4](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/p4/RESULTS.md).
- **[G9] 외부 방향·크기 진단:** [R15 RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r15_lincs/RESULTS.md), [direction_scale](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/r15_lincs/data/external/descriptive/direction_scale.tsv), [계산 코드](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r15_lincs/external_directions.py), [pool 중심화](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/src/r13_optimization/objectives.py).
- **[G10] 과거 생성기·초기 연구의 범위:** [R3 RESULTS](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/anchored_distribution_r3/RESULTS.md), [pilot](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/reports/anchored_distribution_r3/PILOT_GENERATORS.md), [README의 이력](https://github.com/honbul/vcell_liver/blob/f1c087a6da365a6aebac5315ced640aa1a156692/README.md).

## 부록 B. 핵심 1차 출처와 이번 검증 범위

모든 웹 확인은 2026-11-21 기준이다. 아래 링크가 있다는 것만으로 우리 데이터에서의 재현이나 성능을 확인했다는 뜻은 아니다. 미검증 세부 수치를 성공 목표로 사용하지 않았다.

| ID | 1차 출처 | 확인한 범위와 남은 조건 |
|---|---|---|
| L1 | [Arc VCC 2026 공식 결과](https://arcinstitute.org/news/virtual-cell-challenge-2026-wrap-up) | CRISPRi 새 맥락 전이 과제, reference transfer와 지표 해석 확인. chemical de novo 성능 근거 아님 |
| L2 | [cell-eval2 공식 metric 문서](https://github.com/ArcInstitute/cell-eval2/blob/main/docs/metrics.md) | metric/version·target exclusion·normalization 계약 확인. 프로젝트 adapter는 아직 실행 검증하지 않음 |
| L3 | [AetherCell 공식 코드](https://github.com/Wenyuan-AI4science/AetherCell) | 실제 RNA context·control 입력과 자산 링크 확인. checkpoint 학습 노출·실제 로컬 입력 적격성 추가 감사 필요 |
| L4 | [Rhaister](https://huggingface.co/tahoebio/Rhaister), [zero-shot 구조](https://huggingface.co/tahoebio/Rhaister/blob/main/docs/zeroshot_architecture.md) | target basal-only와 source treatment-specific 구조 확인. 기존 3역할/parent 계약 성능은 미검증 |
| L5 | [InfoAlign 공식 코드](https://github.com/liugangcode/InfoAlign) | checkpoint와 morphology/LINCS/Hetionet 학습 소스 확인. query response 미노출은 별도 검증 |
| L6 | [MAP 공식 코드](https://github.com/MAGIC-AI4Med/MAP) | KG/모델 파이프라인과 대규모 데이터 요구 확인. 전체 파이프라인을 필수 도입하지 않음 |
| L7 | [Boltz 공식 코드](https://github.com/jwohlwend/boltz) | affinity 출력 단위·용도·license 확인. ΔRNA 증분효과 미검증 |
| L8 | [UnDetect 원 프리프린트](https://www.biorxiv.org/content/10.64898/2026.11.05.764505v1) | context 분리·multi-view 원리와 저자 보고 확인. 공식 코드 실제 설치·수치 재현은 미확인 |
| L9 | [Bison 원 프리프린트 주소](https://arxiv.org/abs/2609.32467) | 첨부 PDF의 matched-contrast 설계·서지 참고. 이번 원문 접근이 제한돼 세부 구현·성능은 독립 검증하지 못함. 여기의 경량 손실은 자체 가설이며 원 모델 재현이 아님 |
| L10 | [Centered cross-context transfer](https://www.biorxiv.org/content/10.64898/2026.10.20.760514v1) | training targets를 사용하는 과제 확인. γ=1.25를 DMSO-only 보정의 검증된 상수로 채택하지 않음 |
| L11 | [GRAFT 원 프리프린트](https://www.biorxiv.org/content/10.64898/2026.10.27.762107v1) | genetic-context를 활용한 chemical transfer와 추가 target chemical 적응의 구분. 세부 구현·자산은 착수 전 확인 |
| L12 | [MoABox 기탁](https://zenodo.org/records/14291446), [Novartis 공식 경로](https://github.com/Novartis/DRUG-seq/tree/main/data/Novartis_drugseq_U2OS_MoABox) | 별도 공개 데이터·파일 목록 확인. 새 독립 cohort 적격성이나 fold-safe preprocessing이 자동 보장되지 않음 |
| L13 | [GSE176150](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176150) | U2OS 24 h·14 compounds·8 doses인 별도 자료임을 확인 |
| L14 | [ALTEX 원논문](https://www.altex.org/index.php/altex/article/view/2694), [저자기관 원문](https://scholarlypublications.universiteitleiden.nl/access/item%3A3765768/view) | E-MTAB-12668의 좁은 약물·노출 설계 확인. EBI 실제 파일 내용/다운로드는 미검증 |
| L15 | [Open TG-GATEs 공식 설명](https://dbarchive.biosciencedbc.jp/en/open-tggates/desc.html), [CEL metadata](https://dbarchive.biosciencedbc.jp/en/open-tggates/data-4.html) | microarray/CEL과 metadata 구조 확인. RNA-seq count와 별도 관측모형 필요 |
| L16 | [Chem-PerturBridge 공식 처리 저장소](https://github.com/theislab/Chem-PerturBridge) | 통합 처리 후보. 원자료 중복·배포 버전·upstream license를 착수 전 확인 |

CREST로 PDF에 제시된 [OpenReview 주소](https://openreview.net/pdf/2728456f40639957871069ce84f232def80a108c.pdf), [cross-technology alignment 주소](https://openreview.net/pdf?id=Bj4biReH4G)는 이번 검토에서 원문을 독립 확인하지 못했다. 존재하지 않는다고 단정하지 않으며, 성능 수치와 즉시 실행 가능성을 확인한 것으로도 쓰지 않는다. PhenMol 및 추가 최신 생성 모델 역시 상세 원문·공식 자산 계약 검증 후 조건부 후보로만 등록한다.

## 부록 C. 최종 요약

현재 연구는 “전부 실패했으니 처음부터 새 대형 모델을 만들자”로 정리할 단계가 아니다. **R17에서 남은 신호를 찾았고, 이제 그 신호가 다른 측정에서도 유지되는지 확인할 단계**다. 동시에 첨부 PDF의 좋은 아이디어를 작은 실험으로 바꾸면 shift·context·구조·손실 중 무엇이 실제 병목인지 더 분명해질 수 있다.

이번 계획의 핵심은 세 가지다. **기존 후보 확인과 새 후보 개발을 분리하고, 한 번에 하나의 원인을 바꾸며, 데이터가 답할 수 있는 범위까지만 결론을 내린다.** 이 원칙이 지켜져야 다음 음성 결과도 막연한 실패가 아니라 이후의 불필요한 반복을 줄이는 지식이 된다.
