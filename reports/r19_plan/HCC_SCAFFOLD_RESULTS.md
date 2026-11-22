# SNU-423 형제 복사의 이득과 구조 지원

**가장 가까운 학습 약물과 Tanimoto가 0.7 이상인 6군이, B0 대비 양의 제곱오차 감소의 90.8%를 차지했다.** 203군 전체의 R² 우위는 scaffold 연결 성분을 묶어 재표집했을 때 99% 구간에 0이 포함됐다. P4를 본 뒤 등록한 Tahoe 5 µM·24 h 동결 예측의 사후 검증이다. [사전 명세](HCC_SCAFFOLD_PROTOCOL.md) · [원 P4 결과](../p4/RESULTS.md)

## 어떤 구조에서 이득이 나왔나

원래 학습 fold에서 같은 SNU-423 반응을 가진 donor와 비교했다. 여러 entity가 있는 parent는 모든 entity에 가까운 donor가 있는지를 평가했다. 표의 R² 변화는 P4 형제 복사와 B0 사이의 차이로, 원 전체 pool의 중심을 유지했다. 양의 SSE 비중의 합은 네 기본 층에서 100%다.

| 최고 fit Tanimoto | 화합물군 | 물리 웰 | Scaffold 성분 | B0 대비 ΔR² [99% 참고 구간] | 양의 SSE 이득 비중 |
|---|---:|---:|---:|---:|---:|
| <0.3 | 155 | 211 | 146 | +0.35%p [−0.68, +1.33] | 5.8% |
| 0.3 이상, 0.5 미만 | 29 | 39 | 28 | +1.36%p [+0.42, +3.27] | 1.3% |
| 0.5 이상, 0.7 미만 | 13 | 15 | 7 | −10.24%p [−39.22, +4.61] | 2.1% |
| ≥0.7 | 6 | 10 | 4 | +69.34%p [+28.05, +73.08] | 90.8% |

Tanimoto 0.7 미만의 197군은 B0 대비 ΔR² −0.06%p [−1.20, +1.05], ΔRMSE −0.00002 [−0.00053, +0.00063]였다. 같은 scaffold 연결 성분의 fit donor가 없는 178군에서도 B0 대비 ΔR²는 −0.09%p [−1.33, +0.95]였다. 유사도 0.7 이상 중 이득이 가장 큰 두 parent는 idarubicin과 doxorubicin/epirubicin이다. dexamethasone과 triamcinolone도 개선 방향이었고, erythromycin과 azithromycin에서는 오차가 늘었다. 가장 가까운 6군은 기술 층의 최소 20군에 미달하며 4개 성분에 걸쳐 있었다. 두 DMSO 역할에서 모두 원 설정의 문턱을 넘은 범위는 전체의 16군이었다. [구조 지원 원표](data/hcc_scaffold/structure/parent_support.tsv) · [최근접 donor](data/hcc_scaffold/structure/entities.tsv) · [화합물별 기여](data/hcc_scaffold/score/by_parent.tsv)

## 구조 계열 단위로 짝을 묶으면 구간이 달라진다

동일한 203군과 275웰에서 기존 VC 최선을 지표·재표집마다 다시 선택했다. 원 parent 단위 추정값과 99% 구간을 `1e−10` 오차 이내로 재현했다. 아래 scaffold 재표집은 동일 pool을 매 draw 재중심화하되, R12 분할에 등록된 181개 scaffold 연결 성분 안의 parent에 공동 가중치를 줬다.

| 지표, 형제 복사 − 최선 기존 VC | 점추정 | 원 parent 단위 99% | Scaffold 성분 단위 99% |
|---|---:|---:|---:|
| ΔRMSE | −0.001446 | [−0.004381, +0.000521] | [−0.005341, +0.000723] |
| Δpooled R² | +8.11%p | [+0.34, +14.25]%p | [−0.64, +15.37]%p |

Scaffold 성분으로 묶으면 B0 대비 전체 R² 구간도 [−0.48, +32.37]%p로 0을 포함했다. 가까운 학습 유사체가 있는 몇몇 화학 계열에서 형제 복사의 적용 가능성이 관측됐다. 같은 계열의 화합물이 가중치를 공유하면 증거가 약해졌고, 0.7 미만의 넓은 지원에서 무반응 기준선보다 나은 근거는 확보되지 않았다. 이 민감도는 기존 parent 분할의 동결 예측에서 재표집 단위만 바꿨다. 새 scaffold 전체를 제외하고 재적합한 검증은 별도 질문이다. [재표집 단위별 원표](data/hcc_scaffold/score/paired.tsv) · [구조 층별 원표](data/hcc_scaffold/score/strata.tsv)

명세 커밋 `73770ed`와 연결 성분 명시 `4f141a6`, 실행 코드 `7b20b3d`. G1 작업 `31219`가 발현 접근 없이 구조 층을 고정했고, `31220`이 원 P4 동결 예측과 대조해 별도 채점했다. CPU 할당 388 CPU초, GPU 0개다. 두 작업은 종료 코드 0으로 완료됐고 예약·격리·봉인 자료에 접근하지 않았다. [구조 동결](data/hcc_scaffold/structure/receipt.json) · [채점 영수증](data/hcc_scaffold/score/receipt.json) · [실행 원장](data/hcc_scaffold/jobs.tsv)
