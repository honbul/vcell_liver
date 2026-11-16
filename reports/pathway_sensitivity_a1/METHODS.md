# 연구 A 방법

이 문서는 실제로 실행한 절차를 순서대로 적습니다. 모든 규칙은 `configs/pathway_sensitivity_a1/analysis.yaml`에 있고, 이 파일은 예측과 실측을 비교한 점수가 하나도 없던 시점(커밋 `606d065`)에 고정되었습니다. 이후 규칙을 바꾼 일은 없습니다(`amendments`가 비어 있음).

## 1. 원자료와 출처

| 자료 | 출처 | 받은 방법 |
|---|---|---|
| LIMORE 약물 반응(AA, IC50, Emax) | Qiu 외, Cancer Cell 2019 | 서버 공유 영역에 보관된 사본 `drug.txt` |
| LIMORE 모델 주석 | 같은 연구 | 서버 보관 사본 `81clinic_information.txt` |
| LIMORE RNA | GEO GSE97098 | `GSE97098_HCC81.genes.fpkm_table.log2.txt.gz`를 NCBI FTP에서 받음 |
| LCCL RNA, AUC, GI50, 주석, 약물표 | Caruso 외, Gastroenterology 2019 | `lccl.zucmanlab.com/downloads/catalogue/`에서 받음 |
| Hallmark 유전자 집합 | MSigDB 2026.1.Hs | 서버에 이미 있던 사본 |
| 세포주 진단·계열 관계 | Cellosaurus 57.0 | 공개 API |
| 환자 ID 대조 | DepMap 25Q3 `Model.csv` | 서버 공유 사본 |

파일별 URL·시각·크기·SHA-256은 `manifests/pathway_sensitivity_a1/downloads.tsv`에 있습니다. 접근에 실패한 경로는 `DATA_CARD.md`에 적었습니다.

## 2. 변환 순서

1. **모델 등록표.** 두 연구의 세포 이름을 Cellosaurus 등록번호에 대응시킨 표(`configs/.../model_aliases.tsv`)를 만들고, 진단과 계열을 붙였습니다. 원 연구와 Cellosaurus가 모두 간세포암으로 적은 모델만 포함했습니다. 계열은 같은 등록번호, 조상 세포주, 같은 개인 연결, DepMap 환자 ID로만 묶었습니다.
2. **유전자 축.** GEO 파일의 protein_coding 유전자 중 LCCL 표에도 같은 Ensembl ID가 있는 것만 남겼습니다. 같은 기호가 여러 ID에 붙은 경우 LIMORE 평균 발현이 가장 높은 ID를 남겼습니다. 19,018개입니다. 내부 분석과 외부 분석이 같은 축을 씁니다.
3. **발현 척도.** LIMORE는 등록된 log2 값을 그대로 썼습니다. LCCL은 FPKM에 `log2(FPKM+1)`을 한 번 적용했습니다.
4. **순위.** 표본 안에서 유전자 순위(동점은 최소 순위)를 구해 유전자 수로 나눴습니다.
5. **경로 점수.** Bioconductor singscore 1.30.0의 `rankGenes(tiesMethod="min")`와 `simpleScore(upSet=집합, centerScore=TRUE)`를 썼습니다. 50개 집합 모두를 같은 방식으로 점수화했고, 이름에 `DN`이 붙은 집합도 뒤집지 않았습니다.
6. **섞은 경로.** 19,018개 유전자 이름 전체를 무작위로 바꿔 끼운 뒤 같은 집합 구성표에 적용했습니다. 집합 크기, 집합 사이 겹침, 유전자가 등장하는 횟수가 유지됩니다. 고정된 seed 20개(101–120)를 모두 썼습니다. 유전자 축이 모든 fold에서 같으므로 섞기도 fold와 무관하게 한 번씩 했습니다.
7. **정답.** `drug.txt`의 `Activity Area`를 그대로 썼습니다. 변환·보간·평균은 하지 않았습니다.

## 3. 표현과 예측기

| ID | 입력 | 차원 | 절차 |
|---|---|---|---|
| A0 | 없음 | – | 학습 계열의 가중 평균 |
| A1 | log2 FPKM | 19,018 | 표준화 → Ridge |
| A1r | 상대 순위 | 19,018 | 표준화 → Ridge |
| A2 | 상대 순위 | fold별 d | 표준화 → PCA → 표준화 → Ridge |
| A3 | Hallmark 점수 | 50 | 표준화 → Ridge |
| A4 | 섞은 Hallmark 점수 | 50 | 표준화 → Ridge |

- 약물마다 별도의 회귀입니다. 약물 구조는 쓰지 않습니다.
- scikit-learn 1.9.1의 `Pipeline`, `StandardScaler`, `PCA(svd_solver="full")`, `Ridge`를 썼습니다.
- alpha 후보는 모든 표현에서 `[0.01, 0.1, 1, 10, 100, 1000, 10000]`입니다. 동점이면 큰 alpha를 고릅니다.
- 정답은 약물별로 그 학습 집합의 가중 평균·표준편차로 표준화한 뒤 학습하고 원 단위로 되돌립니다.
- 한 계열에 모델이 여럿이면 각 모델의 가중치를 `1/계열 내 모델 수`로 두어 계열의 총 가중치를 같게 했습니다. 표준화와 Ridge 모두에 같은 가중치를 썼습니다.
- A2의 차원 d는 `min(50, 가장 작은 내부 학습 표본 수 − 1, 행렬의 rank)`입니다. 기획서의 식을 "바깥 fold마다 하나의 d를 정해 alpha 선택과 최종 학습에 똑같이 쓴다"로 해석했습니다. 실제 값은 결과 문서에 적었습니다.
- 학습에서 분산이 0인 유전자는 표준화 뒤 값이 0이 되어 Ridge 계수가 0입니다(fold 안에서의 저분산 제거와 같은 효과). 그 밖의 유전자 선택은 하지 않았습니다.

## 4. 분할과 모델 선택

- 바깥: `family_id` 단위 5-fold를 seed를 바꿔 3회 반복했습니다(`GroupKFold(shuffle=True)`). 배정표는 `manifests/pathway_sensitivity_a1/splits_outer.tsv`이며 모든 표현·약물이 같은 표를 씁니다.
- 안쪽: 학습 계열을 다시 3-fold로 나눠 alpha를 골랐습니다. 손실은 계열 안에서 먼저 평균한 제곱오차(학습 표준편차 단위)입니다.
- 표준화, PCA, 정답 표준화는 매번 그 학습 부분에서만 fit했습니다. 예측 함수는 학습 행만 인자로 받으므로 시험 정답이 들어갈 길이 없습니다.
- 반복 3회는 같은 76개 계열을 다르게 나눈 것이며, 새로운 생물학적 표본이 아닙니다.

## 5. 평가

- **주지표 `macro_drug_normalized_rmse`.** 바깥 fold와 약물마다 `(실측 − 예측) / s_train`을 구합니다. `s_train`은 그 fold 학습 반응의 가중 표준편차입니다. 제곱오차를 계열 안에서 평균하고, 계열 사이를 평균해 제곱근을 취하면 약물별 값이 됩니다. 이것을 90개 약물에 대해 같은 가중치로 평균하고, 반복 3회의 값을 평균합니다. 낮을수록 좋습니다.
- 보조: 약물별 원 단위 RMSE·MAE, 계열 평균값으로 계산한 Spearman.
- 반복을 먼저 평균한 예측으로 계산한 점수는 `ensemble_of_repeats_macro_nrmse`라는 다른 이름으로만 적었고, 결론에는 쓰지 않았습니다.
- 모든 약물×모델 쌍을 한 벡터로 편 상관은 계산하지 않았습니다.
- **불확실성.** 76개 계열을 복원추출하는 bootstrap 2,000회(seed 9101)입니다. 뽑힌 계열의 모든 약물·반복·모델 예측이 함께 움직입니다. 95% 구간은 2.5–97.5 백분위입니다. 이 구간은 고정된 예측값에 대한 표본 불확실성이며, 학습을 다시 했을 때의 변동은 포함하지 않습니다.
- 학습곡선(50%)의 점수도 100% 학습의 `s_train`으로 나눠 같은 척도에서 비교했습니다.

## 6. A3 개선 판정 기준(사전 고정)

"A3의 주지표가 A0과 A1보다 낮고, (A3 − A0)과 (A3 − A1)의 95% bootstrap 구간이 모두 0보다 아래"일 때만 실질적 개선으로 봅니다. 이 기준을 충족하지 못하면 A4와 학습곡선은 한 번만 실행하고 "소규모 진단"으로 표시합니다.

## 7. 외부 평가(LCCL)

- 대응 약물 18개는 이름과 LCCL 별칭 열로 정했습니다(`configs/.../drug_match_lccl.tsv`). 조합 4개와 대응이 없는 13개는 제외했습니다. 비슷하지만 다른 화합물(예: 17-DMAG와 17-AAG)은 합치지 않았습니다.
- **새 계열:** LIMORE에 없는 LCCL 계열은 포함 대상 LIMORE 모델 전체로 학습한 모델로 예측했습니다.
- **같은 계열의 다른 연구 측정(cross-study leave-family-out):** LIMORE에도 있는 계열은 그 계열을 LIMORE 학습과 alpha 선택에서 뺀 모델로 예측했습니다. 계열마다 모델을 따로 학습했습니다(26개).
- 입력은 LCCL의 RNA이고, 표준화·PCA는 LIMORE 학습 자료에서만 fit했습니다. 두 연구를 합친 정규화는 없습니다.
- 채점은 약물별 Spearman(예측 AA, −AUC)입니다. 계열 안에서 예측과 실측을 먼저 평균했습니다. AA와 AUC 값을 같은 척도로 놓거나 섞지 않았습니다. GI50은 보조로만 계산했습니다.
- 참고값으로, 모델 없이 같은 계열의 LIMORE 실측 AA와 LCCL 실측 −AUC의 Spearman도 계산했습니다. 이는 두 연구의 측정이 서로 얼마나 일치하는지를 보여 줍니다.
- 구간은 계열 bootstrap 2,000회(seed 9202)입니다.

## 8. 실행 명령

서버 작업 폴더에서 실행합니다. `RUN_ID`는 한 번 쓰면 다시 쓸 수 없습니다(같은 파일이 있으면 실행이 멈춥니다).

```bash
# 환경 (한 번): micromamba로 assets/envs/psa1 생성. 패키지 목록은 manifests/pathway_sensitivity_a1/environment.txt
sbatch experiments/pathway_sensitivity_a1/a0_build.sbatch                                   # 자료 연결 + 점검
sbatch --export=ALL,RUN_ID=a2_main_d284715 experiments/pathway_sensitivity_a1/a2_internal.sbatch
sbatch --export=ALL,STAGE=main,MAIN_RUN=a2_main_d284715 experiments/pathway_sensitivity_a1/evaluate.sbatch
sbatch --export=ALL,RUN_ID=a3_controls_45aa42a experiments/pathway_sensitivity_a1/a3_controls.sbatch
sbatch --export=ALL,STAGE=controls,MAIN_RUN=a2_main_d284715,CONTROLS_RUN=a3_controls_45aa42a experiments/pathway_sensitivity_a1/evaluate.sbatch
sbatch --export=ALL,STAGE=external,MAIN_RUN=a2_main_d284715,EXT_RUN=a4_external_45aa42a experiments/pathway_sensitivity_a1/evaluate.sbatch
python src/pathway_sensitivity_a1/make_figures.py --tables reports/pathway_sensitivity_a1/tables --out reports/pathway_sensitivity_a1/figures --oof-dir assets/runs/a2_main_d284715/oof
```

모든 작업은 CPU 1–2개, BLAS 스레드 1개로 실행했습니다. 전체 계산 시간은 자료 연결 약 6분(대부분 R 점수 계산), 내부 평가는 표현당 20초 이하입니다.

## 9. 실행한 점검(기획서 12절)

`src/pathway_sensitivity_a1/tests/test_checks.py`에 있으며 자료 연결 작업의 마지막에 실행됩니다. 결과는 `RESULTS.md` 7절에 적었습니다.
