"""Render the Korean reports from the verified worker tables and package receipt."""

import argparse
import csv
import json
from pathlib import Path

NAMES = {
    "drug_mean": "약물 평균", "additive": "약물 + 세포 가산", "drug_ridge": "약물별 Ridge",
    "cellhit": "CellHit 재적합", "structured": "정정 구조 커널",
    "ridge_structured_50_50": "Ridge + 구조 커널 50:50", "pairwise_rank": "V0+ v2 Pairwise Rank",
    "pairwise_top5": "V0+ v2 Pairwise Top-5", "ecfp4_path_elasticnet": "ECFP4 + 경로 ElasticNet",
    "a_2d_structured": "a · 구조 커널", "a_2d_pairwise_rank": "a · Pairwise Rank",
    "b_local_mean": "b · 현지 평균", "c_eb_shrinkage": "c · EB 수축",
    "d_eb_structured": "d · EB + 구조 커널 RNA", "d_eb_pairwise_rank": "d · EB + Pairwise Rank RNA",
}
PDO_NAMES = list(NAMES)[9:]
PDO_RNA = PDO_NAMES[-2:]
AUDITED = ("additive", "drug_ridge", "cellhit", "structured", "pairwise_rank", "ecfp4_path_elasticnet")
AXES = ("within_drug_spearman", "within_drug_pearson", "ndcg_at_5", "precision_at_5", "within_cell_spearman")
METRICS = ("약물 내 ρ", "약물 내 r", "NDCG@5", "Precision@5", "세포 내 ρ")
RESIDUAL = ("residual_spearman", "residual_pearson", "residual_rmse")


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def number(x, digits=4):
    return f"{float(x):.{digits}f}" if x is not None and x not in ("", "NA", "nan") else "—"


def signed(x):
    return f"{float(x):+.4f}" if x is not None and x not in ("", "NA", "nan") else "—"


def interval(row, estimate="delta"):
    return f"{signed(row[estimate])} [{signed(row['ci95_low'])}, {signed(row['ci95_high'])}]"


def read(root, filename):
    with (root / filename).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def js(root, filename):
    return json.loads((root / filename).read_text())


def write(root, file, txt):
    (root / file).write_text(txt.strip() + "\n")


def render(root):
    ledger = read(root, "UNIFIED_2D_LEDGER.tsv")
    own = {r["model_id"]: r for r in ledger}
    receipt = js(root, "score_receipt.json")
    axes_tbl = table(["모델", *METRICS], [
        [NAMES[r["model_id"]], *[number(r[k]) + ("†" if r["model_id"] == "drug_mean" and k in AXES[:2] else "")
                                 for k in AXES]] for r in ledger])
    oof = {r["model_id"]: r for r in read(root, "SOURCE_OOF_MACRO.tsv")}
    oof_tbl = table(["모델", *METRICS], [[NAMES[name], *[number(oof[name][k]) for k in (
        *AXES[:3], "precision_at_k", AXES[-1])]] for name in own])
    tk = read(root, "LIMORE_TKI.tsv")
    tki_tbl = table(["모델", "약물", "계통", "Spearman ρ", "Pearson r"],
                    [[NAMES[r["model_id"]], r["drug"], r["n_lineages"], number(r["spearman"]),
                      number(r["pearson"])] for r in tk])
    residuals = {(r["model_id"], r["metric"]): r for r in read(root, "LIMORE_RESIDUALS.tsv")}
    diversity = {r["model_id"]: r for r in read(root, "TOP5_DIVERSITY.tsv")}
    r_table = table(["모델", "잔차 Spearman [95%]", "잔차 Pearson [95%]", "잔차 RMSE [95%]", "고유 Top-5 / 50",
                     "평균 Jaccard", "약물 평균과 겹침"],
                    [[NAMES[m], *[interval(residuals[m, key], "estimate") for key in RESIDUAL],
                      diversity[m]["unique_top5_sets"], number(diversity[m]["mean_pairwise_top5_jaccard"]),
                      number(diversity[m]["mean_overlap_with_drug_mean_top5"])] for m in own])
    compact_r = table(["모델", "잔차 ρ [95%]", "고유 Top-5 / 50", "쌍별 Jaccard"],
                      [[NAMES[name], interval(residuals[name, "residual_spearman"], "estimate"),
                        diversity[name]["unique_top5_sets"], number(diversity[name]["mean_pairwise_top5_jaccard"])]
                       for name in ("drug_mean", *AUDITED)])
    interventions = {(r["model_id"], r["metric"]): r for r in read(root, "RNA_INTERVENTIONS.tsv")}
    audit_tbl = table(["모델", "지표", "정상 RNA", "학습 평균 RNA", "셔플 평균 [2.5%, 97.5%]",
                       "정상−셔플", "순열 p"],
                      [[NAMES[m], label, number(r["native"]), number(r["train_mean_rna"]),
                        f"{number(r['shuffle_average'])} [{number(r['shuffle_p025'])}, {number(r['shuffle_p975'])}]",
                        signed(r["native_minus_shuffle_mean"]), number(r["permutation_p_one_sided"], 3)]
                       for m in AUDITED for metric, label in zip((*AXES[:3], *RESIDUAL),
                                                               (*METRICS[:3], "잔차 ρ", "잔차 r", "잔차 RMSE"))
                       for r in [interventions[m, metric]]])
    decision = js(root, "reference_decision.json")
    selected = decision["designated_model"]
    pass_2d = decision["joint_promotion"]
    fit_scope = {
        "joint research reference": "두 평가 축과 상호작용 기준을 모두 통과한 연구용 2D 기준",
        "limited within-drug research reference": "약물 내 감수성 비교용 연구 기준",
        "nonpersonalized ranking reference": "약물 기본 순위 기준",
    }[decision["scope"]]
    reason = ("LIMORE 두 축에서 단순 대조군 대비 paired 하한이 모두 양수였고 내부 OOF 및 상호작용 기준을 통과했다."
              if pass_2d else "고정한 동시 승격 기준을 통과한 모델은 없었다. 세포 내 약물 순위와 약물 내 감수성 수치는 모델별로 해석한다.")
    audit_receipt = js(root, "personalization_receipt.json")
    max_swap = max(max(d.values()) for d in audit_receipt["replayed_perturbation_max_abs_diff"].values())
    proc = js(root, "package_replay_receipt.json")
    witness = js(root, "fewshot_collection_receipt.json")
    pass3d = js(root, "fewshot_decision.json")
    support = witness["source_support"]
    metrics3 = read(root, "FEWSHOT_3D_METRICS.tsv")
    paired3 = read(root, "FEWSHOT_3D_PAIRED_95CI.tsv")
    calibration = {r["track"]: r for r in read(root, "FEWSHOT_3D_CALIBRATION.tsv")}
    transfer = read(root, "TRANSFER_MATCHED_PATIENT_PANELS.tsv")
    tri = {(r["track"], r["model_id"]): r for r in metrics3}
    ci3 = {(r["track"], r["model_id"], r["baseline"], r["metric"]): r for r in paired3}
    match_tracks = [f"{study}/exact68/K{k}" for study in ("LICOB", "Wong", "Yang") for k in (2, 4, 8)]
    full_tracks = [f"Yang/full_local/K{k}" for k in (2, 4, 8)]
    panel_order = [*match_tracks, *full_tracks]
    pdo_table = table(["코호트·패널·K", "모델", "채점 환자/run", "약물", "NDCG@5", "P@k", "P@1", "환자 내 ρ",
                       "약물 내 ρ", "약물 내 r", "잔차 ρ", "잔차 r", "잔차 RMSE"],
                      [[p.replace("/exact68/", " / ").replace("/full_local/", " 전체5 / "), NAMES[name],
                        r["held_patients_each"], r["drugs"], number(r["ndcg_at_5"]),
                        number(r["precision_at_k"]) + f" (@{r['precision_k']})", number(r["precision_at_1"]),
                        *[number(r[key]) for key in ("within_cell_spearman", *AXES[:2], *RESIDUAL)]]
                       for p in panel_order for name in PDO_NAMES if (p, name) in tri for r in [tri[p, name]]])
    compact3 = table(["코호트 / K", "b NDCG@5", "c NDCG@5", "d 구조 Δ NDCG [95%]", "d Pairwise Δ NDCG [95%]",
                      "d 구조 Δ 잔차 ρ [95%]", "d Pairwise Δ 잔차 ρ [95%]"],
                     [[p.replace("/exact68/", " / "), number(tri[p, "b_local_mean"]["ndcg_at_5"]),
                       number(tri[p, "c_eb_shrinkage"]["ndcg_at_5"]),
                       *[interval(ci3[p, name, "c_eb_shrinkage", key]) for key in ("ndcg_at_5", "residual_spearman")
                         for name in PDO_RNA]] for p in match_tracks])
    support_tbl = table(["코호트", "환자", "PDO", "PDO×약물", "전체 약물", "2D 정확 매칭", "최저 RNA 범위"],
                        [[name, r["patients"], r["pdos"], r["pdo_drug_rows"], r["drugs"], r["registered_68_exact_drugs"],
                          f"{r['rna_genes_min_fraction']:.3%}"] for name, r in support.items()])
    balances = table(["코호트", "K", "채점 환자/run", "환자별 보정 횟수 범위", "고유 보정 패널/100",
                      "γ=0 구조", "γ=0 Pairwise"],
                     [[s, k, props["test_patients_per_repeat"],
                       f"{props['calibration_min_per_patient']}–{props['calibration_max_per_patient']}",
                       props["unique_calibration_sets"],
                       calibration[f"{s}/exact68/K{k}"]["gamma_d_eb_structured_zero_count"],
                       calibration[f"{s}/exact68/K{k}"]["gamma_d_eb_pairwise_rank_zero_count"]]
                      for s, detail in witness["patient_balance"].items() for k, props in detail.items()])
    k4_rows = [tri[p, key] for p in match_tracks if p.endswith("/K4") for key in ("c_eb_shrinkage", *PDO_RNA)]
    k4 = table(["코호트", "K=4 모델", "NDCG@5", "약물 내 ρ", "잔차 ρ"],
               [[r["study"], NAMES[r["model_id"]], *[number(r[k]) for k in (
                   "ndcg_at_5", "within_drug_spearman", "residual_spearman")]] for r in k4_rows])
    n_pass = pass3d["matched_panels_with_gate"]
    conclusion3 = (
        f"공통 비교 18개 중 {n_pass}개 모델·코호트·K 조건이 두 기준선 대비 동시 기준을 통과했다. "
        "해석 범위는 통과한 공개 패널과 보정 예산에 한정된다." if n_pass else
        "현지 소수 라벨 보정의 주된 효용은 약물 기본 순위의 현지화였다. RNA 개인화의 추가 가치는 현재 공개 3D 코호트에서 미확증으로 판정했다. "
        "운영 연구의 단순 기준으로 현지 K명 평균을 유지한다.")
    a_mean_delta = table(["코호트 / K", "현지 평균 NDCG−무보정 구조", "현지 평균 NDCG−무보정 Pairwise"],
                         [[p.replace("/exact68/", " / "),
                           *[signed(float(tri[p, "b_local_mean"]["ndcg_at_5"]) - float(tri[p, key]["ndcg_at_5"]))
                             for key in PDO_NAMES[:2]]] for p in match_tracks])
    cross = table(["적합 → 평가", "적합/평가 환자", "모델", "공통 3종 NDCG@3", "Sorafenib 단독 ρ", "Sorafenib 단독 r"],
                  [[f"{r['fit_study']} → {r['test_study']}", f"{r['fit_patients']} / {r['test_patients']}",
                    NAMES[r["model_id"]], number(r["ndcg_at_3"]),
                    *[number(s[key]) for key in AXES[:2]]]
                   for r in transfer if r["panel"] == "common_3_exact_TKIs" and r["model_id"] in ("c_eb_shrinkage", *PDO_RNA)
                   for s in transfer if s["panel"] == "sorafenib_only_direct_single_drug"
                   and all(r[key] == s[key] for key in ("fit_study", "test_study", "model_id"))])
    cold = proc["cpu_latency_seconds"]["cold_cli_single_68_drugs_median"]
    warm = proc["cpu_latency_seconds"]["warm_kernel_single_68_drugs_median"]
    acc = proc["single_batch_cli_max_abs_diff"]
    max_frozen = proc["frozen_cpu_replay_max_abs_diff"]
    promotion = "통과" if pass_2d else "미통과"
    status = table(["작업 완료", "가설 통과", "도구 완성", "현장 효용"],
                   [["Task 1–4 코드·동결·별도 채점·리포트·브랜치 푸시", f"2D 동시 승격 {promotion}, 3D {n_pass}/18",
                     f"정정 구조 커널 CPU CLI 3,400행 재생 및 입력 거절", "신규 실험 미실시. 공개 자료 후향 검증"]])
    footer = ("LIMORE, LICOB, Wong, Yang은 앞선 프로젝트에서 평가한 이력이 있다. 모든 수치는 고정된 비교 계약 하의 후향 통합 확증이다. "
              "† 약물 평균은 세포 사이에서 상수라 약물 내 ρ와 r이 수학적으로 미정의다. "
              "대조 계산에는 0을 넣었고 TKI 원표의 해당 필드는 NA다.")
    write(root, "RESULTS.md", f"""
# 간암 2D 효능과 3D 현지 보정 확증

{status}

**단일 2D 기준은 {NAMES[selected]}**이다. 사용 범위는 {fit_scope}이다. {reason} {conclusion3}

## Task 1 · 정정 65,678행 단일 원장

GDSC2·CTRPv2·PRISM20Q2의 1,014모델·991계통·68약물을 적합하고 독립 LIMORE 성인 HCC 50계통 × 68약물의 동일 Activity Area 3,400행을 한 번 채점했다. 명세가 열거한 9개 비교군을 각각 적합했다. 순위 이득 NDCG@5와 동률 기대 Precision@5를 모든 모델에 적용했다.

{axes_tbl}

{footer}

학습표 SHA-256은 `{receipt['training_sha256']}`이다. 진단 코드는 종전 LIMORE 평가식과 최대 절대 오차 {receipt['legacy_2d_metric_max_abs_difference']:.2g}였다. 2,400회 계통·약물 교차 재표집을 동일 예측에 적용했다. [단일 원장](UNIFIED_2D_LEDGER.tsv) · [paired 95% 구간](LIMORE_PAIRED_95CI.tsv) · [채점 영수증](score_receipt.json)

### 4개 HCC TKI 성적

{tki_tbl}

[TKI 원표](LIMORE_TKI.tsv)

## Task 2 · 2D 상호작용과 추천 다양성

{compact_r}

가산 모델은 계통별 높낮이를 바꿀 수 있지만 이중 중심화하면 상호작용이 0이다. 100회 독립 계통 derangement에서 RNA와 RNA 유도 벡터를 함께 옮겼다. [전체 개인화 감사](PERSONALIZATION_AUDIT.md)

## Task 3 · 3D 현지 라벨 보정

{k4}

[세 코호트의 K=2,4,8 전체 결과 및 환자 단위 전이표](FEWSHOT_3D_CONFIRM.md)

## Task 4 · 입력 검증과 논문 골격

정정 구조 커널 CPU 패키지가 3,400개 예측을 재생했다. 동결 대비 최대오차 {max_frozen:.3g}, 단일 파일과 배치의 최대오차 {acc:.3g}, 단일 RNA·68약물 CPU CLI 중앙값 {cold:.3f}초였다. 음수 입력, 전체 및 선택 유전자 미달, 모든 값 0, 미등록 exact parent를 검사했다. [패키지 영수증](package_replay_receipt.json) · [모델 카드](model_card.md) · [Paper 2 초안](PAPER2_OUTLINE_AND_TABLES.md)
""".replace("T\u200b\u200bKI", "TKI"))
    write(root, "PERSONALIZATION_AUDIT.md", f"""
# LIMORE 50계통에서의 상호작용과 개인화

측정과 예측의 각 행렬에서 약물 평균과 세포 평균을 빼고 전체 평균을 더했다. 잔차에 대한 Pearson, Spearman, RMSE 구간은 계통·약물 교차 재표집 2,400회에서 매번 다시 중심화해 계산했다. RMSE는 각 모델의 원래 점수 단위 영향을 받는다. 순수 약물 평균과 가산 모델의 상호작용 상관은 상수 예측에 대한 관례적 0이다. 추천 집합 동률은 exact parent 키 순으로 결정했고, 채점용 Precision@5는 동률 기대값을 사용했다.

{r_table}

## RNA 대조

학습 1,014개 RNA의 유전자별 평균을 모든 평가 계통에 대입했다. RNA 유래 scGPT·Geneformer 벡터도 각각 학습 평균으로 치환했다. 셔플 100회에서는 고정점 없는 배정으로 RNA 및 동반 벡터를 같은 계통에서 가져왔다. 완전한 50×68 행렬의 프로필별 예측을 재배열한 뒤 3회 실제 재추론으로 확인했다. 최대 오차는 {max_swap:.3g}였다.

{audit_tbl}

[100회 원표](RNA_SHUFFLE_100.tsv) · [전체 지표](RNA_INTERVENTIONS.tsv) · [다양성](TOP5_DIVERSITY.tsv) · [상호작용 구간](LIMORE_RESIDUALS.tsv)

## 정정 학습표의 3-fold OOF

역사적 설정이 적합된 3개 lineage fold의 계통 배정을 유지했다. 멀티어세이 성적은 원래 endpoint 방향 결정을 사용해 소스와 폴드를 동등 평균했다. 외부 판정과 하이퍼파라미터 적합에서 LIMORE 라벨의 역할을 분리했다.

{oof_tbl}

지정한 단일 연구 기준은 **{NAMES[selected]}**이며 사용 범위는 {fit_scope}이다. {reason} 후보의 두 축이 엇갈리는 경우 위의 두 개별 지표를 함께 인용한다. [결정 영수증](reference_decision.json) · [OOF 소스별 원표](SOURCE_OOF_BY_FOLD.tsv) · [OOF 분할 지원](SOURCE_OOF_SUPPORT.tsv)
""")
    write(root, "FEWSHOT_3D_CONFIRM.md", f"""
# 3D 환자 단위 소수 라벨 현지 보정

세 공개 코호트 모두 앞서 평가 이력이 있어 고정 계약 하에 후향 분석했다. LICOB는 검증된 iLICOB 기저 RNA와 S6 정규화 AUC, Wong은 S7 AUC, Yang은 S6 AUC와 기저 RNA를 사용했다. 감수성이 높을수록 큰 방향으로 통일했고 각 환자의 지역별 PDO를 먼저 평균했다. Yang은 P1과 P2 제외 후 74명이다. [LICOB 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10949980/) · [Wong 원논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC12205652/) · [Yang 원논문](https://pubmed.ncbi.nlm.nih.gov/38593780/)

{support_tbl}

K=2,4,8에서 100회 보정 환자를 균형 선택한 뒤 그 외 환자를 모두 채점했다. 각 반복에서 보정 K명의 라벨 파일과 평가 파일을 분리했고 환자 교집합은 0이다. 0명 무보정(a), K명 현지 평균(b), K명 경험적 베이즈 수축(c), 동일한 K명으로 양의 잔차 배율만 적합(d)을 비교했다. d의 이중 중심화에 쓰는 약물 평균은 보정 환자에서 계산했고 환자 평균은 해당 환자 자신의 예측값에서 계산했다. Pairwise Rank는 전사체 임베딩이 없는 입력을 처리하는 기존 오프라인 경로로 실행했다. 분석의 입력은 연구용 벤치마크 프로필이다. 특히 LICOB의 서명 있는 발현값은 비음수 입력을 요구하는 출시 CLI의 입력 형식과 다르다.

Yang의 Ivosidenib 및 Apatinib은 정정 68개 출처 패널 밖에 있다. 세 정확 매칭 TKI에서 a–d를 모두 비교했다. Yang 전체 5개 패널은 b–d의 보조 분석이며 두 미지원 약물에서는 c와 d가 b와 동일한 현지 평균을 반환했다.

{balances}

## 모형별 성적

{pdo_table}

Yang 정확 매칭 3약물에서 Precision@3은 회수한 약물 집합 전체이므로 1이다. Top-1도 함께 제시했다. Wong K=8의 평가 표본은 회차당 두 환자다. 약물 내 ρ와 r은 표본 두 점일 때 ±1 또는 동률이어서 해당 수치는 기술적으로 해석한다. 상수 예측의 약물 내 상관은 관례적 0이다.

## 같은 라벨 예산에서의 차이

{compact3}

{conclusion3}

CI는 500회 환자·약물 교차 재표집에서 모든 반복의 환자 가중치를 공유한 paired 구간이다. 보정 모델별 100개 적합을 고정했으므로 이 구간은 그 보정 설계에서의 외부 평가 변동을 나타낸다. [모형별 원표](FEWSHOT_3D_METRICS.tsv) · [d−c 및 d−b 95% 구간](FEWSHOT_3D_PAIRED_95CI.tsv) · [γ 및 수축 강도](FEWSHOT_3D_CALIBRATION.tsv) · [데이터와 무누수 분할 영수증](fewshot_collection_receipt.json)

{a_mean_delta}

## 연구 간 공통 약물과 환자 단위 전이

Sorafenib 단독 분석은 출처 코호트에서 그 약물로만 보정하고, 대상 연구에서 환자 간 상관을 평가한다. 공통 세 약물 분석은 Sorafenib, Lenvatinib, Regorafenib의 같은 exact parent와 같은 환자 평균을 사용했다. 두 패널에서 서로 다른 지표를 보고한다.

{cross}

[공통 세 약물과 단독 Sorafenib 원표](TRANSFER_MATCHED_PATIENT_PANELS.tsv)
""")
    write(root, "model_card.md", f"""
# 정정 구조 커널 연구용 모델 카드

`{proc['data_revision']}`은 고정한 정정 구조 커널의 CPU 패키지다. 연구용 2D 세포주 후보 선별 도구이며, 환자 임상 치료·용량·3D 무보정 효능을 보장하지 않는다. 결과의 사용 범위는 투약 전 bulk RNA와 68개 정확 parent의 연구용 스크린 감수성 점수다. 비교 평가에서 지정한 단일 기준은 {NAMES[selected]}이며 사용 범위는 {fit_scope}이다.

- 식별자와 전처리. 정확한 부모 InChIKey 또는 해당하는 표준화 SMILES를 받는다. 유전자별 TPM, log₂(TPM+1), 또는 카운트를 명시한다. `bulk_counts`에서는 유전자 길이 TSV를 주면 길이 보정 TPM을 쓰고 길이 파일을 생략하면 CPM으로 환산한다. 결측 입력은 학습 중앙값으로 채운다. 전체 {proc['gene_count']:,}개 및 선택 {proc['selected_gene_count']:,}개 유전자의 관측 범위가 각각 85% 이상이고 양수 발현이 있어야 한다.
- 거절 상태. 음수·무한대는 처리 전에 입력 오류를 반환한다. 전체 및 선택 유전자 부족 또는 모든 발현 0은 `insufficient_rna_coverage`, 미등록 parent는 `compound_outside_training_panel`과 빈 점수를 반환한다.
- 모형. 기저 RNA 순위, 공개 경로·전사인자, 적합 집단에서 추정한 공개 보조 특성, exact parent 화학구조·표적을 정정 소스에서만 적합했다. 출력은 학습 스크린에 정렬한 높은 값 방향의 점수, 약물 평균과 예측 잔차다. 이 잔차는 개별 예측에서 약물 anchor를 뺀 값이며 검증 보고서의 이중 중심화 잔차 정의는 별도로 적용한다.
- 2D 검증. LIMORE 50계통 × 68약물 동결 예측을 단일 모델로 재생했다. 동결 대비 최대오차 {max_frozen:.3g}, 단일 파일과 배치 간 최대오차 {acc:.3g}, 전체 50개 직접 커널 비교 최대오차 {proc['single_batch_kernel_all_profiles_max_abs_diff']:.3g}였다. CPU `{proc['cpu']}` {proc['cpu_threads']}스레드에서 CLI 중앙값 {cold:.3f}초, warm kernel 중앙값 {warm:.4f}초였다.
- 무결성. 학습 입력 65,678행의 SHA-256 `{proc['train_sha256']}`, 배포 체크포인트 `{proc['checkpoint_sha256']}`. [재생 영수증](package_replay_receipt.json) · [배포 매니페스트](package_release.json)

## G1에서 오프라인 실행

RNA 파일은 `sample_id` 열과 유전자별 열을 갖는 TSV거나 `sample_id`, `gene`, `value` 열의 long TSV다. 약물 파일은 `compound_id` 또는 `smiles` 열의 TSV다. `--compounds`를 생략하면 68개를 전부 예측한다.

```bash
study=/mnt/S1/sdata/workbench/Virtual_Cell/gssong
worker="$study/confirm_20261123/worker1_eff"
export PYTHONPATH="$worker/code/src:$study/vc_build_v1/v0plus-model-deps:$study/vc_build_v1/function/prepare-deps:$study/vcell_liver/vcell_r12_runtime_cu126"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
"$study/vcell_liver_r11_chemical_probe/assets/study-env/bin/python" -m vc_build.v0plus_cli predict \
  --package "$worker/task4/package" --rna rna.tsv --scale bulk_log2_tpm_plus_1 \
  --compounds compounds.tsv --out predicted.tsv
```
""")
    write(root, "PAPER2_OUTLINE_AND_TABLES.md", f"""
# Paper 2 · 간암 약물 반응에서 기본 순위, 세포 감수성 및 상호작용의 분리

## Abstract 골격

간암 약물 반응 모델의 높은 상관과 순위 회수율은 약물 기본 순위, 전반적인 세포 감수성, 실제 세포×약물 상호작용이 함께 반영된 결과다. 정정 65,678개 2D 관측으로 아홉 모델을 적합하고 LIMORE 3,400개의 동결 예측을 같은 지표로 평가한다. 예측의 이중 중심화, RNA 치환 및 100회 셔플, 추천 집합의 지문을 사용한다. 세 독립 3D 공개 코호트에서 K=2,4,8명 보정과 연구 간 전이를 환자 단위로 대조한다. 2D 동시 승격은 **{promotion}**이고 3D 동시 기준 통과는 모델·코호트·K의 {n_pass}/18개였다. 본문은 어떤 모델이 어떤 사용 축을 개선하는지와 소수 보정이 바꾸는 항을 구분해 해석한다. [전체 결과](RESULTS.md)

## Research questions

1. 동일한 소스 표, 학습/평가 분리, 순위 정의에서 모델 간 차이는 어느 축에서 재현되는가?
2. 전반적인 세포 감수성을 제거하고 RNA를 교환하면 상호작용과 추천 다양성이 얼마나 남는가?
3. 같은 3D 환자 지원과 동일 약물에서 연구 간 전이가 얼마나 유지되는가?
4. 같은 K명의 라벨을 주면 비음수 RNA 잔차 배율로 얻는 이득이 현지 약물 평균과 수축을 모두 넘는가?

## Figure 설계

1. **공통 분석 계약**. CTRPv2에서 복합물 네이티브 ID를 걷어낸 정정 접점, 3-fold의 계통 경계, LIMORE 동결 및 독립 채점.
2. **상호작용 분해**. 원시 측정/예측과 이중 중심화 잔차, 100회 셔플 분포, 공통 50계통 Top-5 집합 요약.
3. **전이의 환자 단위 경계**. LICOB·Wong·Yang 6개 방향에서 공통 3종 약물 NDCG@3. Sorafenib 단독은 별도 상관 패널로 표시.
4. **라벨 예산별 현지 보정**. K=2,4,8에서 c와 d의 같은 분할 paired NDCG 및 상호작용 차이와 95% 구간.

## Table 1 · 정정 2D 단일 모델 비교

{axes_tbl}

{footer}

## Table 2 · 약물과 세포 평균 제거 후 상호작용

{compact_r}

## Table 3 · 현지 RNA의 3D 추가 가치

{compact3}

{conclusion3}

## 선행 연구와 차별점

- CLAMP는 분자와 어세이 언어의 대조 사전학습을 사용해 새로운 약물 탐색과 few-shot 분자 활성 모델을 다뤘다. 이 연구는 고정 RNA·약물 예측에서 세포×약물 상호작용과 동일 예산 현지 평균 대비 증가분을 평가한다. [Seidl et al., ICML 2023](https://proceedings.mlr.press/v202/seidl23a.html)
- CellHit은 세포주 약물 스크린과 환자 발현 도메인 정렬을 활용한다. 정정 자료에 재적합한 공개 boosted ensemble을 비교군으로 두고, 정확한 동일 약물과 assay 방향에서 두 축을 별도로 평가했다. [Carli et al., Nucleic Acids Research](https://doi.org/10.1093/nar/gkaf414)
- PDO 프리프린트는 네 조직 스크린 연구와 세포주 학습, 합산 학습 및 PDO 미세조정 판을 비교한다. 여기서는 공개 HCC 연구의 같은 환자·약물을 사용해 환자 단위와 약물 지원을 맞추고, 같은 K명에서 단순 평균·수축·RNA의 한 배율을 비교했다. [Zhu et al., arXiv:2611.15769](https://arxiv.org/abs/2611.15769)

## 한계와 중단

공개 코호트는 앞선 평가에 노출되었다. RNA와 AUC의 플랫폼·용량 창 차이는 같은 exact parent에서도 남는다. 3D의 환자 재표집 구간은 기존 공개 관측에 조건부다. 여러 코호트와 K의 구간은 다중검정 보정 없이 개별 탐색 결과로 해석한다. Wong K=8은 회차당 두 환자, Yang 정확 일치 패널은 세 약물이다. 모든 구간을 명시하고 복수의 코호트에서 공통으로 개선했다는 주장은 등록된 두 축을 통과한 경우에만 한다. 2D 복잡 모델이 단순 기준을 못 넘는 축에서 증분 효용을 주장하지 않는다. 3D RNA 배율이 같은 현지 라벨을 사용한 단순 대조를 이기지 못한 조건에서는 현지 약물 순위 보정만 채택한다. 신규 검증 실험 전에는 현장 효용 판정을 유지한다.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    render(parser.parse_args().report_dir)
