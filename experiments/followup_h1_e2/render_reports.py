"""Render follow-up reports from the signed small aggregate release."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

SCREENS = ("CTRPv2", "GDSC2")
ARMS = ("AnchorOnly", "Dual")
CURVES = ("Full-curve", "Reduced-8pt", "Reduced-6pt", "Reduced-3pt", "Single-dose", "Full-curve-1rep")
BASES = ("B0", "B1")
NDCG = "ndcg_at_5"
PRECISION = "precision_at_5"
RHO = "within_drug_spearman"
INTERACTION = "interaction_spearman"
COMP = "COMPRESSION-PASS"
UTILITY = "UTILITY-PASS"
ALT = "RHO-IMPROVEMENT_NDCG-NONINFERIOR"
REQUIRED = ("PLAN_LOCK.md", "E2_CURVE_SUPPORT.md", "E2_H1_MEASUREMENT_BUDGET.md",
            "FIXED_PANEL_CI_SENSITIVITY.md", "H1_GATE_AND_CLAIM_SUMMARY.md")


def load(path):
    return json.loads(path.read_text())


def tsv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def table(headers, rows):
    def row(values):
        return "| " + " | ".join(str(v).replace("|", "\\|") for v in values) + " |"
    return "\n".join((row(headers), row(["---"]*len(headers)), *(row(r) for r in rows)))


def fmt(value, decimals=4, signed=False):
    if value in ("", None, "NA"):
        return "—"
    n = float(value)
    if abs(n) < .5 * 10**(-decimals):
        n = 0.
    return format(n, ("+" if signed else "") + f".{decimals}f").replace("-", "−")


def contrast(row, keys=("delta", "lo95", "hi95"), digits=4):
    if not row or row[keys[0]] in ("", "NA"):
        return "—"
    delta, lo, hi = (fmt(row[k], digits, True) for k in keys)
    return f"{delta} [{lo}, {hi}]"


def verdict(key):
    return {"REFERENCE": "기준", "NOT-APPLICABLE": "—", COMP: "통과",
            "COMPRESSION-NOT-PASSED": "미통과", UTILITY: "순위 통과",
            ALT: "ρ 개선·NDCG 비열등", "UTILITY-NOT-PASSED": "미통과",
            "INFEASIBLE": "지원 없음", "INCONCLUSIVE": "표본 부족"}.get(key, key)


def report(out, name, summary, body):
    tags = ("질문", "실제 바뀐 것", "주 비교 결과·구간", "판정", "다음 결정")
    if len(summary) != len(tags) or any("\n" in s for s in summary):
        raise ValueError(f"Required five one-line takeaways: {name}")
    head = "".join(f"**{label}** — {v}<br>\n" for label, v in zip(tags, summary))
    (out / name).write_text(head + "\n" + body.strip() + "\n")


def e2_tables(out):
    metrics, intervals, costs = map(lambda n: tsv(out / n),
        ("E2_METRICS.tsv", "E2_PAIRED_95CI.tsv", "E2_COST_AND_GATES.tsv"))

    def primary(rows):
        return (r for r in rows if r["scope"] == "primary" and r["level"] == "study")
    p = {(r["screen"], r["arm"], r["condition"], r["metric"]): r for r in primary(metrics)}
    ci = {(r["screen"], r["model_condition"], r["reference"], r["metric"]): r for r in primary(intervals)}
    cost = {(r["screen"], r["arm"], r["condition"]): r for r in primary(costs)}
    return p, ci, cost


def headline(ci, cost, screen, condition):
    r = cost[screen, "Dual", condition]
    contrasts = [contrast(ci.get((screen, f"Dual/{condition}", "Dual/Full-curve", m))) for m in (NDCG, PRECISION)]
    return (f"{screen} {condition} {fmt(r['wells_max'], 0)}웰, "
            f"ΔNDCG {contrasts[0]}, ΔPrecision {contrasts[1]}")


def render_plan(out, ci, cost, fixed):
    contract = out.parents[1] / "experiments/followup_h1_e2/contract.json"
    n = sum(r["zero_test_changed"] == "True" for r in fixed)
    report(out, "PLAN_LOCK.md",
      ("실측 농도를 줄일 때 순위가 유지되는지, 공여자만 재표집할 때 HCC 신뢰구간이 달라지는지 확인한다.",
       "H1-v2 동결 모델·Q8을 재사용하고 E2 입력만 바꿨다. P0는 동결 예측의 재표집 축을 바꿨다.",
       f"{headline(ci, cost, 'CTRPv2', 'Reduced-8pt')}. P0는 {len(fixed)}개 대조 중 {n}개에서 0 포함 여부나 방향이 달라졌다.",
       "E2 압축과 B0·B1 대비 효용을 독립 판정했다. P0는 기존 계약의 사후 민감도다.",
       "2D 지원 분모에서 두 E2 판정을 함께 확인하고, HCC 원 판정과 사후 구간을 함께 제시한다."),
      f"""
# H1 후속 검증 잠금

채점 전에 커밋 **1b6625c**에서 연구 계약을 고정했다. 계약 SHA-256은 {sha256(contract.read_bytes()).hexdigest()}이다. 기준 H1-v2는 **fa64b59**, 2D 원자료 감사 기준은 **5ec8ed2**, P0는 **1912b5d**다. 사용자 명세 SHA-256은 bbfad9a70f3c223366fb5dbb53426242f047fc109f611045024ed223f22a6b4f이다. [실행 계약](../../experiments/followup_h1_e2/contract.json)

| 구분 | 사전 설계 |
| --- | --- |
| E2 평가 단위 | CTRPv2·GDSC2 각각 3개 원래 외부 폴드. 각 폴드의 60개 exact-parent Q8과 K=5 순서 유지. 두 팔 K=5 합집합의 완전 원곡선 지원을 갖춘 계통을 공통 분모로 사용 |
| 물리 농도 | 기존 h1v2-anchor-selector 첫 Inner split 적합부의 약물별 최빈 완전 grid 사용. 로그 등분 위치에 가장 가까운 실제 점을 양 끝 포함 선택. 단일점은 로그 구간 2/3에 가장 가까운 실측점 |
| endpoint | 관측 생존율을 0–1로 제한하고 실제 로그 µM에서 사다리꼴 AUC 계산. Full→모델 원 척도의 약물별 단조 선형 변환은 모든 축소 곡선에 공통 적용. 단일점은 단조 Single→Full 매핑을 먼저 적용. Inner 적합부 20개 이상의 쌍 요구 |
| 반복 | CTRPv2는 다른 plate barcode의 두 웰과 고정 첫 웰을 비교. GDSC2 기본은 한 웰, 같은 물리 곡선에 두 별도 기술 웰이 있는 경우 2 대 1 비교 |
| 점수와 구간 | NDCG@5, Precision@5, 약물 내 Spearman, 정답 Top-5 누락, 추천 Top-5의 실측 평균, Oracle regret. 2,000회 원래 폴드 내 계통 재표집 및 폴드 동등 가중. 폴드당 최소 20계통 |
| 압축 | 모든 폴드 웰 절감 ≥20%, Full 대비 NDCG 하한 ≥−0.01, Precision 하한 ≥−0.02 |
| 무측정 효용 | B0와 B1 각각에 대한 NDCG 하한 >0. 약물 내 ρ 하한 >0과 NDCG 하한 ≥−0.01은 별도 표시 |
| P0 | LIMORE 50계통×68약물 및 LICOB 29공여자×53약물. 원 교차 bootstrap 재현 후 2,000회 계통·공여자만 복원추출. LICOB는 K=2/4/8과 원래 100개씩의 에피소드에서 공여자 가중치 공유 |
| 채점 | E2 키 감사 → K5 실측 입력과 동결 예측 → 라벨 없는 봉인 검증 → 새 E2 채점 1회. P0 원 예측·라벨·분할 해시와 원 수치 검산 후 고정 패널 CI 계산 |

H1 모델 선택, RNA 표현, A8과 앵커 순서, reference atlas를 재사용했다. G1 분석은 CPU 전용 노드에서 fu-h1 작업으로 실행했다. 행별 산출물은 격리된 G1 경로에 저장했다. 처음 P0 실행의 원 소스 import 경로와 처음 E2 예측 실행의 상위 출력 디렉터리를 수정했다. 해당 E2 예측 실행에서는 Q 반응이 열리지 않았다. [작업 원장](SLURM_JOBS.tsv) · [통합 영수증](score_receipt.json)
""")


def render_support(out, support, ci, cost):
    folds = support["folds"]
    totals = [(s, sum(f["common_curve_held_count"] for f in folds.values() if f["study"] == s),
               sum(f["archived_original_held_count"] for f in folds.values() if f["study"] == s)) for s in SCREENS]
    coverage = ", ".join(f"{s} {n}/{original}" for s, n, original in totals)
    rows, anchors = [], []
    for name, f in sorted(folds.items()):
        qa, atlas = f["q8_metadata_support_audit"], f["original_reference_atlas"]
        n, old = f["common_curve_held_count"], f["archived_original_held_count"]
        rows.append((name, f"{n}/{old} ({100*(old-n)/old:.1f}% 제외)", f["inner_fit_lineages"],
                     f["original_held_count_with_complete_two_rep"], qa["parents_with_raw_keys"],
                     f"{qa['registered_pairs_with_raw_keys']}/{qa['registered_sample_parent_pairs']}",
                     atlas["fit_lineages"], atlas["native_measured_training_pairs"],
                     atlas["raw_treatment_well_keys_found"], atlas["illustrative_12_controls_per_lineage"],
                     atlas["raw_treatment_well_keys_found"] + atlas["illustrative_12_controls_per_lineage"]))
        for parent, a in f["a8_curve_support_audit"].items():
            full = a.get("dose_schedules_uM", {}).get("Full-curve")
            window = f"{full[0]:.5g}–{full[-1]:.5g} / {len(full)}" if full else "—"
            fit = a.get("native_calibration_fit_only") or {}
            single = a.get("single_to_full_fit_only") or {}
            rep = a.get("measured_held_replicate_error", {})
            anchors.append((name, f"{a['name']} · {parent}", ", ".join(a["native_drugs"]),
                ", ".join(a["modes_k5"]) or "A8만", window, f"{fit.get('n', 0)}/{single.get('n', 0)}",
                a["held_registered_sample_pairs"], a["held_pairs_missing_raw"], a["held_pairs_no_resolved_curve"],
                a.get("held_lineages_on_locked_grid_with_primary_reps", 0), rep.get("pairs", 0)))
    inputs = [(s, a["raw_well_identity_rows"], a["anchor_measured_rows"], a["anchor_resolved_curves"],
               a["anchor_complete_two_rep_curves"], sum(a["anchor_incomplete_curves"].values()))
              for s, a in sorted(support["screens"].items())]
    r = cost["CTRPv2", "Dual", "Reduced-8pt"]
    report(out, "E2_CURVE_SUPPORT.md",
      ("두 공개 스크린에서 동결 H1-v2 K=5 앵커를 물리 웰에서 재구성할 수 있는가?",
       "native 시료와 exact parent를 물리 농도·시간·반복 키에 연결했다.",
       f"완전 K5 지원은 {coverage}계통이다. {headline(ci, cost, 'CTRPv2', 'Reduced-8pt')}.",
       f"Q8 유지. CTRPv2 Dual 8점 압축 {verdict(r['compression'])}. GDSC2 8점과 패널 전체 2반복은 지원 없음.",
       "확인된 생존율 분모에서 해석한다. LICOB 3D의 농도별 원자료 지원은 확인되지 않았다."),
      f"""
# E2 원곡선 지원과 적합 경계

CTRPv2는 저자 공개 per_cpd_well, per_cpd_post_qc, 실험·plate·시료·화합물 메타데이터를 사용했다. SNP-matched 실험과 master_ccl_id를 DepMap 모델에 연결했다. 웰 키는 실험 ID·native 약물·µM·plate barcode다. 생존율은 DMSO 차감 로그 신호를 2의 거듭제곱으로 환산해 [0,1]에 제한했다. 처리 후 72시간은 공개 논문의 공통 설계다. QC 파일에서는 농도 키만 사용하고, 모든 농도에서 두 웰로 복원되는 16점 곡선을 주 분석에 적용했다. [CTRP 원자료](https://portals.broadinstitute.org/ctrp.v2.1/) · [원 논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC4631646/)

GDSC2는 공개 27Oct23 raw archive의 Sanger 모델 ID·native drug ID·barcode·scan·제작일·라이브러리·well position·µM 키를 사용했다. 시간은 배포 파일의 DURATION=4 필드를 그대로 표기한다. Glo, 차량 대조 ≥8개, blank ≥4개, Z′≥0.3과 등록 물리 농도 범위 양 끝을 확인했다. 두 스크린의 native 약물 ID는 vc_build_v1의 exact-parent 등록에 연결했다. [GDSC 배포 설명](https://depmap.sanger.ac.uk/documentation/datasets/drug-sensitivity/) · [종말점 프로토콜](../vc_build_v1/v0plus/LICOB_DOSE_ENDPOINT_PROTOCOL.md) · [곡선 전이](../vc_build_v1/v0plus/LICOB_CURVE_TRANSFER_RESULTS.md)

{table(("스크린", "공개 웰 키", "앵커 수치 웰", "QC 통과 곡선", "전 농도 두 웰", "불완전 탈락 곡선"), inputs)}

**원래 H1-v2 전체 분모와 다른 원곡선 지원 서브셋이다.** A8 전체를 감사하고 두 팔의 고정 K5 합집합에 완전한 계통만 포함했다. GDSC2 첫 폴드에서 팔별 K5가 달라 합집합을 적용했다. 원 Q8의 60개 후보와 계통별 기존 Q 관측 마스크는 유지했다. Q 원자료 매칭은 식별 키 감사다. GDSC2 Q 원자료에는 plate 수치 QC 적용 전 키가 포함된다. 스크린 단위 추출은 폴드들의 A8 합집합을 읽는다. 각 폴드의 입력과 적합은 그 폴드의 앵커 약물 및 허용 계통에 한정했다. 한 약물이 다른 폴드의 앵커가 될 수 있다. 해당 폴드의 Q 반응값은 예측 봉인 뒤 기존 H1 Q8 정답으로 채점할 때 사용했다. [예측 봉인](e2_forecast_seal_receipt.json)

{table(("폴드", "E2/원 평가 계통", "Inner 적합 계통", "K5 두 웰 계통", "Q8 raw 약물/60",
         "Q8 raw 시료×약물/등록", "atlas 적합 n", "atlas native 실측쌍", "atlas raw 처리 웰 키",
         "atlas 12대조/계통 가정", "atlas 처리+가정 대조 합계"), rows)}

기존 68약물 reference atlas를 재사용해 신규 구축에 든 웰은 0이다. 표의 처리 웰 키 수는 atlas 계통에 물리 키가 연결되는 공개 archive의 관측값이다. 폴드와 스크린 간에 계통이 겹친다. 대조 열은 재구축 시 12대조/계통을 가정하며 과거 실험의 실제 대조 웰 총수와 구별한다.

## A8 exact parent별 지원

원자료 미매칭·curve 탈락은 시료×약물 값이고, 적격과 2웰은 약물별 계통 수다. 전체 K5 반복 지원은 앞 표에 있다. 약물별 Full·8·6·3·단일점의 실제 µM, fit-only 변환 계수, 반복에서 직접 관측한 AUC 및 웰 변동, 원파일과 SHA-256은 영수증에 있다. [지원 영수증](curve_support_receipt.json)

{table(("폴드", "공개 약물명 · exact parent", "native ID", "K5 용도", "Full µM / 점",
         "Full / 단일 적합 n", "등록쌍", "raw 미매칭", "raw 후 curve 탈락", "잠금 grid 적격", "2웰"),
         anchors)}

LICOB에서는 S5 농도 설계와 S6–S8 AUC·IC50·Emax가 공개 등록돼 있다. 농도별 3D 생존율 원자료는 등록되지 않았다. 2D에서 확인한 물리 농도 지원은 3D의 독립 실측 근거가 아니다. [H1-v2 LICOB 자산 범위](../nextgen_h1_v2/ROUND1_TO_3_CEILING_TUNING.md)
""")


def render_budget(out, scores, ci, cost):
    rows, uses = [], []
    for screen in SCREENS:
        for arm in ("No-anchors", *ARMS):
            for cond in (*BASES, *CURVES, "Published-anchor"):
                key = (screen, arm, cond)
                if key not in cost:
                    continue
                r = cost[key]
                value = lambda m: fmt(scores[key + (m,)]["value"])
                name = f"{arm}/{cond}"
                delta = [contrast(ci.get((screen, name, f"{arm}/Full-curve", m))) for m in (NDCG, PRECISION, RHO)]
                fraction = r["well_reduction_min_fraction"]
                saving = f"{float(fraction)*100:.1f}%" if fraction not in ("", "NA") else "—"
                rows.append((screen, arm, cond, r["lineages"], fmt(r["wells_max"], 0), saving, value(NDCG),
                             *delta, value("missed_true_top5_fraction"), value("selected_top5_actual_mean"),
                             value("oracle_top5_regret"), verdict(r["compression"])))
                if arm in ARMS and cond != "Published-anchor":
                    result = []
                    for base in BASES:
                        result += [contrast(ci.get((screen, name, base, metric))) for metric in (NDCG, RHO)]
                        result.append(verdict(r[f"utility_vs_{base.lower()}"]))
                    uses.append((screen, arm, cond, *result, verdict(r["utility_vs_both"])))

    def passed(utility):
        return ", ".join(f"{s} {a} {c}" for (s, a, c), r in cost.items()
                         if r["compression"] == COMP and r["utility_vs_both"] == utility) or "해당 없음"

    rep = {s: [r for r in tsv(out / "E2_COST_AND_GATES.tsv") if r["screen"] == s and
               r["scope"] == "two_rep_subset" and r["level"] == "study" and r["condition"] == "Full-curve-1rep"]
           for s in SCREENS}
    ctrp_rep = [a["measured_held_replicate_error"]
                for f in load(out / "curve_support_receipt.json")["folds"].values()
                if f["study"] == "CTRPv2" for a in f["a8_curve_support_audit"].values()
                if a["modes_k5"] and "measured_held_replicate_error" in a]
    n_rep = sum(r["pairs"] for r in ctrp_rep)
    auc_repeat = sum(r["pairs"] * r["mean_abs_auc_between_wells"] for r in ctrp_rep) / n_rep
    well_repeat = sum(r["pairs"] * r["mean_abs_viability_between_wells"] for r in ctrp_rep) / n_rep
    if len(rep["GDSC2"]) != 2 or any(r["status"] != "INFEASIBLE" for r in rep["GDSC2"]):
        raise ValueError("GDSC repeat support changed from the reported comparison")
    report(out, "E2_H1_MEASUREMENT_BUDGET.md",
      ("K=5 앵커의 농도와 반복을 줄여도 Top-5 및 무측정 대비 효용이 유지되는가?",
       "동일한 H1-v2 모델·atlas·Q8에 실측 Full·축소곡선·단일점·1반복 입력을 바꿔 넣었다.",
       headline(ci, cost, "CTRPv2", "Reduced-8pt") + ". " + headline(ci, cost, "GDSC2", "Reduced-3pt") + ".",
       f"압축과 B0·B1 대비 순위 효용 동시 통과는 {passed(UTILITY)}이다.",
       "조건별 정밀도 하한과 실제 지원 분모를 적용한다. LICOB 3D 92웰의 실측 판정에는 별도 원곡선이 필요하다."),
      f"""
# E2 실측 농도 감소와 비용

**원래 H1-v2 전체 분모와 다른 원곡선 지원 서브셋이다.** K=5이고 각 폴드의 원 Q8 후보 60개를 유지했다. 같은 스크린의 모든 팔과 대조는 동일한 계통과 Q 관측 마스크를 쓰며 3개 외부 폴드 평균을 동등 가중했다. 폴드 안의 독립 계통만 2,000회 짝지어 복원추출했다. Published-anchor는 이 분모에 원 H1-v2의 앵커 요약 입력을 그대로 준 진단용 대조다. [지원 감사](E2_CURVE_SUPPORT.md) · [전체 지표](E2_METRICS.tsv) · [짝지은 구간](E2_PAIRED_95CI.tsv)

## 측정 압축

비용은 K=5에 대해 실제 농도점×웰 반복을 합하고 대조 12웰을 더했다. B0와 B1에는 새 약물 측정과 plate 대조가 없어 증분 0웰이다. RNA 측정비는 이 약물 웰 회계 밖에 있다. 96 초과는 목표 plate 한 장에 들어가지 않으며, 실제 공개 스크린의 실험 배치와 회계 가정은 다르다. Atlas의 공개 처리 웰 키 및 대조 가정은 지원 보고서에 있다. [비용 및 판정 원표](E2_COST_AND_GATES.tsv)

{table(("스크린", "팔", "조건", "계통", "총 웰", "절감", "NDCG@5", "ΔNDCG [95%]",
        "ΔPrecision [95%]", "Δ약물 내 ρ [95%]", "정답 Top-5 누락률", "추천 Top-5 실측 평균",
        "Oracle regret", "압축"), rows)}

Δ는 각 팔의 Full-curve 대비다. 압축 조건은 웰 ≥20% 절감, NDCG 하한 ≥−.010, Precision 하한 ≥−.020을 모두 적용한다. CTRPv2 Dual의 3점은 75.6% 절감했으나 Precision 하한이 허용값을 넘지 못했다. GDSC2 6점은 10.6% 절감에 그쳤다. 추천 Top-5 평균과 regret는 같은 스크린의 감수성 방향으로 변환된 native Q8 AUC 척도다. regret는 Oracle Top-5 평균에서 추천 Top-5 평균을 뺀 값이다. 누락률은 1−Precision@5다. 컷오프 동률은 기대 멤버십으로 처리했다.

CTRPv2의 Full-curve-1rep는 같은 16농도 실험·계통에서 두 웰 평균을 고정 첫 웰로 바꾼 실제 반복 민감도다. K5에 쓰인 약물의 완전한 원 두 웰 계통×약물 {n_rep:,}쌍에서 웰 간 평균 절대 차이는 정규화 AUC {auc_repeat:.4f}, 농도별 생존율 {well_repeat:.4f}였다. 이 기술값에는 일부 K5 앵커만 완전한 계통도 포함된다. GDSC2는 K=5의 원 grid가 약물당 7점·1반복이므로 8점이 정의되지 않았다. 같은 물리 곡선의 두 웰로 전체 K5를 구성할 수 있는 GDSC2 평가 계통은 0명이어서 반복 차이를 추정할 수 없었다. 두 웰이 있던 일부 약물의 AUC·생존율 차이는 지원 영수증에 있다. [지원 영수증](curve_support_receipt.json)

## 무측정 대비 효용

{table(("스크린", "팔", "조건", "B0 ΔNDCG", "B0 Δ약물 내 ρ", "B0 판정",
        "B1 ΔNDCG", "B1 Δ약물 내 ρ", "B1 판정", "동시 판정"), uses)}

순위 통과는 ΔNDCG의 짝지은 하한 >0이다. ρ 개선·NDCG 비열등은 ρ 하한 >0과 NDCG 하한 ≥−.01을 확인한 별도 신호다. 압축도 통과하면서 두 무측정 대조 중 하나 이상에서 이 두 번째 기준에 의존한 조건은 {passed(ALT)}이다. B0는 새 시료의 앵커 측정이 없는 현지 약물 평균이며, B1은 RNA만 사용한 기존 구조 커널이다.

공개 2D 원곡선에서 관측한 결과다. H1-v2의 LICOB는 공개 AUC에 가한 인공 노이즈로 민감도를 검토했다. CTRPv2의 92웰 예산을 LICOB 3D에 적용할 근거에는 실제 3D 농도별 생존율과 반복 측정이 필요하다.
""")


def render_p0(out, receipt, fixed):
    kept = {"structured", "ridge_structured_50_50", "pairwise_top5", "ecfp4_path_elasticnet"}
    rows = []
    for r in fixed:
        if r["study"] == "LICOB" or r["model_id"] in kept and r["baseline"] == "drug_mean":
            rows.append((r["study"], r["k"] if r["study"] == "LICOB" else "—", r["model_id"], r["baseline"],
                         r["metric"], contrast(r, ("delta", "original_low", "original_high")),
                         contrast(r, ("delta", "fixed_low", "fixed_high")),
                         "변경" if r["zero_test_changed"] == "True" else "유지"))
    gates = [g for trial in receipt["analyses"].values() for g in trial["resample"]["gates"]]
    changed = [g for g in gates if g["original_joint_gate"] != g["fixed_counterfactual_joint_gate"]]
    selected = next(r for r in fixed if r["study"] == "LIMORE" and r["model_id"] == "structured"
                    and r["baseline"] == "drug_mean" and r["metric"] == NDCG)
    nd, pr = (sum(r["zero_test_changed"] == "True" and r["metric"] == m for r in fixed) for m in (NDCG, PRECISION))
    lc = sum(r["zero_test_changed"] == "True" and r["study"] == "LICOB" for r in fixed)
    d = contrast(selected, ("delta", "original_low", "original_high"))
    new = contrast(selected, ("delta", "fixed_low", "fixed_high"))
    report(out, "FIXED_PANEL_CI_SENSITIVITY.md",
      ("후보 약물을 고정하면 LIMORE·LICOB의 추천 신뢰구간과 기존 판정이 달라지는가?",
       "원 동결 예측과 점추정량을 재현하고, 독립 계통 또는 공여자만 2,000회 재표집했다.",
       f"LIMORE 구조 커널 대 약물 평균 ΔNDCG는 교차 {d}, 고정 {new}였다.",
       f"LIMORE 공동 기준 {len(changed)}개가 사후 민감도에서 바뀌었다. LICOB K=2/4/8은 모두 미통과를 유지했다.",
       "기존 확증 판정은 보존하고 고정 패널 조건에서 달라지는 불확실성을 함께 보고한다."),
      f"""
# P0 고정 패널 사후 민감도

원 커밋은 **1912b5d**다. 동결 예측, 원 라벨, LICOB 보정/평가 분할과 원 집계 영수증을 해시 대조했다. 원 metrics.py와 fewshot.py로 원 점추정량과 교차 bootstrap을 재현했다. 수치 최대 오차는 5×10⁻¹¹ 미만으로 원 TSV의 반올림 범위였다. 학습 및 예측 재생성은 0회였다. [재현 영수증](p0_replay_receipt.json)

LIMORE는 독립 50계통과 exact parent 68개다. 원 구간은 계통×약물 2,400회 교차 재표집이었다. LICOB는 독립 29공여자와 exact parent 53개이며 H1의 38공여자와 분석 로스터가 다르다. K=2·4·8에서 평가 공여자는 반복별 27·25·21명이다. 보정된 100개 반복 예측을 각각 재사용했다. 원 구간은 공여자×약물 500회 교차 재표집이다. 고정 패널에서는 원 cohort에서 뽑은 공여자 가중치를 모든 K·반복·모델에 동일 적용했다. 각 반복의 원래 평가 멤버십을 유지한 뒤 유효한 반복 평균을 동등 가중했다. 점추정량과 기대 동률은 동일하다.

아래는 LIMORE의 주요 약물 평균 대조와 LICOB의 추천 대조 전체다. 구간의 0 포함 여부나 부호는 NDCG {nd}개, Precision {pr}개에서 바뀌었으며 LICOB는 두 지표를 합쳐 {lc}개가 바뀌었다. 공동 판정은 별도로 확인했다. [전체 {len(fixed)}개 대조](FIXED_PANEL_CI_SENSITIVITY.tsv)

{table(("코호트", "K", "모델", "대조", "지표", "원 교차 Δ [95%]", "고정 패널 Δ [95%]", "0 기준"), rows)}

## 원 공동 판정과 고정 패널을 대입한 판정

{table(("코호트·설계", "모델", "원 공동 판정", "고정 패널 사후 공동 판정"), [
    (g["track"], g["model_id"], "통과" if g["original_joint_gate"] else "미통과",
     "통과" if g["fixed_counterfactual_joint_gate"] else "미통과") for g in gates])}

LIMORE의 정정 구조 커널 및 Ridge+구조 커널 50:50은 NDCG 하한을 고정 패널 구간으로 바꾸면 사후 공동 기준에 든다. 약물 내 ρ, 이중 중심화 상호작용 ρ, Top-5 다양성, source OOF의 약물 평균·가산 대조에는 원 근거를 적용했다. Precision은 참고 축이다. LICOB의 공동 기준은 NDCG와 상호작용이 현지 평균 및 EB 수축 모두를 넘어야 한다. 고정 패널 구간을 적용해도 LICOB의 6개 모델×K 조합에는 공동 통과가 없었다.

고정 패널 구간은 등록 약물 위에서 계통·공여자 불확실성을 추정한다. 원 교차 구간은 후보 약물이 변할 때의 불확실성까지 포함한다. 두 추정량의 적용 범위를 함께 보고하며 기존 계약과 확증 판정을 보존한다.
""")


GATES = (
    ("v1", "Tier1-macro", "H1-Anchor-Best", "공동 NO-GO. K=0와 N1 양성, Bsimple 미달"),
    ("v1", "LIMORE", "H1-Anchor-Best", "기술 비교. K=0와 N1 양성, Bsimple 하한 음수"),
    ("v1", "LICOB", "H1-Anchor-Best", "공동 NO-GO. Bsimple·Precision·K=0·N1 미달"),
    ("v1", "LIMORE_to_LICOB", "H1-Anchor-Best", "전이 기술값. Bsimple과 N1 양성, K=0 하한 음수"),
    ("v2", "Tier1-macro", "H1-v2-Dual-Best", "Track A GO-DEV. 원안 +.020 기준 미달"),
    ("v2", "Tier1-macro", "H1-v2-AnchorOnly-Best", "보조 AnchorOnly. K=0와 N1 양성, Bsimple 구간은 0 포함"),
    ("v2", "LIMORE", "H1-v2-Dual-Best", "Track A 판정 불가. v1 공개 Dual 부재. 순열 전 대조 진단 NO-GO"),
    ("v2", "LICOB", "H1-v2-AnchorOnly-Best", "Track B 순위 통과·층화 미통과. 원안 Precision·K=0·N1 미달"),
    ("v2", "LICOB", "H1-v2-Dual-Best", "보조 Dual 순위·층화 미통과"),
    ("v2", "LIMORE_to_LICOB", "H1-v2-AnchorOnly-Best", "Track B 전이 순위 통과. B1 부재로 층화 판정 불가"),
)


def render_gates(out):
    reports = out.parent
    sources = {
        "v1": reports / "nextgen_h1_anchor/H1_PAIRED_95CI.tsv",
        "v2": reports / "nextgen_h1_v2/H1V2_PAIRED_95CI.tsv",
    }
    evidence, ledger = {}, []
    watched = ("Bsimple", "B0", "B1", "B2", "B3", "N1", "N1-Dual", "N2",
               "v1-RNA+Anchor", "v1-RNA+Anchor-ungated")
    for version, path in sources.items():
        expected = {(s, m) for v, s, m, _ in GATES if v == version}
        for row in tsv(path):
            model, budget = row["candidate"].rsplit("/", 1)
            base, base_k = row["comparator"].rsplit("/", 1)
            study, metric = row["study"], row["metric"]
            if ((study, model) not in expected or budget != "5" or row.get("panel", "Q8") != "Q8" or
                (base not in watched or base_k != "5") and (base != model or base_k != "0") or
                metric not in (NDCG, PRECISION, RHO, INTERACTION)):
                continue
            control = "K=0" if base == model else base
            evidence[version, study, model, control, metric] = row
            ledger.append({"version": version, "study": study, "candidate": row["candidate"],
                           "contrast": control, "original_comparator": row["comparator"], "metric": metric,
                           **{k: row[k] for k in ("delta", "lo95", "hi95", "independent_groups", "draws")}})
    path = out / "H1_GATE_EVIDENCE.tsv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ledger[0], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ledger)

    def g(v, s, m, ref, metric=NDCG):
        return contrast(evidence.get((v, s, m, ref, metric)))

    rows = [
        ("H1 원안", "사전 계약", "현지 K=5 개인화", "≥+.020 및 하한 >0",
         "Bsimple 풀의 대조", "Bsimple 풀의 대조", "둘 다 하한 >0",
         "Bsimple 대비 Precision 하한 ≥−.020. 주요 적용 범위 Tier 1 또는 LICOB"),
        ("H1-v2 개정", "Track A", "범암·LIMORE Dual", "Bsimple 보조",
         "B1 ΔNDCG ≥+.005, Δ약물 내 ρ ≥+.05, Precision 하한 ≥−.005",
         "B2/B3는 별도 대조", "N1-Dual·N2 순열",
         "B1과 공개 v1 Dual 각각에서 NDCG 또는 약물 내 ρ 하한 >0과 순열"),
        ("H1-v2 개정", "Track B", "LICOB / 전이", "순위 ΔNDCG ≥+.015 및 하한 >0",
         "층화 B1 대비", "층화 B2/B3 대비", "K=0·N1은 별도 근거",
         "층화는 모든 대조 Δ약물 내 ρ ≥+.05, Δ상호작용 ρ 하한 >0, NDCG 하한 ≥−.01"),
    ]
    for v, study, model, outcome in GATES:
        n = "N1-Dual" if model == "H1-v2-Dual-Best" else "N1"
        scalar, affine = (g(v, study, model, control) for control in ("B2", "B3"))
        s_a = scalar if scalar == affine else scalar + " / " + affine
        rows.append((v, study, model, g(v, study, model, "Bsimple"), g(v, study, model, "B1"), s_a,
                     g(v, study, model, "K=0") + " / " + g(v, study, model, n), outcome))
    dual = "H1-v2-Dual-Best"
    anchor = "H1-v2-AnchorOnly-Best"
    report(out, "H1_GATE_AND_CLAIM_SUMMARY.md",
      ("H1 원안, v1 평가, v2 평가에서 어떤 근거가 통과로 이어졌는가?",
       "동결 K=5 Q8의 원 보고 구간으로 공동 개인화, 개정 범암 Dual, 3D 순위와 층화를 대조했다.",
       f"v2 범암 Dual 대 B1 {g('v2', 'Tier1-macro', dual, 'B1')}, LICOB AnchorOnly 대 Bsimple {g('v2', 'LICOB', anchor, 'Bsimple')} ΔNDCG.",
       "v1 공동 NO-GO. v2 범암 GO-DEV, LICOB 현지·전이 제한 순위 통과, 3D 층화는 미입증.",
       "v2의 순위 통과 범위를 그대로 인용하고, LICOB의 개인화·실측 plate 주장은 별개로 다룬다."),
      f"""
# 원안·v1·v2 판정과 실제 대조

같은 원 K=5 Q8의 짝지은 ΔNDCG@5 [95% CI]를 사용했다. 원안은 사전 기준이며 v1이 그 기준을 실행했다. B0는 현지 평균, B1은 RNA-only, B2는 scalar shift, B3는 양의 shrinkage affine이다. Bsimple은 해당 버전의 Inner에서 B0–B4 중 선택한 최강 단순 후보이며 외부에서 모든 기준선보다 높다는 뜻은 아니다. K=0은 같은 추정기의 앵커 0개 조건이고, N1은 해당 팔의 전체 앵커 패킷 순열이다. B0와 K=0이 수치상 같을 때도 원 정의를 구별했다. [v1 사전 계약](../nextgen_h1_anchor/PLAN_LOCK.md) · [v1 결과](../nextgen_h1_anchor/RESULTS.md) · [v2 결과](../nextgen_h1_v2/RESULTS.md)

{table(("기준·버전", "범위", "팔", "Bsimple", "B1 RNA-only", "B2 / B3",
         "K=0 / N1", "판정과 제약"), rows)}

## 통과 축의 정확한 한계

- **v1 범암**에서 Bsimple 대비 Precision 하한은 {g('v1', 'Tier1-macro', 'H1-Anchor-Best', 'Bsimple', PRECISION)}으로 보호 기준 범위였고, K=0와 N1 신호도 관측됐다. Bsimple 대비 NDCG가 원안의 +.020 및 양의 하한을 충족해야 공동 승격이 가능했다. **v1 LICOB**에서는 Precision 하한도 {g('v1', 'LICOB', 'H1-Anchor-Best', 'Bsimple', PRECISION)}였다. v1 B4는 범암에서 RNA-only 및 scalar에 열세였으며 LICOB에서는 두 대조의 구간이 모두 0을 포함했다. [v1 대조](../nextgen_h1_anchor/H1_PAIRED_95CI.tsv)
- **v2 범암 Dual**의 B1 대비 약물 내 ρ는 {g('v2', 'Tier1-macro', dual, 'B1', RHO)}, Precision은 {g('v2', 'Tier1-macro', dual, 'B1', PRECISION)}이다. v1 공개 Dual 대비 NDCG 차이는 +.001055이고 하한은 +9.60×10⁻⁷이다. 약물 내 ρ는 {g('v2', 'Tier1-macro', dual, 'v1-RNA+Anchor', RHO)}다. N1-Dual과 N2 대비 NDCG가 각각 {g('v2', 'Tier1-macro', dual, 'N1-Dual')}, {g('v2', 'Tier1-macro', dual, 'N2')}로 양성이었다. 이 값들이 개정 Track A 통과에 해당한다. [v2 대조](../nextgen_h1_v2/H1V2_PAIRED_95CI.tsv)
- **v2 LIMORE Dual**은 B1 대비 약물 내 ρ {g('v2', 'LIMORE', dual, 'B1', RHO)}였으나 Precision은 {g('v2', 'LIMORE', dual, 'B1', PRECISION)}였다. 공개 v1 Dual이 없어 정식 Track A 비교가 성립하지 않는다. v1 RNA 순열 전 동결 후보를 사용한 진단 NDCG는 {g('v2', 'LIMORE', dual, 'v1-RNA+Anchor-ungated')}다. [LIMORE 대조](../nextgen_h1_v2/H1V2_PAIRED_95CI.tsv)
- **v2 LICOB AnchorOnly**의 B1 대비 약물 내 ρ는 {g('v2', 'LICOB', anchor, 'B1', RHO)}, 상호작용 ρ는 {g('v2', 'LICOB', anchor, 'B1', INTERACTION)}이다. B2·B3 상호작용은 각각 {g('v2', 'LICOB', anchor, 'B2', INTERACTION)}, {g('v2', 'LICOB', anchor, 'B3', INTERACTION)}다. B1·B3 상호작용과 양쪽 scalar/affine NDCG 하한이 층화 기준을 충족하지 못했다. Bsimple 대비 Precision은 {g('v2', 'LICOB', anchor, 'Bsimple', PRECISION)}로 원안 보호 기준도 미달한다. v2판 B3는 원 v1의 단순 affine과 달리 shift까지 수축했다. [B3 구현 감사](../nextgen_h1_anchor/H1_B3_INNER_CORRECTION.tsv) · [v2 대조](../nextgen_h1_v2/H1V2_PAIRED_95CI.tsv)
- **v2 LIMORE→LICOB 전이**의 Bsimple은 현지 적합 공여자 0명과 RNA 0인 계약의 단순 출발점이다. 같은 모델의 K=0, B2/B3, N1은 양의 95% 순위 하한을 확보하지 못했고, B1은 적용 범위 밖이다. 전이 순위 통과는 관측 앵커를 제공한 이 고정 패널에 해당한다. [v2 전이 결과](../nextgen_h1_v2/RESULTS.md)

H1-v1/v2와 P0는 다른 비교 계약이다. P0의 29공여자 고정 패널 민감도는 H1-v2의 38공여자 LICOB 결과를 대체하지 않는다. E2에서 새로 얻은 범위는 공개 2D 원곡선의 측정 압축이며 3D 환자 층화나 LICOB의 92웰 채택 근거에는 별도의 실측 검증이 필요하다. [P0 고정 패널](FIXED_PANEL_CI_SENSITIVITY.md) · [E2 측정 압축](E2_H1_MEASUREMENT_BUDGET.md) · [원 게이트 수치 발췌](H1_GATE_EVIDENCE.tsv)
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    out = args.reports.resolve()
    score, ci, cost = e2_tables(out)
    fixed = tsv(out / "FIXED_PANEL_CI_SENSITIVITY.tsv")
    support = load(out / "curve_support_receipt.json")
    p0 = load(out / "p0_replay_receipt.json")
    render_plan(out, ci, cost, fixed)
    render_support(out, support, ci, cost)
    render_budget(out, score, ci, cost)
    render_p0(out, p0, fixed)
    render_gates(out)
    print(f"Rendered {len(REQUIRED)} reports")


if __name__ == "__main__":
    main()
