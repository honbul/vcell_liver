"""Render evidence-linked v2 reports from the signed aggregate tables."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

ANCHOR = "H1-v2-AnchorOnly-Best"
DUAL = "H1-v2-Dual-Best"
NDCG = "ndcg_at_5"
RHO = "within_drug_spearman"
PRECISION = "precision_at_5"
INTER = "interaction_spearman"
MAIN_STUDIES = ("Tier1-macro", "GDSC2", "CTRPv2", "PRISM20Q2", "LIMORE", "LICOB", "LIMORE_to_LICOB")


def load(root, name):
    with (root / name).open() as table:
        return list(csv.DictReader(table, delimiter="\t"))


def num(value):
    return float(value) if value not in {"", "NA", None} else float("nan")


def fmt(value, signed=False):
    x = num(value)
    return format(x, "+.4f" if signed else ".4f") if math.isfinite(x) else "—"


def interval(record, name="value"):
    if not record or not math.isfinite(num(record[name])):
        return "—"
    delta = name == "delta"
    return f"{fmt(record[name], delta)} [{fmt(record['lo95'], delta)}, {fmt(record['hi95'], delta)}]"


def study(task):
    return task.rsplit("-", 1)[0]


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def average(rows, key):
    values = [num(row[key]) for row in rows if math.isfinite(num(row[key]))]
    return mean(values) if values else float("nan")


def flat_recipe(row):
    recipe = json.loads(row["recipe"])
    stage = recipe.pop("stage", {})
    return recipe | {f"stage.{key}": value for key, value in stage.items()}


def boundary_summary(grid, winners, records):
    tested = grouped(grid, ("task", "modality", "family"))
    probes = defaultdict(list)
    for winner in winners:
        cfg = flat_recipe(winner)
        key = tuple(winner[k] for k in ("task", "modality", "family"))
        neighbors = defaultdict(dict)
        for row in tested[key]:
            comparison = flat_recipe(row)
            fields = [n for n in cfg.keys() | comparison.keys() if cfg.get(n) != comparison.get(n)]
            if len(fields) != 1:
                continue
            name = fields[0]
            value = comparison.get(name)
            if isinstance(cfg.get(name), (int, float)) and isinstance(value, (int, float)):
                neighbors[name][value] = num(row["objective"]) - num(winner["objective"])
        for name, trials in neighbors.items():
            lower = max((v for v in trials if v < cfg[name]), default=None)
            upper = min((v for v in trials if v > cfg[name]), default=None)
            probes[key[-1], name].append((trials.get(lower), trials.get(upper)))
    lines = ["### Matched numerical turnover", "",
             "| Family | Parameter | Lower probes | Median lower Δ utility | Upper probes | Median upper Δ utility | Two-sided brackets |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (family, name), pairs in sorted(probes.items()):
        lower, upper = ([pair[i] for pair in pairs if pair[i] is not None] for i in (0, 1))
        limits = [len(lower), fmt(str(median(lower)), True) if lower else "—",
                  len(upper), fmt(str(median(upper)), True) if upper else "—"]
        two_sided = sum(lo is not None and hi is not None for lo, hi in pairs)
        lines.append(f"| {family} | {name} | {' | '.join(map(str, limits))} | {two_sided} |")
    lines += ["", "Each probe fixes all other recipe fields and uses the nearest tested value on its side. "
              "Each reported median uses the probe-minus-winner utility differences for that parameter and direction. "
              "One-sided rows show the side tested. The complete grid also records joint recipe changes. "
              "[Grid scores](H1V2_INNER_GRID.tsv) · [Boundary trace](H1V2_INNER_BOUNDARIES.tsv)", ""]
    limits = [r for r in records if r["phase"] == "final" and r["status"] == "domain-limit"]
    if limits:
        lines += ["| Family | Parameter at its tested limit | Value | Selected fold-modality recipes |",
                  "|---|---|---:|---:|"]
        for (family, name, value), rows in sorted(grouped(limits, ("family", "parameter", "winner")).items()):
            lines.append(f"| {family} | {name} | {value} | {len(rows)} |")
        lines.append("")
    return lines


def learning_summary(root):
    rows = load(root, "H1V2_LEARNING_CURVES.tsv")
    traces = []
    for (task, mode, family, _, head), points in grouped(rows, ("task", "modality", "family", "fold", "head")).items():
        points.sort(key=lambda r: int(r["epoch"]))
        metric = "NDCG@5" if family == "C" else points[0]["metric"]
        train, val = ("train_ndcg", "val_ndcg") if family == "C" else ("train", "val")
        direction = -1 if metric == "rmse" else 1
        best = max(points, key=lambda row: direction * num(row[val]))
        traces.append(dict(study=study(task), modality=mode, family=family, head=head, metric=metric,
                           start=points[0][val], peak=best[val], end=points[-1][val], train=best[train],
                           epoch=best["epoch"]))
    lines = ["### Selected train and validation curves", "",
             "| Study | Mode | Family | Head / metric | Validation first | Validation best | Train at best | Validation last | Mean best step |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for (s, mode, family, head, metric), points in sorted(grouped(traces, ("study", "modality", "family", "head", "metric")).items()):
        metrics = [fmt(str(average(points, field))) for field in ("start", "peak", "train", "end")]
        lines.append(f"| {s} | {mode} | {family} | {head if head != 'NA' else 'set'} / {metric} | "
                     f"{' | '.join(metrics)} | {average(points, 'epoch'):.1f} |")
    lines += ["", "For each fold, the table reports the first, best and final recorded validation points and the train value "
              "at the best validation point. A late validation decline despite continued fit separates overfitting from lack "
              "of early learning. Neural checkpoint selection used the combined ranking and Spearman objective. "
              "The full selected traces include neural within-drug Spearman. [Learning curves](H1V2_LEARNING_CURVES.tsv)", ""]
    return lines


def inner(root):
    anchors = load(root, "H1V2_INNER_ANCHORS.tsv")
    winners = load(root, "H1V2_INNER_WINNERS.tsv")
    receipt = json.loads((root / "inner_receipt.json").read_text())
    for t in anchors + winners:
        t["study"] = study(t["task"])
    lines = [
        "# H1-v2 inner ceiling search", "",
        f"The signed inner searches evaluated {receipt['grid_rows']:,} fold-aggregated recipes. "
        "Each recipe was compared on three held inner folds per task. "
        "Supervised anchor search cross-fitted inside each of those fit folds.", "",
        "## Round 1 anchor ordering", "",
        "| Policy | Mode | Mean ΔNDCG@5 vs K=0 | Mean Δ drug rho vs K=0 | Residual R² | Mean diversity | Selected tasks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    fixed = [x for x in anchors if x["scope"] == "fixed8"]
    by_policy = grouped(fixed, ("policy", "modality"))
    for (policy, mode), rows in sorted(by_policy.items()):
        count = sum(w["family"] == "A" and w["policy"] == policy and w["modality"] == mode for w in winners)
        means = [fmt(str(mean(num(x[k]) for x in rows)), k.startswith("delta")) for k in
                 ("delta_ndcg", "delta_rho", "residual_r2", "tanimoto_diversity")]
        lines.append(f"| {policy} | {mode} | {' | '.join(means)} | {count} |")
    for mode in ("anchor", "dual"):
        fixed_seeds = [r for r in fixed if r["modality"] == mode and r["policy"].startswith("random")]
        metrics = [fmt(str(mean(num(x[key]) for x in fixed_seeds))) for key in ("delta_ndcg", "delta_rho")]
        lines.append(f"| random-3seed mean | {mode} | {' | '.join(metrics)} | — | — | 0 |")
    lines += ["", "The primary comparison preserves the complete-case denominator and one fit-reserved Q8 per inner split. "
              "The exploratory LICOB 16-candidate reserve holds a common Q fixed across all three supervised policies. "
              "The anchor table includes mechanism counts and per-fold diagnostics. [Round 1 table](H1V2_INNER_ANCHORS.tsv)",
              "", "## Round 2 and Round 3 family comparators", "",
              "| Study | Modality | Family | Mean R2 utility | Mean R3 utility | Mean NDCG@5 | Mean drug rho | Numerical boundary trials pending |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for (s, mode, family), rows in sorted(grouped(winners, ("study", "modality", "family")).items()):
        args = [fmt(str(mean(num(x[k]) for x in rows))) for k in ("round2", "objective", "ndcg", "rho")]
        left = sum(max(0, int(x["remaining"])) for x in rows)
        lines.append(f"| {s} | {mode} | {family} | {' | '.join(args)} | {left} |")
    lines += [
        "", "The fitted policy was shared across families within each modality. Selection used NDCG@5 plus "
        "0.06 × within-drug Spearman, with 0.12 × Spearman for LICOB.", "",
        "- **A** varied EB priors, signal shrinkage and regularization. Its product kernel fits measured drug–sample pairs "
        "in the tensor product of the RNA/anchor and structure/target feature spaces.",
        "- **B** compared L2, Huber, LambdaRank@5 and regression/rank blends. The development group was the LightGBM rank group.",
        "- **C** trained self-attention over the anchor set, query cross-attention and gated RNA FiLM. SmoothL1, Pearson and "
        "ListNet or top-five pairwise loss were applied to the native ordering.",
        "- **D** used an inner cross-fit EB prediction as a feature in the LambdaRank shortlist reranker.", "",
        "### Boundary expansion and diagnostics", "",
        "Each numerical expansion is tied to an incumbent, a previous range and the additional tested edge. "
        "Natural caps include unit-interval mixture weights, valid attention head divisors and positive penalties. "
        "Remaining numerical proposals, if any, appear in the table above. An empirical turnover applies "
        "to the tested configurations and split support.", "",
        "[Grid scores](H1V2_INNER_GRID.tsv) · [Boundary trace](H1V2_INNER_BOUNDARIES.tsv) · "
        "[Selected recipes](H1V2_INNER_WINNERS.tsv) · [Train and validation curves](H1V2_LEARNING_CURVES.tsv) · "
        "[Tissue, mechanism and TKI errors](H1V2_INNER_ERRORS.tsv) · [Budget and noise sensitivity](H1V2_INNER_BUDGET.tsv)", "",
    ]
    grid = load(root, "H1V2_INNER_GRID.tsv")
    lines += ["| Family | Boundary recipes | Targeted refinements | Refined recipe wins | Median winning utility gain over Round 2 |",
              "|---|---:|---:|---:|---:|"]
    for family in "ABCD":
        explored = [r for r in grid if r["family"] == family]
        refined = [r for r in explored if r["phase"] == "round3-error-refinement"]
        expanded = sum(r["phase"].startswith(("round3-expansion", "round3-closure", "round3-final-boundary")) for r in explored)
        gains = [num(r["objective"]) - num(r["round2"]) for r in winners if r["family"] == family]
        wins = sum(r["winner"].casefold() == "true" for r in refined)
        lines.append(f"| {family} | {expanded} | {len(refined)} | {wins} | {median(gains):+.4f} |")
    lines += ["", *boundary_summary(grid, winners, load(root, "H1V2_INNER_BOUNDARIES.tsv")), *learning_summary(root)]
    leaders = {
        (task, mode, r["family"]) for (task, mode), fits in grouped(winners, ("task", "modality")).items()
        for r in sorted(fits, key=lambda item: num(item["round2"]), reverse=True)[:2]
    }
    errors = [r | {"study": study(r["task"])} for r in load(root, "H1V2_INNER_ERRORS.tsv")
              if (r["task"], r["modality"], r["family"]) in leaders]
    lines += ["### Subgroup errors in the two leading Round 2 families", "",
              "| Study | Modality | Group type | Group | Summed validation entries | Mean fold MAE |",
              "|---|---|---|---|---:|---:|"]
    for (s, mode, typ), points in sorted(grouped(errors, ("study", "modality", "group_type")).items()):
        ranked = [(name, sum(int(r["n"]) for r in entries), average(entries, "mae"))
                  for (name,), entries in grouped(points, ("group",)).items()]
        if typ != "drug":
            ranked = sorted((r for r in ranked if r[1] >= 10), key=lambda r: (-r[2], r[0]))[:2]
        for name, n, error in sorted(ranked, key=lambda row: -row[2]):
            group_label = name.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {s} | {mode} | {typ} | {group_label} | {n} | {error:.4f} |")
    lines += ["", "The counts sum across family and fold evaluations. Each outcome can contribute to both families. "
              "The table lists the two highest-error supported tissue and mechanism groups per study and modality, "
              "plus all observed HCC TKIs. MAE uses the study-specific response scale.", ""]
    size = root / "H1V2_LICOB_SAMPLE_SIZE.tsv"
    if size.is_file():
        size_rows = load(root, size.name)
        lines += ["### LICOB reference sample size", "",
                  "| Modality | Fraction | Mean fit donors | Mean NDCG@5 | Mean within-drug Spearman |",
                  "|---|---:|---:|---:|---:|"]
        for (mode, fraction), values in sorted(grouped(size_rows,
                                                       ("modality", "training_fraction")).items()):
            f, n, rho = (mean(num(x[k]) for x in values) for k in ("fit_groups", "ndcg", "rho"))
            lines.append(f"| {mode} | {fraction} | {f:.1f} | {n:.4f} | {rho:.4f} |")
        lines += ["", "The two leading families retained their fixed inner recipes, anchor ordering and validation donors "
                  "while reference fit sizes varied. This estimates sample-size sensitivity conditional on the chosen strategy. "
                  "[Sample-size results](H1V2_LICOB_SAMPLE_SIZE.tsv)", ""]
        for (mode,), rows in sorted(grouped(size_rows, ("modality",)).items()):
            half, full = ([r for r in rows if num(r["training_fraction"]) == f] for f in (.5, 1.))
            lines.append(f"- **{mode}** mean fit size increased from {average(half, 'fit_groups'):.1f} to "
                         f"{average(full, 'fit_groups'):.1f} donors. The corresponding mean changes were "
                         f"{average(full, 'ndcg') - average(half, 'ndcg'):+.4f} NDCG@5 and "
                         f"{average(full, 'rho') - average(half, 'rho'):+.4f} within-drug Spearman.")
        lines.append("")
    (root / "ROUND1_TO_3_CEILING_TUNING.md").write_text("\n".join(lines))


def results(root):
    axes = {(r["study"], r["panel"], r["arm"], int(r["k"]), r["metric"]): r
            for r in load(root, "H1V2_METRICS.tsv")}
    paired = {(r["study"], r["panel"], r["candidate"], r["comparator"], r["metric"]): r
              for r in load(root, "H1V2_PAIRED_95CI.tsv")}
    score = json.loads((root / "score_receipt.json").read_text())
    replay = json.loads((root / "replay_receipt.json").read_text())
    inner_winners = load(root, "H1V2_INNER_WINNERS.tsv")
    # Paired tables retain the model and comparator budget in their identifiers.
    def p(s, candidate, comparator, metric=NDCG, k=5, panel="Q8", other=None):
        return paired.get((s, panel, f"{candidate}/{k}", f"{comparator}/{other if other is not None else k}", metric))

    def get(s, candidate, k=5, metric=NDCG, panel="Q8"):
        return axes.get((s, panel, candidate, k, metric))

    def field(row, name):
        return num(row.get(name)) if row else float("nan")

    def lower(s, x, b, metric=NDCG):
        return field(p(s, x, b, metric), "lo95")

    def delta(s, x, b, metric=NDCG):
        return field(p(s, x, b, metric), "delta")

    def gate_a(s):
        old = p(s, DUAL, "v1-RNA+Anchor")
        if old is None:
            return "비교군 지원 없음"
        cond = (delta(s, DUAL, "B1") >= .005 and delta(s, DUAL, "B1", RHO) >= .05
                and max(lower(s, DUAL, "B1"), lower(s, DUAL, "B1", RHO)) > 0
                and max(lower(s, DUAL, "v1-RNA+Anchor"), lower(s, DUAL, "v1-RNA+Anchor", RHO)) > 0
                and lower(s, DUAL, "B1", PRECISION) >= -.005
                and lower(s, DUAL, "N1-Dual") > 0 and lower(s, DUAL, "N2") > 0)
        return "GO-DEV" if cond else "NO-GO"

    def gate_b(s, model):
        rank = delta(s, model, "Bsimple") >= .015 and lower(s, model, "Bsimple") > 0
        if get(s, "B1") is None:
            return ("통과" if rank else "미통과", "RNA 기준선 지원 없음")
        strat = all(delta(s, model, b, RHO) >= .05 and lower(s, model, b, INTER) > 0
                    and lower(s, model, b) >= -.01 for b in ("B1", "B2", "B3"))
        return ("통과" if rank else "미통과", "통과" if strat else "미통과")

    a = {s: gate_a(s) for s in ("Tier1-macro", "LIMORE")}
    b_loc = gate_b("LICOB", ANCHOR)
    b_transfer = gate_b("LIMORE_to_LICOB", ANCHOR)
    remaining = sum(max(0, int(r["remaining"])) for r in inner_winners)
    top = interval(p("Tier1-macro", DUAL, "B1"), "delta")
    three_d = interval(p("LICOB", ANCHOR, "Bsimple"), "delta")
    lines = [
        "**가설** — 지도 앵커와 순위 손실로 범암 RNA+Anchor 및 LICOB 환자 층화 성능을 높인다.",
        "**달라진 점** — v1의 분할과 Q8에서 네 모델 패밀리를 평가하고 K=1과 별도 Q12를 추가했다.",
        f"**결과** — K=5 ΔNDCG@5는 범암 RNA-only 대비 {top}, LICOB 최강 단순 기준선 대비 {three_d}였다.",
        f"**판정** — Track A 범암 {a['Tier1-macro']} / LIMORE {a['LIMORE']}. Track B 현지 순위 {b_loc[0]}·층화 {b_loc[1]} / 전이 순위 {b_transfer[0]}.",
        f"**다음 결정** — 통과한 축의 개발 근거를 유지한다. 수치 경계 후속 검증 항목은 {remaining}개다.", "",
        "# H1-v2 앵커 적응 성능", "",
        f"봉인 Outer는 `score_passes = {score['score_passes']}`로 채점했다. "
        f"구간은 {score['bootstrap_draws']:,}회 계통·공여자 단위 짝지은 재표집이다. "
        "모든 코호트는 기존 평가에 노출된 기술 평가다.", "",
        "## K=5 v1–v2 직접 비교", "",
        "| 연구 | 그룹 | Q | 팔 | NDCG@5 | 약물 내 Spearman | 상호작용 Spearman | Precision@5 |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for s in MAIN_STUDIES:
        for arm in ("Bsimple", "B1", "v1-Bsimple", "v1-Anchor-Best", "v1-RNA+Anchor", "v1-RNA+Anchor-ungated", ANCHOR, DUAL):
            first = get(s, arm)
            if first is None:
                continue
            r = [interval(get(s, arm, metric=metric)) for metric in (NDCG, RHO, INTER, PRECISION)]
            lines.append(f"| {s} | {first['independent_groups']} | {first['min_q']} | {arm} | {' | '.join(r)} |")
    lines += ["", "v1에 보조 모델이 채택된 연구의 v1-RNA+Anchor 값은 동결 예측에서 가져왔다. "
              "ungated 행은 당시 N2에 사용한 RNA 모델의 순열 전 참조로, v1에 공개된 보조 성적이 없는 연구에 표시했다.",
              "", "## 판정 근거", "",
              "| 연구 | 후속 모델 | 대조 | ΔNDCG@5 | Δ 약물 내 Spearman | Δ 상호작용 Spearman | ΔPrecision@5 |",
              "|---|---|---|---:|---:|---:|---:|"]
    for s in ("Tier1-macro", "LIMORE", "LICOB", "LIMORE_to_LICOB"):
        for model in ((DUAL,) if s in a else (ANCHOR, DUAL) if s == "LICOB" else (ANCHOR,)):
            for control in ("Bsimple", "B1", "B2", "B3", "v1-Anchor-Best", "v1-RNA+Anchor", "N1-Dual", "N1", "N2"):
                row = p(s, model, control)
                if row is None:
                    continue
                diffs = [interval(p(s, model, control, m), "delta") for m in (NDCG, RHO, INTER, PRECISION)]
                lines.append(f"| {s} | {model} | {control} | {' | '.join(diffs)} |")
    lines += ["", "Tier 4는 현지 공여자 0명·RNA 0의 전이다. 원 측정 척도가 다른 절대오차는 생략했다. "
              "전체 지표와 비교는 집계표에 있다. [지표](H1V2_METRICS.tsv) · [짝지은 차이](H1V2_PAIRED_95CI.tsv)",
              "", "## 모델 패밀리", "",
              "| 연구 | A NDCG / rho | B NDCG / rho | C NDCG / rho | D NDCG / rho |",
              "|---|---:|---:|---:|---:|"]
    for s in MAIN_STUDIES:
        vals = []
        for f in "ABCD":
            n, r = get(s, f"Family-{f}"), get(s, f"Family-{f}", metric=RHO)
            vals.append("—" if n is None else f"{fmt(n['value'])} / {fmt(r['value'])}")
        lines.append(f"| {s} | {' | '.join(vals)} |")
    lines += ["", "범암·LIMORE의 패밀리 대표는 RNA+Anchor, LICOB·Tier 4의 대표는 Anchor-only다. "
              "[패밀리 지표와 95% CI](H1V2_FAMILY_BENCHMARK.tsv)", "",
        "## K 예산 곡선", "", "각 셀은 NDCG@5 / 약물 내 Spearman이다.", "",
              "| 연구 / 모델 | K=0 | K=1 | K=3 | K=5 | K=8 | K=12 별도 Q12 | Q12 그룹 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in MAIN_STUDIES:
        for arm in (ANCHOR, DUAL):
            if get(s, arm) is None:
                continue
            vals = [f"{fmt(get(s, arm, k)['value'])} / {fmt(get(s, arm, k, RHO)['value'])}"
                    for k in (0, 1, 3, 5, 8)]
            aux = get(s, arm, 12, panel="Q12")
            vals.append(f"{fmt(aux['value'])} / {fmt(get(s, arm, 12, RHO, 'Q12')['value'])}" if aux else "—")
            lines.append(f"| {s} / {arm} | {' | '.join(vals)} | {aux['independent_groups'] if aux else '—'} |")
    tissues = load(root, "H1V2_PANCANCER_BY_TISSUE.tsv")
    lines += ["", f"범암 조직 표는 등록 라벨 {len(set(t['tissue'] for t in tissues))}개를 포함한다. "
              "조직별 9개 지표와 짝지은 신뢰구간을 함께 제공한다. [범암 조직별](H1V2_PANCANCER_BY_TISSUE.tsv)", "",
              "| 범암 패널의 Liver | 계통 | RNA+Anchor NDCG@5 | 최강 단순 기준선 대비 Δ |",
              "|---|---:|---:|---:|"]
    for row in tissues:
        if row['tissue'].casefold() == 'liver' and row['arm'] == DUAL and row['metric'] == NDCG:
            paired_liver = {"delta": row["delta_vs_bsimple"], "lo95": row["delta_lo95"], "hi95": row["delta_hi95"]}
            lines.append(f"| {row['study']} | {row['n_groups']} | {interval(row)} | {interval(paired_liver, 'delta')} |")
    lines += ["",
              "## 4대 HCC TKI", "",
              "| 연구 | 약물 | 팔 | 관측 수 | 약물 내 Spearman [95% CI] |",
              "|---|---|---|---:|---:|"]
    for row in load(root, "H1V2_HCC_TKI.tsv"):
        if row["arm"] not in ("", ANCHOR, DUAL):
            continue
        rho = interval({"value": row["spearman"], "lo95": row["spearman_lo95"], "hi95": row["spearman_hi95"]})
        lines.append(f"| {row['study']} | {row['tki']} | {row['arm'] or '지원 없음'} | {row['n']} | {rho} |")
    lines += ["", "Cabozantinib은 LICOB 등록 패널에 없다. 나머지 TKI의 분모는 폴드별 Q 포함 여부를 반영한다.",
              "", "## 96웰 비용과 민감도", "",
              "K=5에서 11농도 × 2반복 + 대조 12웰은 122웰이다. 8농도는 92웰, 7농도는 82웰, "
              "6농도는 72웰이다. 농도별 LICOB 실측이 등록 입력에 없으므로 농도별 절삭 성능 대신 "
              "Inner-fit 표준편차 0.10·0.25·0.50 배의 앵커 노이즈 감도를 계산했다. "
              "[비용 영수증](cost_receipt.json) · [노이즈 감도](H1V2_INNER_BUDGET.tsv)", "",
              "| LICOB Inner 모델 | K | 무노이즈 NDCG / rho | 0.10 SD의 Δ | 0.25 SD의 Δ | 0.50 SD의 Δ |",
              "|---|---:|---:|---:|---:|---:|"]
    budgets = load(root, "H1V2_INNER_BUDGET.tsv")
    loc_winners = {(task, mode): max(fits, key=lambda r: num(r["objective"]))["family"]
                   for (task, mode), fits in grouped(inner_winners, ("task", "modality")).items()
                   if study(task) == "LICOB"}
    chosen = [r for r in budgets if r["family"] == loc_winners.get((r["task"], r["modality"]))
              and r["k"] in {"3", "5", "8"}]
    for (mode, k), rows in sorted(grouped(chosen, ("modality", "k")).items()):
        noise = {spread: [r for r in rows if num(r["noise_sd"]) == spread] for spread in (0., .1, .25, .5)}
        n0, r0 = (average(noise[0.], m) for m in ("ndcg", "rho"))
        changes = [f"{average(noise[sd], 'ndcg') - n0:+.4f} / {average(noise[sd], 'rho') - r0:+.4f}"
                   for sd in (.1, .25, .5)]
        lines.append(f"| {mode} | {k} | {n0:.4f} / {r0:.4f} | {' | '.join(changes)} |")
    lines += ["", "## 오프라인 검증", "",
              f"오프라인 CPU CLI 최대 절대 차이 `{replay['max_abs_diff']:.4g}`. "
              f"리플레이 기준 `1e-6` {'통과' if replay['passed'] else '미통과'}. "
              "[CPU 리플레이](replay_receipt.json) · [모델 카드](model_card.md) · [채점 영수증](score_receipt.json)"]
    lines[:4] = [s + "<br>" for s in lines[:4]]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("--stage", choices=("inner", "results", "all"), default="all")
    args = cli.parse_args()
    if args.stage in ("inner", "all"):
        inner(args.root)
    if args.stage in ("results", "all"):
        results(args.root)
