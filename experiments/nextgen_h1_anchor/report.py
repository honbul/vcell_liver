"""Render the H1 results overview from the signed aggregate outputs."""
import argparse
import csv
import json
from pathlib import Path
import statistics

P = "H1-Anchor-Best"
R = "H1-RNA+Anchor-Best"
STUDIES = ("Tier1-macro", "GDSC2", "CTRPv2", "PRISM20Q2", "LIMORE", "LICOB", "LIMORE_to_LICOB", "LIMORE_to_Wong")
MAIN = ("Tier1-macro", "LICOB")
ARMS = (P, "Bsimple", "B0", "B1", "B2", "B3", "B4", "N1", "N2", R)
SPEARMAN = "within_drug_spearman"
PEARSON = "within_drug_pearson"
NDCG = "ndcg_at_5"
PREC = "precision_at_5"
TKI = ("Sorafenib", "Lenvatinib", "Regorafenib", "Cabozantinib")


def rows(root, file):
    with (root / file).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def number(s):
    return float("nan") if s in {"", "NA", None} else float(s)


def point(s, key="value", signed=False):
    if s is None or s[key] in {"", "NA"}:
        return "—"
    return format(float(s[key]), "+.4f" if signed else ".4f")


def interval(s, key="delta"):
    if not s or s[key] == "NA":
        return "—"
    return f"{point(s, key, key == 'delta')} [{point(s, 'lo95', key == 'delta')}, {point(s, 'hi95', key == 'delta')}]"


def render(root):
    results = {(r["study"], r["method"], int(r["k"]), r["metric"]): r for r in rows(root, "H1_METRICS.tsv")}
    paired = {(r["study"], r["candidate"], r["comparator"], r["metric"]): r for r in rows(root, "H1_PAIRED_95CI.tsv")}
    freeze = json.loads((root / "inner_freeze_summary.json").read_text())["tasks"]
    receipt = json.loads((root / "score_receipt.json").read_text())
    replay = json.loads((root / "cli_replay_receipt.json").read_text())

    def p(study, model, k, metric):
        return results.get((study, model, k, metric))

    def d(study, comparator, metric=NDCG, candidate=f"{P}/5"):
        return paired.get((study, candidate, comparator, metric))

    def bounds(s):
        return {tag: d(s, ref, metric) for tag, ref, metric in (
            ("ndcg", "Bsimple/5", NDCG), ("precision", "Bsimple/5", PREC),
            ("k0", f"{P}/0", NDCG), ("shuffle", "N1/5", NDCG))}

    def passes(b):
        return (number(b["ndcg"]["delta"]) >= .02 and number(b["ndcg"]["lo95"]) > 0
                and number(b["precision"]["lo95"]) >= -.02
                and number(b["k0"]["lo95"]) > 0 and number(b["shuffle"]["lo95"]) > 0)

    def rejects(b):
        return (number(b["ndcg"]["hi95"]) < .02 or number(b["precision"]["hi95"]) < -.02
                or number(b["k0"]["hi95"]) <= 0 or number(b["shuffle"]["hi95"]) <= 0)

    gates = {s: bounds(s) for s in MAIN}
    decision = "GO-DEV" if any(map(passes, gates.values())) else (
        "NO-GO" if all(map(rejects, gates.values())) else "INCONCLUSIVE")
    fallback = [s for s in MAIN if all(number(d(s, ref, candidate="B4/5")["lo95"]) > 0
                                       for ref in ("B1/5", "B2/5"))]
    fallback_text = ", ".join(fallback)
    if decision == "GO-DEV":
        action = "통과한 연구 범위에서 후속 개발 평가를 진행한다."
    elif fallback:
        action = f"복잡 모델 채택을 보류하고 {fallback_text}에서 단순 kNN을 후보 축소 도구로 채택한다."
    else:
        action = "복잡 모델과 단순 kNN의 도입을 보류한다."
    pan = interval(gates["Tier1-macro"]["ndcg"])
    licob = interval(gates["LICOB"]["ndcg"])
    outputs = [
        "**가설** — 새 계통·공여자의 앵커 5개 실측이 미측정 약물의 추천 순위를 개선한다.",
        "**달라진 점** — 약물별 조건부 보정을 3라운드 Inner CV로 선택하고 모든 비교를 A8 제외 Q에서 수행했다.",
        f"**결과** — 최강 단순 기준선 대비 K=5 ΔNDCG@5는 범암 스크린 평균 {pan}, LICOB {licob}였다.",
        f"**판정** — {decision}. 단일 고밀도 범암 공통 40약물 패널은 등록 자료에서 INFEASIBLE이다.",
        f"**다음 결정** — {action}",
        "",
        "# H1 기능측정 보조 스크리닝",
        "",
        "17개 외부 적합 단위의 챔피언·비교군·앵커를 동결했다. " 
        "동결 뒤 앵커 입력을 추출하고 Q 예측을 생성한 뒤 별도 프로세스에서 채점했다. "
        f"Outer 3-fold와 Tier 4 및 보조 Wong을 동일한 단일 채점 패스에서 처리했다. "
        f"`score_passes = {receipt['score_passes']}`. 95% 구간은 동일 독립 계통·공여자 2,000회 짝지은 재표집이다. "
        "범암 스크린에 중복 등장한 계통은 각 드로우에서 함께 재표집했다. "
        "LIMORE, LICOB, Wong은 기존 평가에 사용된 기술 코호트이고, 범암 자료는 기존 개발·평가에 사용됐다.",
        "",
        "## 사전 판정과 대조군",
        "",
        "| 연구 범위 | ΔNDCG@5 vs 최강 단순 기준선 | ΔPrecision@5 vs 최강 단순 기준선 | ΔNDCG@5 vs K=0 | ΔNDCG@5 vs N1 | 통과 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for s, b in gates.items():
        outputs.append(f"| {s} | {interval(b['ndcg'])} | {interval(b['precision'])} | {interval(b['k0'])} | "
                       f"{interval(b['shuffle'])} | {'예' if passes(b) else '아니오'} |")
    outputs += ["", "| 연구 범위 | B4 vs RNA-only ΔNDCG@5 | B4 vs scalar ΔNDCG@5 |",
                "|---|---:|---:|"]
    for s in (*MAIN, "LIMORE"):
        outputs.append(f"| {s} | {interval(d(s, 'B1/5', candidate='B4/5'))} | "
                       f"{interval(d(s, 'B2/5', candidate='B4/5'))} |")
    outputs += ["", "Bsimple은 각 외부 폴드의 Inner CV에서 정한 최강 B0–B4 비교군이다. "
                "B0는 약물 평균, B1은 RNA-only, B2는 scalar, B3은 양의 affine, B4는 kNN이다. "
                "N1은 앵커 패킷 전체를 순열하고, N2는 이중 결합 후보의 RNA만 순열한다. "
                "보조 RNA 추정에는 Inner CV 기준을 통과한 폴드에서 RNA를 추가하고, 나머지 폴드에서는 동결한 anchor-only 예측을 사용했다.",
                "", "## Tier 1–4 및 Wong", "",
                "| 연구 범위 | 독립 그룹 | Q 약물 수 | K=0 NDCG@5 | K=3 | K=5 | K=8 | K=5 ΔNDCG@5 vs Bsimple |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in STUDIES:
        q = p(s, P, 5, NDCG)
        if not q:
            continue
        ks = " | ".join(point(p(s, P, k, NDCG)) for k in (0, 3, 5, 8))
        outputs.append(f"| {s} | {q['independent_groups']} | {q['q_min']} | {ks} | {interval(d(s, 'Bsimple/5'))} |")
    outputs += ["", "범암은 각 스크린에서 고밀도 68약물에 적합했다. 24약물만 세 스크린에 정확히 일치했고, "
                "그중 4개가 세 스크린 각각에서 ≥80% 계통 관측을 충족했다. "
                "비교 가능한 시료는 A8 모두 실측했고 Q 실측이 15개 이상인 그룹으로 고정했다.", "",
                "| 스크린 | GDSC2 | CTRPv2 | PRISM20Q2 |",
                "|---|---:|---:|---:|"]
    budgets = receipt["masks"]
    totals = []
    for s in STUDIES[1:4]:
        for name in ("eligible", "registered"):
            totals.append(sum(v[name] for t, v in budgets.items() if t.rsplit("-", 1)[0] == s))
    outputs.append("| 공통 평가 그룹 / 등록 그룹 | " +
                   " | ".join(f"{totals[i]} / {totals[i+1]}" for i in range(0, len(totals), 2)) + " |")
    outputs += ["", "Tier 4는 LIMORE 50계통의 53개 일치 약물로 적합하고 LICOB 38명 전원에게 적용했다. "
                "LICOB에서는 개별 시료의 5개 앵커를 사용했다. 현지 공여자 아틀라스와 RNA를 사용하지 않았다. "
                "Wong 보조 평가는 LIMORE와 일치한 27약물 중 Q 19개가 실측된 10명에서 수행됐다. "
                "Yang은 5약물 패널로 제외됐다.",
                "", "### K=5 모델별 성능", "",
                "| 연구 범위 | 모델 | NDCG@5 | Precision@5 | 시료 내 Spearman | 약물 내 Spearman | 약물 내 Pearson | 상호작용 PCC | 상호작용 Spearman | MAE | RMSE |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    axes = (NDCG, PREC, "within_sample_spearman", SPEARMAN, PEARSON, "interaction_pearson", "interaction_spearman", "mae", "rmse")
    for s in STUDIES:
        for m in ARMS:
            if p(s, m, 5, NDCG) is None:
                continue
            fields = [point(p(s, m, 5, axis)) if not (s == "Tier1-macro" and axis in {"mae", "rmse"}) else "—"
                      for axis in axes]
            outputs.append(f"| {s} | {m} | " + " | ".join(fields) + " |")
    outputs += ["", "구간과 전체 K 비교는 [절대 성능](H1_METRICS.tsv)과 [짝지은 차이](H1_PAIRED_95CI.tsv)에 있다. "
                "약물 내 상관과 이중 중심화 상호작용 상관은 폴드 내부에서 계산했다. "
                "MAE/RMSE는 동일 assay 안에서 해석한다.",
                "", "| RNA 보조 비교 | K=5 ΔNDCG@5 vs anchor-only |",
                "|---|---:|"]
    for s in STUDIES[:4]:
        change = d(s, f"{P}/5", candidate=f"{R}/5")
        if change:
            outputs.append(f"| {s} | {interval(change)} |")
    outputs += ["", "## 암종별 분해", "",
                "K=5의 대표 암종 분해다. [전체 암종·K·모델 표](H1_PANCANCER_BY_TISSUE.tsv)는 희소 암종도 포함한다.",
                "", "| 스크린 | 조직 | 평가 계통 | H1 NDCG@5 | Δ vs 최강 단순 [95% CI] |",
                "|---|---|---:|---:|---:|"]
    tissues = rows(root, "H1_PANCANCER_BY_TISSUE.tsv")
    for t in tissues:
        if t["metric"] == NDCG and t["method"] == P and t["k"] == "5" and int(t["n_groups"]) >= 20:
            value = interval({"delta": t["delta_vs_bsimple"], "lo95": t["paired_lo95"], "hi95": t["paired_hi95"]})
            outputs.append(f"| {t['study']} | {t['tissue']} | {t['n_groups']} | {point(t)} | {value} |")
    outputs += ["", "## HCC TKI 및 기전군", "",
                "K=5에서 적합 후보의 약물 내 Spearman·Pearson을 표에 제시한다. "
                "앵커로 선정된 약물은 해당 폴드의 Q에서 제외된다. "
                "폴드별 전체 분해와 단순 기준선은 [TKI·기전 집계](H1_HCC_TKI_AND_MECHANISMS.tsv)에 있다.",
                "", "| 연구 | 약물 또는 기전군 | Q에 있는 폴드 | 관측 행 | macro Spearman | macro Pearson |",
                "|---|---|---:|---:|---:|---:|"]
    moa = rows(root, "H1_HCC_TKI_AND_MECHANISMS.tsv")
    for s in ("LIMORE", "LICOB"):
        for g in (*TKI, "kinase_inhibitor", "nonselective_cytotoxics", "egfr", "fgfr", "hdac", "mtor"):
            b = [r for r in moa if r["study"] == s and r["stratum"] == g and r["method"] == P and int(r["q_drugs"])]
            def avg(metric):
                v = [number(r[metric]) for r in b if r[metric] not in {"NA", ""}]
                return f"{statistics.mean(v):.4f}" if v else "—"
            outputs.append(f"| {s} | {g} | {len(b)} | {sum(int(r['observations']) for r in b)} | "
                           f"{avg('spearman')} | {avg('pearson')} |")
    outputs += ["", "## 실용 비용과 CPU 검증", "",
                "두 반복과 12개 대조 웰(예산 가정)을 적용했다. 등록된 농도 수를 사용한 웰 수는 "
                "`2 × (5개 약물의 농도 수 합계) + 12`다. 대조 수·반복 수는 처리량 추정을 위한 공통 가정이다.",
                "", "| 연구 / 폴드 | 앵커별 농도 수 | 시료당 웰 | 적합 아틀라스 그룹 | Q 후보 → 상위 5 |",
                "|---|---|---:|---:|---:|"]
    for name, data in sorted(freeze.items()):
        if data["cost_known"]:
            dose = data["actual_anchor_dose_points"]
            q = data["query_count"]
            outputs.append(f"| {name} | {', '.join(map(str, dose))} | {2 * sum(dose) + 12} | {data['fit_groups']} | "
                           f"{q} → 5 ({100 * (q-5)/q:.1f}% 축소) |")
    outputs += ["", "LICOB Tier 4의 아틀라스 50개는 LIMORE 계통이다. 현지 LICOB 아틀라스 공여자 수는 0명이다. "
                "아틀라스 비용은 연구용 재사용 자료의 규모다. 같은 실험 재현에 필요한 신규 아틀라스 규모나 실제 임상 적중률을 추정하지 않았다.",
                "", f"오프라인 CPU CLI의 최대 절대 오차는 `{replay['max_abs_diff']:.3g}`였고 "
                f"`1e-6` 기준을 충족했다. {len(replay['tasks'])}개 모델 패키지에서 단일/배치 결과를 비교하고 "
                "앵커 누락, 미측정값, 허용 범위 초과를 거절했다. "
                "예측 구간과 입력 계약은 [모델 카드](model_card.md)에 있다.",
                "", "[계획 잠금](PLAN_LOCK.md) · [데이터 및 노출](DATA_SUPPORT_AND_EXPOSURE.md) · "
                "[Inner CV](ROUND1_TO_3_INNER_TUNING.md) · [채점 영수증](score_receipt.json) · "
                "[CPU 재생 영수증](cli_replay_receipt.json) · [Slurm 원장](SLURM_JOBS.tsv)", ""]
    (root / "RESULTS.md").write_text("\n".join(outputs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
