"""Render the H3 reports using sealed summaries and score outputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ("round1", "round2-block", "round2-opt", "round2-head")
SCREENS = ("CTRPv2", "GDSC2", "PRISM20Q2", "macro")
METRICS = ("ndcg_at_5", "within_drug_spearman", "precision_at_k", "residual_spearman")
TITLES = ("NDCG@5", "약물 내 Spearman", "Precision@5", "잔차 Spearman")


def load(path):
    return json.loads(path.read_text())


def write(dst, name, lines):
    (dst / name).write_text("\n".join(lines).rstrip() + "\n")


def table(cols, rows):
    return ["| " + " | ".join(map(str, cols)) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def signed(n):
    return f"{n:+.5f}"


def values(entry):
    return [f"{entry[k]:.5f}" for k in METRICS]


def pull_summary(src, dst):
    books, all_metrics, all_runs, epochs = {}, [], [], []
    replay = load(src / "inner-round1-additive-replay.json")["comparisons"]
    for stage in STAGES:
        books[stage] = load(src / f"{stage}-summary.json")
        if stage == "round1":
            checks = [v for key, v in replay.items() if key.startswith("round1/")]
            if len(checks) != 9 or any(v["non_drug_mean_max_delta"] > 1e-9 for v in checks):
                raise ValueError("Missing correction invariance for the inner additive reference")
            books[stage]["scores"]["drug_mean"]["residual_spearman"] = 0.0
        for suffix, bucket in (("metrics", all_metrics), ("runs", all_runs), ("gradient-by-epoch", epochs)):
            path = src / f"{stage}-{suffix}.tsv"
            if path.exists():
                frame = pd.read_csv(path, sep="\t")
                if stage == "round1" and suffix == "metrics":
                    frame.loc[frame.arm.eq("drug_mean"), "residual_spearman"] = 0.0
                frame.insert(0, "stage", stage)
                bucket.append(frame)
    for name, parts in (("INNER_TUNING", all_metrics), ("INNER_RUNTIME", all_runs), ("GRADIENT_BY_EPOCH", epochs)):
        pd.concat(parts).to_csv(dst / f"H3_{name}.tsv", sep="\t", index=False)
    receipts = {
        stage: {
            "summary_sha256": hashlib.sha256((src / f"{stage}-summary.json").read_bytes()).hexdigest(),
            "input_forecast_and_gradient_sha256": book["inputs_sha256"],
            "settings_by_split": book["settings_by_split"],
        }
        for stage, book in books.items()
    }
    (dst / "inner_analysis_receipts.json").write_text(json.dumps(receipts, sort_keys=True, indent=2) + "\n")
    return books, pd.concat(all_runs)


def gradient(dst, src, books, runs):
    audit = pd.read_csv(src / "round1-gradient-summary.tsv", sep="\t")
    block = pd.read_csv(src / "round2-block-gradient-summary.tsv", sep="\t")
    stat = books["round1"]["diagnostics"]
    cols = ["arm", "mean_valid_pairs", "mean_density", "mean_valid_drug_groups",
            "mean_valid_cell_groups", "mean_eligible_cell_share"]
    support = pd.concat([audit.loc[audit.arm.eq("G1_w0"), cols], block[cols]])
    support = support.groupby("arm", as_index=False).mean(numeric_only=True)
    timing = runs.loc[runs.stage.eq("round1") & runs.diagnostics].groupby("name").wall_seconds.agg(["median", "sum"])
    lite = runs.loc[runs.timing_replay.eq(True)]
    lines = [
        "# 배치별 기울기 진단", "",
        f"A40에서 18개 inner fold × 8개 arm × 384 update를 추적했다. 목적 쌍 {stat['all_pair_batches']:,}건 중 "
        f"두 norm이 1e−8을 넘은 경우는 {stat['valid_pair_batches']:,}건이다. "
        f"G1의 음의 cosine 비율은 {stat['g1_negative_cosine_frequency']:.2%}다. "
        "아래 수치는 투영 전 공유 인코더의 기울기이며, 가중 norm 비율에는 목적 가중치를 곱했다.", "",
        *table(["Arm", "목적 A", "목적 B", "중앙 cosine", "중앙 norm A/B", "가중 A/B", "충돌", "유효 update"],
               [(r.arm, r.loss_a, r.loss_b, f"{r.median_cosine:+.4f}", f"{r.median_norm_ratio:.2f}",
                 f"{r.median_weighted_ratio:.2f}", f"{r.conflict_frequency:.2%}", f"{r.batches:,}")
                for r in audit.itertuples()]),
        "## 배치 지지", "",
        "밀도는 관측 쌍을 방문 세포×약물 슬롯으로 나눈 값이다. Pearson은 약물당 유효 관측 8개와 "
        "비상수 실측, ListNet은 세포당 관측 2개를 요구한다.", "",
        *table(["구성", "평균 관측", "밀도", "Pearson 그룹", "ListNet 그룹", "방문 세포 중 ListNet"],
               [(r.arm, f"{r.mean_valid_pairs:.1f}", f"{r.mean_density:.2%}", f"{r.mean_valid_drug_groups:.1f}",
                 f"{r.mean_valid_cell_groups:.1f}", f"{r.mean_eligible_cell_share:.2%}") for r in support.itertuples()]),
        "## 실행 시간", "",
        "감사 실행은 모든 arm에서 목적별 기울기를 계산한다. GPU-hour는 A40 적합의 동기화된 실측 "
        "wall-time이다. 감사 없는 재실행은 한 fold에서 같은 update 수로 측정했다.", "",
        *table(["감사 arm", "중앙 초 / 적합", "총 GPU-hour"],
               [(key, f"{r['median']:.2f}", f"{r['sum'] / 3600:.4f}") for key, r in timing.iterrows()]),
        *table(["감사 없는 arm", "초 / 적합", "GPU-hour", "감사 예측과 최대 차이"],
               [(r.name, f"{r.wall_seconds:.3f}", f"{r.gpu_hours:.5f}", f"{r.max_forecast_difference:.3g}")
                for r in lite.itertuples()]),
        "G1 직접 역전파와 목적별 합의 수치 차이를 별도로 기록했다. 최종 H3a 적합은 G1과 G2에 "
        "동일한 목적별 역전파를 썼다. [에포크 요약](H3_GRADIENT_BY_EPOCH.tsv) · [실행 원장](H3_INNER_RUNTIME.tsv)", "",
        "원시 배치별 epoch/update, cosine, 양쪽 norm, 두 norm 비율, 충돌 및 CUDA 이벤트 시간은 G1 H3 루트의 "
        "round1/<split>/gradients-inner*.parquet와 round2-block/<split>/gradients-inner*.parquet에 있다. "
        "배치 파일 해시는 inner 분석 영수증에 기록했다. [Inner 분석 영수증](inner_analysis_receipts.json) · [동결과 결과](ROUND1_TO_3_INNER_TUNING.md)",
    ]
    write(dst, "GRADIENT_DIAGNOSTICS.md", lines)


def inner(dst, books, frozen):
    a, b = frozen["a"], frozen["b"]
    r1 = books["round1"]["scores"]
    lines = [
        "# Round 1–3 Inner CV", "", "## H3a · Round 1", "",
        "9개 outer 분할마다 2개 inner fold를 두었다. 고정 가중치 실험의 기술적 분할 평균을 제시한다. "
        "각 outer 분할의 설정은 그 분할의 두 inner fold에서만 선택했다. 약물 평균 잔차는 수치적 사영 오차를 "
        "제거한 0이다. 후보와 구조 커널의 지표는 전부 1e−9 이내에서 재현됐다. [정밀도 재현](inner_additive_replay.json)", "",
        *table(["Arm", *TITLES], [(name, *values(row)) for name, row in r1.items()]),
        *table(["동일 가중치", "ΔNDCG@5", "Δ약물 내 ρ", "ΔPrecision@5", "Δ잔차 ρ"], [
            (f"w{w}", *(signed(r1[f"G2_w{w}"][k] - r1[f"G1_w{w}"][k]) for k in METRICS)) for w in range(3)]),
        *table(["Outer 분할", "최강 G1", "최강 G2", "동일 가중치 G1", "파일럿 중단"],
               [(name, row["strongest_g1"], row["strongest_g2"], row["matched_g1"], row["pilot_stop"])
                for name, row in a["fold_choices"].items()]),
        "## H3b · Round 2", "",
        "두 블록의 G1과 G2는 같은 분할의 Round 1 가중치로 비교했다. 최적화기와 분해 단계는 해당 분할에서 "
        "선택된 G1 블록을 사용했다. 데이터 노출과 update 수는 고정했다. adaptive는 각 분할에서 선택된 "
        "Normalized PCGrad, GradNorm 또는 CAGrad다.", "",
    ]
    for stage in STAGES[1:]:
        lines.extend([f"### {stage}", "", *table(["구성", *TITLES], [
            (name, *values(row)) for name, row in books[stage]["scores"].items()])])
    rows = []
    for split, row in b["fold_choices"].items():
        best = row["best_by_ndcg"]
        gains = row["candidates"][best]["inner_gains"]
        rows.append((split, books["round2-opt"]["settings_by_split"][split]["O_normalized"]["batch"],
                     books["round2-head"]["settings_by_split"][split]["D_shared_adaptive"]["method"], best,
                     signed(gains[row["inner_comparators"][0]]["ndcg_at_5"]),
                     signed(gains["structured"]["ndcg_at_5"]), row["champion"] or "없음"))
    lines.extend(table(["분할", "블록", "적응 최적화", "최고 NDCG 구성", "ΔNDCG vs G1",
                        "ΔNDCG vs 구조", "통과 구성"], rows))
    lines.extend([
        f"H3b의 완전 지지 동결 기준 충족은 {b['qualified_folds']}/9 분할이다. "
        "네 지표의 모든 비교와 설정·입력 해시는 동결 원장에 있다.", "", "## Round 3", "",
        "동결 조건을 11711, 11729, 11743에서 재적합했다. H3a의 G2, G1 matched, G1 best를 각각 보존했다. "
        "[Inner 점수 TSV](H3_INNER_TUNING.tsv) · [최종 잠금](final_lock.json) · [Outer 결과](RESULTS.md)", "",
        "블록 단계 첫 집계는 진단 대조군 이름을 잘못 조회해 설정 동결 전에 중단됐다. 집계만 재실행했고 기존 9개 "
        "블록 학습 예측과 이미 저장된 네 TSV를 해시와 바이트로 대조했다. [복구 및 실행 기록](SLURM_JOBS.tsv)",
    ])
    write(dst, "ROUND1_TO_3_INNER_TUNING.md", lines)


def results(dst, frozen, receipt):
    metrics = pd.read_csv(dst / "H3_METRICS.tsv", sep="\t")
    deltas = pd.read_csv(dst / "H3_PAIRED_95CI.tsv", sep="\t")
    a, b = receipt["decisions"]["H3a"], receipt["decisions"]["H3b"]
    macro = deltas.loc[deltas.screen.eq("macro")]
    matched = macro.loc[macro.baseline.eq("G1_matched") & macro.metric.eq("ndcg_at_5")]
    a_summary = "파일럿 종료 조건을 충족했다."
    if not matched.empty:
        r = matched.iloc[0]
        a_summary = f"동일 가중치 G1 대비 ΔNDCG@5 {signed(r.delta)} [{signed(r.lower)}, {signed(r.upper)}]."
    b_summary = f"동결 기준을 충족한 H3b inner 분할 {frozen['b']['qualified_folds']}/9."
    if b["stage"] == "outer":
        r = macro.loc[macro.hypothesis.eq("H3b") & macro.baseline.eq("G1_best")
                      & macro.metric.eq("ndcg_at_5")].iloc[0]
        b_summary = f"H3b 최강 G1 대비 ΔNDCG@5 {signed(r.delta)} [{signed(r.lower)}, {signed(r.upper)}]."
    promoted = [key for key, v in receipt["decisions"].items() if v["verdict"] == "GO-DEV"]
    next_step = "GO-DEV 통과 후보 0개. 확장 패널, 암종별 분해, LIMORE/LICOB 기술 확인은 미실행."
    if promoted:
        next_step = "GO-DEV 후보 " + ", ".join(promoted) + "의 [조건부 검증](TECHNICAL_CONFIRMATION.md)."
    lines = [
        "- **가설** H3a 순수 PCGrad의 두 축 개선과 H3b 배치·최적화·분해의 개선을 각각 평가했다.",
        "- **달라진 점** H3a는 기울기 투영만 적용했고 H3b는 같은 입력과 관측 예산에서 세 요소를 비교했다.",
        f"- **결과** {a_summary} {b_summary}",
        f"- **판정** H3a **{a['verdict']}** ({a['stage']}) · H3b **{b['verdict']}** ({b['stage']}).",
        f"- **다음 결정** {next_step}", "", "# H3 2축 기울기 실험 결과", "",
        f"서로 다른 {receipt['global_lineages']}개 전역 계통의 {receipt['native_observations']:,}개 실측 행을 "
        f"동결 후 한 작업에서 채점했다. score_passes = {receipt['score_passes']}. 스크린 내 계통별 지표는 계통을 균등 "
        "평균하고 최종 점수는 세 스크린을 균등 평균했다. 3개 seed의 성적 평균을 보고한다. "
        "95% 구간의 재표집 단위는 세 스크린에 공유되는 전역 계통이다. [채점 영수증](score_receipt.json)", "",
    ]
    for hypothesis, verdict in (("H3a", a), ("H3b", b)):
        lines.extend([f"## {hypothesis} · {verdict['verdict']}", ""])
        if verdict["stage"] == "inner":
            lines.extend([b_summary if hypothesis == "H3b" else a_summary,
                          "조건에 따라 해당 후보의 outer 적합과 채점을 생략했다. [Inner 수치](ROUND1_TO_3_INNER_TUNING.md)", ""])
            continue
        main = deltas.loc[deltas.hypothesis.eq(hypothesis)]
        for baseline, pieces in main.groupby("baseline", sort=False):
            purpose = "등록 판정" if pieces.required.iloc[0] else "기술 대조"
            rows = []
            for screen in SCREENS:
                sub = pieces.loc[pieces.screen.eq(screen)].set_index("metric")
                display = []
                for key in METRICS[:3]:
                    m = sub.loc[key]
                    display.append(f"{signed(m.delta)} [{signed(m.lower)}, {signed(m.upper)}]")
                rows.append((screen, *display))
            lines.extend([f"**{pieces.candidate.iloc[0]} − {baseline}** · {purpose}", "",
                          *table(["분할", *(f"Δ{m} [95% CI]" for m in TITLES[:3])], rows)])
        req = [row for row in receipt["gates"] if row["required"] and row["hypothesis"] == hypothesis]
        keys = ("ndcg_gain", "ndcg_ci_positive", "drug_spearman_noninferior", "precision_noninferior",
                "screens_positive", "no_severe_loss")
        lines.extend(table(["대조", "ΔNDCG≥.01", "NDCG CI>0", "감수성 보호", "Precision 보호",
                            "2개 스크린 개선", "심한 열세 없음"], [
            (r["baseline"], *("통과" if r["criteria"][key] else "미달" for key in keys)) for r in req]))
    lines.extend([
        "## 세 스크린의 단일 모델 성적", "",
        "NDCG 뒤 ±는 세 seed의 표본 SD다. 정정 구조 커널, Ridge와 약물 평균은 같은 행의 기존 동결 예측을 썼다. "
        "모든 seed의 원 수치, Pearson, 계통 내 Spearman은 TSV에 있다.", "",
    ])
    avg = metrics.loc[metrics.seed.isin(["mean", "fixed"])]
    scores = avg.pivot(index=["screen", "model"], columns="metric", values="value")
    spreads = avg.set_index(["screen", "model", "metric"]).seed_sd
    rows = []
    for screen in SCREENS:
        for model in avg.model.unique():
            vals = values(scores.loc[screen, model])
            sd = spreads.loc[screen, model, "ndcg_at_5"]
            if np.isfinite(sd):
                vals[0] += f" ± {sd:.5f}"
            rows.append((screen, model, *vals))
    lines.extend(table(["스크린", "모델", *TITLES], rows))
    lines.extend([
        "## 평가 범위", "", next_step, "",
        "[전체 지표 TSV](H3_METRICS.tsv) · [전역 계통 짝지은 구간](H3_PAIRED_95CI.tsv) · "
        "[암종별 상태](H3_PANCANCER_BY_TISSUE.tsv) · [모델 카드](model_card.md) · [Slurm 작업](SLURM_JOBS.tsv)",
    ])
    write(dst, "RESULTS.md", lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    locked = load(args.reports / "final_lock.json")
    receipt = load(args.reports / "score_receipt.json")
    books, runs = pull_summary(args.analysis, args.reports)
    gradient(args.reports, args.analysis, books, runs)
    inner(args.reports, books, locked)
    results(args.reports, locked, receipt)


if __name__ == "__main__":
    main()
