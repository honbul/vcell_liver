"""Render the committed result tables as Markdown blocks (reports/.../tables/RESULT_TABLES.md) for the report."""
import json, os
import numpy as np
import pandas as pd

REP = os.path.join(os.environ.get("TB_REP_DIR", "reports/transigen_adaptation_b1"), "tables")
NAMES = {"B0": "B0 범용 그대로", "B1_scratch": "B1 HCC-only (처음부터)", "B2_finetune": "B2 미세조정", "no_change": "무변화 (Δ=0)",
         "target_adapt_mean": "target 적응 약물 평균", "source_same_drug_mean": "source 같은 약물 평균", "source_ridge": "source Ridge",
         "target_ridge": "target Ridge", "b0_mean_offset": "B0 + mean-offset", "b0_mean_offset_shrunk": "B0 + mean-offset (축소 계수 val 선택)"}
ORDER = ["B0", "B1_scratch", "B2_finetune", "no_change", "target_adapt_mean", "source_same_drug_mean", "source_ridge", "target_ridge",
         "b0_mean_offset", "b0_mean_offset_shrunk"]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def cell(s, lo, hi, d=3):
    return f"{s:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main():
    t = lambda n: pd.read_csv(os.path.join(REP, n + ".tsv"), sep="\t")
    prim, sec, con, var, eq, ens, mag, cur = (t(n) for n in ["primary_delta_rmse", "secondary_metrics", "contrasts", "seed_split_variation",
                                                             "equal_target_mean", "ensemble_scores_separate", "magnitude_and_retrieval", "learning_curve_k"])
    targets = sorted(prim.target.unique())
    blocks = []
    models = [m for m in ORDER if m in set(prim.model)] + sorted(m for m in set(prim.model) if m not in ORDER)
    rows = []
    for m in models:
        r = {"모델": NAMES.get(m, m)}
        for tg in targets:
            x = prim[(prim.model == m) & (prim.target == tg)]
            r[tg] = cell(x.score.iloc[0], x.ci_lo.iloc[0], x.ci_hi.iloc[0]) if len(x) else "—"
        e = eq[eq.model == m]
        r["세 target 동등 평균"] = cell(e.score.iloc[0], e.ci_lo.iloc[0], e.ci_hi.iloc[0]) if len(e) else "—"
        rows.append(r)
    blocks.append("### 표 1. 주지표: 시험 약물의 ΔRMSE (seed 평균 → 분할 평균, [약물 그룹 bootstrap 95% 구간])\n\n" + md(pd.DataFrame(rows)))
    c = con[con.metric == "delta_rmse"]
    rows = []
    for name in list(dict.fromkeys(c.contrast)):
        r = {"대비 (음수 = 앞의 모델이 더 좋음)": name.replace("B2_finetune", "B2").replace("B1_scratch", "B1")}
        for tg in targets:
            x = c[(c.contrast == name) & (c.target == tg)]
            r[tg] = cell(x["diff"].iloc[0], x.ci_lo.iloc[0], x.ci_hi.iloc[0]) if len(x) else "—"
        e = eq[eq.model == name]
        r["세 target 동등 평균"] = cell(e.score.iloc[0], e.ci_lo.iloc[0], e.ci_hi.iloc[0]) if len(e) else "—"
        rows.append(r)
    blocks.append("### 표 2. ΔRMSE 대비 (같은 약물 그룹 재표집으로 짝지은 차이)\n\n" + md(pd.DataFrame(rows)))
    for metric, title in [("delta_pearson", "조건별 ΔPearson"), ("centred_pearson", "공통 평균 반응을 뺀 조건별 Pearson"),
                          ("top20_direction", "실측 변화 상위 20개 유전자의 방향 일치율"), ("top20_recall", "상위 20개 회복률"),
                          ("retrieval_mrr", "retrieval MRR"), ("retrieval_top5", "retrieval Top-5")]:
        x = sec[sec.metric == metric]
        rows = []
        for m in models:
            r = {"모델": NAMES.get(m, m)}
            for tg in targets:
                y = x[(x.model == m) & (x.target == tg)]
                r[tg] = ("정의 안 됨" if len(y) and not np.isfinite(y.score.iloc[0]) else cell(y.score.iloc[0], y.ci_lo.iloc[0], y.ci_hi.iloc[0])) if len(y) else "—"
            rows.append(r)
        blocks.append(f"### 보조지표: {title} (클수록 좋음)\n\n" + md(pd.DataFrame(rows)))
    mg = mag.copy()
    mg["model"] = mg.model.map(lambda m: NAMES.get(m, m))
    blocks.append("### 반응 크기 비율과 retrieval 기준값 (분할·seed 평균)\n\n" + md(mg.round(3)))
    v = var.copy()
    v["model"] = v.model.map(lambda m: NAMES.get(m, m))
    blocks.append("### seed·분할 변동 (ΔRMSE, 실행 단위)\n\n" + md(v.round(4)))
    e = ens.copy()
    blocks.append("### seed 3개 예측을 먼저 평균한 ensemble (seed 평균과 다른 값이므로 따로 둔다)\n\n" + md(e.round(4)))
    cu = cur[cur.model.isin(ORDER)].pivot_table(index=["target", "model"], columns="k", values="delta_rmse").round(4).reset_index()
    cu["model"] = cu.model.map(lambda m: NAMES.get(m, m))
    cu.columns = [str(c_) if not isinstance(c_, str) else c_ for c_ in cu.columns]
    blocks.append("### 학습곡선: 적응 약물 수 k별 시험 ΔRMSE (1번 분할, seed 평균)\n\n" + md(cu))
    open(os.path.join(REP, "RESULT_TABLES.md"), "w").write("\n\n".join(blocks) + "\n")
    print("\n\n".join(blocks))


if __name__ == "__main__":
    main()
