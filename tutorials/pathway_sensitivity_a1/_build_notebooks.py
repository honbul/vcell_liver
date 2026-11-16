#!/usr/bin/env python3
"""Write the four tutorial notebooks (no outputs). Run once after editing the cell text below:
    python tutorials/pathway_sensitivity_a1/_build_notebooks.py
Cells are kept here as plain text so that changes are easy to review."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''import os, sys
root = os.getcwd()
while not os.path.isdir(os.path.join(root, "configs", "pathway_sensitivity_a1")):
    parent = os.path.dirname(root)
    assert parent != root, "저장소 안에서 실행해 주세요."
    root = parent
sys.path.insert(0, os.path.join(root, "src", "pathway_sensitivity_a1"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import core
cfg = core.load_config()
print("저장소:", root)
print("입력 캐시:", core.cache_dir(cfg))'''

NB = {}

NB["01_data_and_response.ipynb"] = [
    ("md", "# 01. RNA 한 표본과 약물 민감도 정답 읽기\n\n"
           "이 노트북은 학습을 하지 않습니다. 이미 만들어 둔 입력 캐시를 읽어, 한 세포 모델의 처리 전 RNA와 "
           "그 모델에 대해 측정된 약물 민감도(Activity Area, AA)가 어떻게 생겼는지 봅니다.\n\n"
           "먼저 서버 작업 폴더에서 `sbatch experiments/pathway_sensitivity_a1/a0_build.sbatch`가 한 번 실행되어 있어야 합니다. "
           "노트북 커널은 `assets/envs/psa1` 환경을 사용합니다."),
    ("code", SETUP),
    ("md", "## 1. 어떤 세포 모델이 분석에 들어가는가\n\n"
           "`model_registry.tsv`는 모델마다 연구 이름, 진단, 포함 여부, 계열(`family_id`)을 적어 둔 표입니다. "
           "같은 환자에서 나온 모델은 같은 계열로 묶여 항상 같은 fold에 들어갑니다."),
    ("code", 'reg = core.load_registry()\n'
             'lim = reg[reg.study == "LIMORE"]\n'
             'print("LIMORE 모델:", len(lim), "/ 포함:", int(lim.included.sum()), "/ 계열:", lim[lim.included].family_id.nunique())\n'
             'print(lim[~lim.included][["source_name", "exclusion_reason"]].to_string(index=False))\n'
             'multi = lim[lim.included].groupby("family_id").source_name.apply(list)\n'
             'print(multi[multi.map(len) > 1])'),
    ("md", "## 2. 한 표본의 RNA\n\n행은 세포 모델, 열은 유전자입니다. 값은 이미 log2로 변환된 FPKM입니다. 다시 로그를 취하지 않습니다."),
    ("code", 'limore = core.load_study(cfg, "LIMORE")\n'
             'axis = pd.read_csv(os.path.join(core.MANIFESTS, "gene_axis.tsv"), sep="\\t")\n'
             'print("행렬 크기 (모델 x 유전자):", limore.expr.shape, " 최솟값", limore.expr.min(), " 최댓값", limore.expr.max())\n'
             'm = "LIMORE__Huh7"\n'
             'i = limore.models.index(m)              # 이름으로 행을 찾습니다. 순서를 가정하지 않습니다.\n'
             'one = pd.Series(limore.expr[i], index=axis.symbol)\n'
             'print(one.loc[["ALB", "AFP", "GPC3", "TP53", "MKI67"]])'),
    ("md", "## 3. 그 표본의 정답: 약물별 AA\n\nAA가 클수록 그 약물에 민감합니다. 방향은 자료에서 확인합니다. "
           "AA가 큰 쌍일수록 IC50과 Emax(최대 농도에서 남은 생존 비율)가 작아야 합니다."),
    ("code", 'resp = core.load_limore_response(cfg)\n'
             'mine = resp[resp.model_id == m].sort_values("y", ascending=False)\n'
             'print(mine.head(5).to_string(index=False))\n'
             'from scipy.stats import spearmanr\n'
             'print("AA-IC50 Spearman:", round(spearmanr(resp.y, resp.IC50)[0], 3), " AA-Emax Spearman:", round(spearmanr(resp.y, resp.Emax)[0], 3))'),
    ("md", "## 4. 한 약물의 AA 분포\n\n약물마다 값의 범위가 다릅니다. 그래서 평가는 약물별로 따로 하고, 학습 자료의 표준편차로 나눈 오차를 씁니다."),
    ("code", 'drug = "Sorafenib"\n'
             'elig = set(lim[lim.included].model_id)\n'
             'y = resp[(resp.drug_id == drug) & resp.model_id.isin(elig)].y\n'
             'fig, ax = plt.subplots(figsize=(5, 3))\n'
             'ax.hist(y, bins=20, color="#2a78d6")\n'
             'ax.set_xlabel(f"{drug} Activity Area"); ax.set_ylabel("models")\n'
             'print("모델", len(y), " 평균", round(y.mean(), 3), " 표준편차", round(y.std(), 3))'),
]

NB["02_gene_vs_pathway.ipynb"] = [
    ("md", "# 02. 같은 표본의 유전자·순위·경로 표현\n\n"
           "같은 RNA를 세 가지로 표현합니다. (A1) 유전자별 log2 값, (A1r) 표본 안에서의 유전자 순위, (A3) Hallmark 50개 집합의 점수. "
           "경로 점수는 유전자 집합의 발현 요약이며, 단백질 활성이나 인과적 경로 활성을 직접 잰 값이 아닙니다."),
    ("code", SETUP),
    ("code", 'limore = core.load_study(cfg, "LIMORE")\n'
             'axis = pd.read_csv(os.path.join(core.MANIFESTS, "gene_axis.tsv"), sep="\\t")\n'
             'i = limore.models.index("LIMORE__Huh7")\n'
             'print("A1  유전자 표현:", limore.expr.shape[1], "차원")\n'
             'print("A1r 순위 표현:", limore.rank.shape[1], "차원")\n'
             'print("A3  경로 표현:", limore.hallmark.shape[1], "차원")'),
    ("md", "## 순위 표현\n\n표본 하나 안에서 유전자를 발현 순으로 줄 세우고 전체 유전자 수로 나눕니다. 다른 표본의 값은 전혀 쓰지 않습니다."),
    ("code", 'genes = ["ALB", "AFP", "GPC3", "TP53", "MKI67"]\n'
             'idx = [list(axis.symbol).index(g) for g in genes]\n'
             'print(pd.DataFrame({"log2 FPKM": limore.expr[i, idx], "상대 순위": limore.rank[i, idx]}, index=genes))\n'
             'assert np.allclose(core.relative_rank(limore.expr[[i]]), limore.rank[[i]])   # 캐시와 같은 계산'),
    ("md", "## 경로 점수 (singscore)\n\n집합에 속한 유전자들의 평균 순위를 이론적 최소·최대로 정규화한 값입니다. "
           "저장된 점수는 Bioconductor singscore로 계산했습니다. 아래에서 한 집합을 numpy로 직접 계산해 같은지 확인합니다. "
           "이 셀은 서버의 Hallmark GMT 파일을 읽습니다."),
    ("code", 'gmt = core.read_gmt(core.rpath(cfg["paths"]["hallmark_gmt"]))\n'
             'name = "HALLMARK_G2M_CHECKPOINT"\n'
             'pos = {s: k for k, s in enumerate(axis.symbol)}\n'
             'members = np.array([pos[g] for g in dict.fromkeys(gmt[name]) if g in pos])\n'
             'ranks = core.min_rank(limore.expr[[i]])\n'
             'mine = core.singscore_up(ranks, members)[0]\n'
             'stored = limore.hallmark[i, limore.sets.index(name)]\n'
             'print(name, "유전자", len(members), "개  numpy:", round(mine, 6), " 저장값:", round(stored, 6))\n'
             'assert abs(mine - stored) < 1e-9'),
    ("code", 'scores = pd.Series(limore.hallmark[i], index=limore.sets).sort_values()\n'
             'print("이 표본에서 점수가 가장 낮은 집합:\\n", scores.head(5))\n'
             'print("가장 높은 집합:\\n", scores.tail(5))'),
    ("md", "## 음성 대조: 유전자 이름을 섞은 경로\n\n집합의 크기와 집합끼리의 겹침은 그대로 두고 유전자 이름만 바꾼 20개 대조가 있습니다. "
           "실제 Hallmark가 이 대조보다 예측에 도움이 되는지를 보는 용도입니다."),
    ("code", 'perm = core.load_perm_scores(cfg, "LIMORE", cfg["a4"]["seeds"][0], limore.models, limore.sets)\n'
             'print("실제 집합 점수의 모델 간 표준편차 평균:", round(limore.hallmark.std(axis=0).mean(), 4))\n'
             'print("섞은 집합 점수의 모델 간 표준편차 평균:", round(perm.std(axis=0).mean(), 4))'),
]

NB["03_predict_one_drug.ipynb"] = [
    ("md", "# 03. 한 약물에서 평균·유전자 Ridge·경로 Ridge 비교\n\n"
           "약물 하나(Sorafenib)와 반복 하나(repeat 0)만 사용하는 작은 예제입니다. 몇 초 안에 끝납니다. "
           "전체 실행은 노트북이 아니라 `sbatch experiments/pathway_sensitivity_a1/a2_internal.sbatch`로 합니다.\n\n"
           "절차: 계열 단위로 5조각으로 나누고, 4조각으로 학습해 남은 1조각을 예측합니다. "
           "Ridge의 alpha는 학습 4조각 안에서 다시 3조각으로 나눠 고릅니다. 시험 조각의 정답은 채점에만 씁니다."),
    ("code", SETUP + "\nimport nested, metrics"),
    ("code", 'drug = "Sorafenib"\n'
             'limore = core.load_study(cfg, "LIMORE")\n'
             'sp = core.load_outer_splits(); sp = sp[sp.repeat == 0]\n'
             'resp = core.load_limore_response(cfg)\n'
             'models = list(sp.model_id)\n'
             'rows = [limore.models.index(m) for m in models]          # 이름으로 정렬을 맞춥니다\n'
             'Y = core.response_matrix(resp, models, [drug], "y")\n'
             'fam, fold = sp.family_id.values, sp.outer_fold.values\n'
             'X = {"A0": np.zeros((len(models), 1)), "A1": limore.expr[rows], "A3": limore.hallmark[rows]}\n'
             'print("모델", len(models), " 계열", len(set(fam)), " fold별 모델 수", np.bincount(fold))'),
    ("code", 'pred, s_train = {}, np.zeros(len(models))\n'
             'for kind in X:\n'
             '    pred[kind] = np.zeros(len(models))\n'
             '    for k, tr, te, p, info in nested.outer_folds(X[kind], Y, fam, fold, kind, cfg["model"]["alpha_grid"],\n'
             '                                                 cfg["split"]["inner_folds"], lambda k: 7100 + k, K=50):\n'
             '        assert not set(fam[tr]) & set(fam[te])            # 같은 계열이 학습과 시험에 함께 있지 않음\n'
             '        pred[kind][te] = p[:, 0]\n'
             '        s_train[te] = info["sd"][0]                      # 학습 조각의 표준편차\n'
             '        if kind != "A0": print(kind, "fold", k, "선택된 alpha", info["alpha"][0])'),
    ("md", "## 채점\n\n오차를 학습 표준편차로 나눈 뒤, 같은 계열 안에서 먼저 평균하고 계열 사이를 평균해 제곱근을 취합니다. "
           "1.0 근처면 평균만 예측하는 것과 비슷하다는 뜻입니다."),
    ("code", 'y = Y[:, 0]\n'
             'for kind in X:\n'
             '    z2 = pd.Series(((y - pred[kind]) / s_train) ** 2).groupby(fam).mean()\n'
             '    yf, pf = pd.Series(y).groupby(fam).mean(), pd.Series(pred[kind]).groupby(fam).mean()\n'
             '    print(kind, " 정규화 RMSE", round(float(np.sqrt(z2.mean())), 3), " Spearman", round(metrics.safe_spearman(yf, pf), 3))\n'
             'print("주의: A0의 Spearman은 의미가 없습니다. 시험 조각을 뺀 평균이라 값이 거의 일정합니다.")'),
    ("code", 'fig, ax = plt.subplots(figsize=(4.4, 4))\n'
             'for kind, c in (("A1", "#2a78d6"), ("A3", "#eda100")):\n'
             '    ax.plot(y, pred[kind], "o", color=c, ms=5, label=kind)\n'
             'ax.plot([y.min(), y.max()], [y.min(), y.max()], color="#52514e", lw=1, ls="--")\n'
             'ax.set_xlabel("observed AA"); ax.set_ylabel("out-of-fold prediction"); ax.legend(frameon=False)'),
]

NB["04_results_and_external.ipynb"] = [
    ("md", "# 04. 저장된 예측표에서 결과 표와 그림 다시 만들기\n\n"
           "이 노트북은 학습을 하지 않습니다. Git에 들어 있는 out-of-fold 예측표(`reports/pathway_sensitivity_a1/oof/`)와 "
           "요약표(`reports/pathway_sensitivity_a1/tables/`)를 읽습니다. 예측표에는 예측값만 있으므로 실측 AA는 ID로 다시 붙입니다 "
           "(서버의 입력 캐시가 필요합니다)."),
    ("code", SETUP + "\nimport metrics, json\ntables = os.path.join(core.REPORTS, 'tables')\n"
                     "runs = json.load(open(os.path.join(core.REPORTS, 'oof', 'RUNS.json')))\nprint(runs)"),
    ("md", "## 1. 주지표를 예측표에서 다시 계산"),
    ("code", 'reg = core.load_registry(); family_of = dict(zip(reg.model_id, reg.family_id))\n'
             'long = core.load_limore_response(cfg); sp = core.load_outer_splits()\n'
             'repeats, fams, drugs = sorted(sp.repeat.unique()), sorted(sp.family_id.unique()), sorted(long.drug_id.unique())\n'
             'main = os.path.join(core.REPORTS, "oof", runs["main"])\n'
             'a0 = metrics.load_oof(main, "A0", 3)\n'
             's_ref = a0[["repeat", "outer_fold", "drug_id", "s_train"]].drop_duplicates()\n'
             'cube = {}\n'
             'for name in ["A0", "A1", "A1r", "A2", "A3"]:\n'
             '    df = metrics.attach_truth(metrics.load_oof(main, name, 3), long, family_of, s_ref)\n'
             '    cube[name] = metrics.error_cube(df, repeats, drugs, fams)\n'
             'mine = pd.Series({k: metrics.macro_nrmse(v) for k, v in cube.items()}, name="다시 계산")\n'
             'saved = pd.read_csv(os.path.join(tables, "internal_summary.tsv"), sep="\\t").set_index("model")\n'
             'print(pd.concat([mine, saved.macro_drug_normalized_rmse.rename("저장된 표")], axis=1))\n'
             'assert np.allclose(mine, saved.macro_drug_normalized_rmse.loc[mine.index])'),
    ("md", "## 2. 계열 bootstrap으로 차이의 구간 다시 계산\n\n계열을 복원추출하고, 뽑힌 계열의 모든 약물·반복 예측을 함께 움직입니다."),
    ("code", 'W = metrics.bootstrap_weights(len(fams), cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"])\n'
             'for a, b in [("A3", "A0"), ("A3", "A1"), ("A3", "A1r"), ("A3", "A2")]:\n'
             '    d = metrics.boot_macro(cube[a], W) - metrics.boot_macro(cube[b], W)\n'
             '    print(f"{a} - {b}: {metrics.macro_nrmse(cube[a]) - metrics.macro_nrmse(cube[b]):+.4f}  95% 구간 [{np.quantile(d, .025):+.4f}, {np.quantile(d, .975):+.4f}]")'),
    ("md", "## 3. 대조와 외부 평가 표"),
    ("code", 'for f in ["internal_differences.tsv", "learning_curve.tsv", "external_summary.tsv"]:\n'
             '    print("\\n==", f); print(pd.read_csv(os.path.join(tables, f), sep="\\t").to_string(index=False))\n'
             'print(json.dumps(json.load(open(os.path.join(tables, "controls_results.json")))["a4"], indent=1))'),
    ("md", "## 4. 그림 다시 만들기\n\n결과는 `reports/pathway_sensitivity_a1/figures_regenerated/`에 저장됩니다. 저장소에 있는 그림을 덮어쓰지 않습니다."),
    ("code", 'import make_figures as mf\n'
             'out = os.path.join(core.REPORTS, "figures_regenerated"); os.makedirs(out, exist_ok=True)\n'
             'mf.fig_internal(out, tables); mf.fig_learning(out, tables); mf.fig_a4(out, tables); mf.fig_external(out, tables)\n'
             'mf.fig_cases(out, tables, os.path.join(main, "oof"))\n'
             'print(sorted(os.listdir(out)))'),
]


def cell(kind, text):
    lines = text.split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


for name, cells in NB.items():
    nb = {"cells": [cell(k, t) for k, t in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    with open(os.path.join(HERE, name), "w") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print("wrote", name)
