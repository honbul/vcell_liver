#!/usr/bin/env python3
"""Write the R1 tutorial notebooks (no outputs). Run once after editing the cell text below:
    python tutorials/softprior_multitask_r1/_build_notebooks.py
then execute them (see README.md in this folder). Cells are kept here as plain text so that changes are
easy to review."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = '''import os, sys
root = os.getcwd()
while not os.path.isdir(os.path.join(root, "configs", "softprior_multitask_r1")):
    parent = os.path.dirname(root)
    assert parent != root, "저장소 안에서 실행해 주세요."
    root = parent
sys.path.insert(0, os.path.join(root, "src", "softprior_multitask_r1"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.width", 200)
T = os.path.join(root, "reports", "softprior_multitask_r1", "tables")
print("결과표 폴더를 찾았습니다:", os.path.isdir(T))'''

NB = {}

NB["01_inspect_results.ipynb"] = [
    ("md", "# 01. 저장된 결과 둘러보기 (재학습 없음)\n\n"
           "이 노트북은 저장소에 커밋된 작은 표만 읽습니다. 서버의 원자료, Slurm, GPU가 필요하지 않습니다.\n\n"
           "질문은 하나입니다. **유전자 전체를 남겨 둔 채 경로 방향만 덜 규제하고(soft prior), 약물 90개를 함께 학습하면 "
           "처음 보는 간암 세포 계열의 약물 민감도(AA) 예측이 좋아지는가?**\n\n"
           "점수는 `macro nRMSE`입니다. 약물마다 예측 오차를 학습 자료의 표준편차로 나눈 뒤 평균한 값이며, **낮을수록 좋습니다.** "
           "학습 평균만 내놓는 예측이 대략 1.05입니다. 같은 자료의 결과가 연구 A에서 이미 공개되었으므로, 아래 숫자는 "
           "'동결 뒤 재사용 공개 벤치마크 평가'이지 처음 보는 자료에서의 검증이 아닙니다."),
    ("code", SETUP),
    ("md", "## 1. 최종 비교표\n\n행 하나가 모델 하나입니다. 구간은 계열(환자) 76개를 다시 뽑아 만든 95% 구간입니다. "
           "세 번의 반복 점수와, 반복을 평균한 예측의 점수(`ensemble`)를 따로 적었습니다."),
    ("code", 'summ = pd.read_csv(os.path.join(T, "final_internal_summary.tsv"), sep="\\t")\n'
             'cols = ["model", "macro_drug_normalized_rmse", "ci_low", "ci_high", "repeat0", "repeat1", "repeat2", "ensemble_of_repeats_macro_nrmse", "mean_spearman"]\n'
             'print(summ[cols].round(4).to_string(index=False))'),
    ("md", "모델 이름:\n\n"
           "- `A0` 학습 평균, `A1` 유전자 log2 Ridge, `A1r` 유전자 순위 Ridge, `A3` Hallmark 점수 50개 Ridge — 연구 A의 방법 그대로(유전자 축만 고침)\n"
           "- `enet` raw RNA ElasticNet, `cwkbmf`·`tabpfn` 공개 방법(실행된 경우에만 행이 있습니다)\n"
           "- `ridge_pd` / `ridge_gl` 유전자 순위 Ridge, 규제 강도를 약물마다 / 모든 약물에 하나\n"
           "- `par_pd` / `par_gl` 경로 부분공간만 쓰는 hard prior\n"
           "- `soft_pd` / `soft_gl` soft prior(경로 밖 방향을 더 세게 규제), 약물 공유 없음\n"
           "- `mt_noprior` prior 없는 다중약물 공유, `full` 처음의 통합 설계\n"
           "- `mt_resid` 공유 low-rank + 약물별 Ridge 잔차(prior 없음), `full_resid` 같은 구조에 soft prior를 더한 **최종 통합 모델**"),
    ("code", 'order = summ.sort_values("macro_drug_normalized_rmse")\n'
             'fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(order) + 1))\n'
             'y = np.arange(len(order))[::-1]\n'
             'ax.errorbar(order.macro_drug_normalized_rmse, y, xerr=[order.macro_drug_normalized_rmse - order.ci_low, order.ci_high - order.macro_drug_normalized_rmse], fmt="o", color="#2a78d6", ecolor="#8a8983", capsize=2)\n'
             'ax.set_yticks(y); ax.set_yticklabels(order.model); ax.set_xlabel("macro nRMSE (lower is better), 95% family bootstrap")\n'
             'plt.tight_layout(); plt.show()'),
    ("md", "구간이 서로 크게 겹칩니다. 그래서 모델 둘을 비교할 때는 **같은 계열에서의 차이**를 다시 뽑아 구간을 만듭니다(짝지은 비교). "
           "차이의 구간이 0을 포함하면 '다르다고 말할 수 없다'입니다."),
    ("code", 'diff = pd.read_csv(os.path.join(T, "final_internal_differences.tsv"), sep="\\t")\n'
             'print(diff.round(4).to_string(index=False))'),
    ("md", "## 2. 실제 경로와 무작위 경로\n\n유전자 이름표를 통째로 섞어 만든 무작위 '경로' 20벌로 같은 절차를 반복했습니다. "
           "실제 Hallmark 구성이 의미가 있다면 실제 prior의 점수가 무작위 20개보다 낮아야 합니다."),
    ("code", 'rp = pd.read_csv(os.path.join(T, "final_internal_real_vs_permuted_prior.tsv"), sep="\\t")\n'
             'print(rp.round(5).to_string(index=False))\n'
             'ps = pd.read_csv(os.path.join(T, "final_internal_permuted_prior_scores.tsv"), sep="\\t")\n'
             'fig, axes = plt.subplots(1, len(rp), figsize=(3.2 * len(rp), 2.8))\n'
             'for ax, (_, r) in zip(np.atleast_1d(axes), rp.iterrows()):\n'
             '    v = ps[ps.model == r.model].macro_drug_normalized_rmse\n'
             '    ax.hist(v, bins=8, color="#8a8983"); ax.axvline(r.real, color="#eb6834", lw=2)\n'
             '    ax.set_title(r.model); ax.set_xlabel("macro nRMSE")\n'
             'axes[0].set_ylabel("permuted priors (of 20)"); plt.tight_layout(); plt.show()\n'
             'print("주황 선 = 실제 Hallmark prior")'),
    ("md", "## 3. 각 모델이 실제로 고른 설정\n\n설정 이름은 `ridge|종류|λ`, `nuc|η/λ|λ|τ비율`, `res|η/λ|λ|τ비율|λ₂`입니다. "
           "η/λ가 0이면 prior를 쓰지 않은 것이고, τ비율이 클수록 약물 사이 공유가 강합니다(해의 rank가 낮습니다)."),
    ("code", 'ch = pd.read_csv(os.path.join(T, "final_internal_chosen.tsv"), sep="\\t")\n'
             'for m in ["soft_gl", "mt_noprior", "full_resid"]:\n'
             '    print(m); print(ch[ch.model == m].head(6).to_string(index=False)); print()\n'
             'fr = ch[ch.model == "full_resid"].copy()\n'
             'fr["eta_over_lambda"] = fr.config.str.split("|").str[1]\n'
             'print("full_resid에서 선택된 η/λ의 비율:"); print(fr.groupby("eta_over_lambda").fraction.sum().round(3))'),
    ("md", "## 4. 약물 90개를 함께 학습해도 표본이 90배가 되지 않는 이유\n\n"
           "행렬은 79 × 90 = 7,110칸이지만 서로 독립인 것은 계열 76개입니다. 약물 90개의 값은 같은 세포에서 나왔고 서로 상관되어 있습니다. "
           "아래 표는 학습 계열에서 구한 '약물 공통 성분'이 보지 않은 계열의 분산을 얼마나 설명하는지입니다(개발 구간에서 계산). "
           "첫 성분 하나가 약 30%를 차지합니다. 즉 약물 90개는 90개의 독립된 문제가 아니라 상당 부분 같은 문제의 반복입니다."),
    ("code", 'd2 = pd.read_csv(os.path.join(T, "d2_heldout_subspace.tsv"), sep="\\t")\n'
             'print(d2.round(4).to_string(index=False))'),
    ("md", "약물을 늘려도 구간의 폭은 계열 수가 정합니다. 최종표에서 어느 모델이든 구간의 폭이 약 0.18인 것이 그 결과입니다."),
    ("code", 'print((summ.ci_high - summ.ci_low).round(3).describe()[["min", "max"]])'),
    ("md", "## 5. 전반적 민감도와 약물 선호도\n\n각 세포의 예측과 실측에서 **각자의** 약물 평균을 뺍니다. 평균은 '이 세포가 전반적으로 약에 잘 듣는가', "
           "나머지는 '어떤 약에 상대적으로 더 잘 듣는가'입니다. 두 부분의 일치도를 따로 봅니다."),
    ("code", 'pref = pd.read_csv(os.path.join(T, "final_internal_broad_vs_preference.tsv"), sep="\\t")\n'
             'print(pref.round(4).to_string(index=False))'),
    ("md", "## 6. 외부 자료(LCCL)\n\nLCCL은 다른 연구가 다른 지표(AUC)로 잰 자료입니다. 값을 합치지 않고, 약물마다 예측 순위와 실측 순위의 일치도(Spearman)만 봅니다. "
           "`cross_study_leave_family_out`은 LIMORE에도 있는 계열(해당 계열은 학습에서 뺌), `new_lineage`는 LIMORE에 없는 계열 4개입니다. "
           "4개 계열로는 요약 통계가 의미를 갖기 어렵습니다."),
    ("code", 'ex = pd.read_csv(os.path.join(T, "final_external_summary.tsv"), sep="\\t")\n'
             'print(ex[ex.endpoint == "AUC"].round(3).to_string(index=False))\n'
             'print(pd.read_csv(os.path.join(T, "final_external_differences.tsv"), sep="\\t").query("endpoint == \'AUC\'").round(3).to_string(index=False))'),
    ("md", "## 7. 실행하지 못한 비교군\n\n비어 있으면 모두 실행된 것입니다."),
    ("code", 'print(pd.read_csv(os.path.join(T, "final_not_run.tsv"), sep="\\t").to_string(index=False))'),
]

NB["02_pathway_and_rest_one_cell.ipynb"] = [
    ("md", "# 02. 실제 세포 하나: 경로 부분과 나머지 부분\n\n"
           "이 노트북은 서버의 입력 캐시가 있어야 실행됩니다(`experiments/softprior_multitask_r1/r0_build.sbatch`가 만든 것). "
           "연구 전체를 다시 학습하지 않고, 한 학습 집합에서 닫힌 해로 풀리는 작은 Ridge 몇 개만 계산합니다(수 초). "
           "RNA 값은 출력하지 않습니다(예시 약물 하나의 관측 AA와 예측만 나란히 보여 줍니다)."),
    ("code", SETUP + '\nimport r1_data as R, r1_models as M\ncfg = R.load_config(); data = R.load_data(cfg)\n'
                     'print("모델", len(data.models), "계열", len(set(data.fam)), "유전자", data.rank.shape[1], "약물", len(data.drugs), "경로", len(data.sets))'),
    ("md", "## 1. 유전자 19,018개의 방향 가운데 '경로 방향'은 몇 개인가\n\n"
           "Hallmark 경로 50개의 평균 점수로 만들 수 있는 예측은, 유전자 공간에서 보면 최대 50개의 방향만 사용합니다. "
           "soft prior는 이 50개 방향은 보통 강도로, 나머지 방향은 더 세게 규제합니다. 나머지를 **버리지는 않습니다.**"),
    ("code", 'fold = data.outer_fold(0)\ntr, te = fold != 0, fold == 0\n'
             'prep = M.Prep(data.rank[tr], data.Y[tr], data.fam[tr], data.rank[te], data.sets)\n'
             'print("학습 모델", int(tr.sum()), "/ 보류 모델", int(te.sum()))\n'
             'print("경로 부분공간의 차원:", prep.r)\n'
             'print("표준화한 학습 RNA 분산 가운데 경로 부분공간에 있는 몫: %.1f%%" % (100 * prep.par_fraction))'),
    ("md", "## 2. 보류된 세포 하나를 두 부분으로 나누기\n\n세포의 표준화된 RNA 벡터 x를 x = (경로 부분) + (나머지 부분)으로 나눕니다. 두 부분은 직교합니다."),
    ("code", 'import r1_solvers as S\n'
             'Q = S.orth_basis(S.membership_std(data.sets, prep.x_sd))\n'
             'i = np.where(te)[0][0]\n'
             'x = S.standardise(data.rank[[i]], prep.x_mu, prep.x_sd)[0]\n'
             'x_par = Q @ (Q.T @ x); x_rest = x - x_par\n'
             'print("세포:", data.models[i])\n'
             'print("전체 크기² %.0f = 경로 부분 %.0f + 나머지 %.0f" % (x @ x, x_par @ x_par, x_rest @ x_rest))\n'
             'print("경로 부분의 몫: %.1f%%, 두 부분의 내적: %.1e" % (100 * (x_par @ x_par) / (x @ x), x_par @ x_rest))'),
    ("md", "경로 부분은 이 세포 RNA 변이의 일부만 담습니다. hard prior(경로 점수만 입력)는 나머지를 전부 버리는 선택입니다."),
    ("md", "## 3. 같은 약물에서 hard / soft / prior 없음의 예측\n\n같은 학습 집합, 같은 규제 강도(λ=1)에서 세 가지를 풀어 보류 세포의 예측을 비교합니다. "
           "η/λ = ∞가 hard, 10이 soft, 0이 prior 없는 Ridge입니다."),
    ("code", 'drug = "Sorafenib"; d = data.drugs.index(drug)\n'
             'out = pd.DataFrame({"observed_AA": data.Y[te][:, d]}, index=[m.replace("LIMORE__", "") for m, t in zip(data.models, te) if t])\n'
             'for name, kernel in [("hard", "par"), ("soft_eta10", "r10"), ("no_prior", "whole")]:\n'
             '    out[name] = prep.to_raw(prep.predict_ridge_z(kernel, 1.0))[:, d]\n'
             'out["train_mean"] = prep.y_mu[d]\n'
             'print(out.round(3).to_string())\n'
             'print("\\n예측끼리의 상관:"); print(out[["hard", "soft_eta10", "no_prior"]].corr().round(3))\n'
             'print("\\n이 fold·이 약물의 RMSE:"); print(((out.drop(columns="observed_AA").sub(out.observed_AA, axis=0)) ** 2).mean().pow(0.5).round(4))'),
    ("md", "한 fold, 한 약물의 숫자는 우연에 크게 흔들립니다. 결론은 01 노트북의 전체 표(계열 76개, 약물 90개, 반복 3회)에서만 읽어야 합니다."),
    ("md", "## 4. 유전자를 많이 넣어서 좋아진 것인가, 규제가 기여한 것인가\n\n"
           "같은 학습 집합의 안쪽 fold에서, 규제 강도 λ를 바꿔 가며 세 입력의 손실을 봅니다. "
           "'전체'는 λ가 작으면 나빠지고 적당한 λ에서 가장 좋습니다. 유전자를 더 넣는 것만으로는 좋아지지 않고, 맞는 규제가 함께 있어야 합니다. "
           "'경로 뺀 나머지'가 '전체'와 거의 같다는 점도 보입니다."),
    ("code", 'lams = cfg["objective"]["lambda_grid"]\n'
             'cfgs = M.ridge_configs(lams, ["whole", "par", "perp"])\n'
             'loss, _ = M.inner_losses(data.rank[tr], data.Y[tr], data.fam[tr], data.sets, cfgs, 3, R.inner_seed(cfg, 0, 0))\n'
             'tab = pd.DataFrame(np.sqrt(loss).mean(axis=1).reshape(3, len(lams)), index=["전체", "경로만(hard)", "경로 뺀 나머지"], columns=lams)\n'
             'print(tab.round(4).to_string())\n'
             'fig, ax = plt.subplots(figsize=(5.5, 3.2))\n'
             'for (name, row), c in zip(tab.iterrows(), ["#2a78d6", "#eda100", "#1baf7a"]):\n'
             '    ax.plot(lams, row.values, "o-", color=c, label=["whole", "pathway only (hard)", "complement only"][list(tab.index).index(name)])\n'
             'ax.set_xscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("inner-fold macro nRMSE"); ax.legend(frameon=False); plt.tight_layout(); plt.show()'),
]

NB["03_penalty_demo_synthetic.ipynb"] = [
    ("md", "# 03. 가상 자료로 벌점(penalty) 바꿔 보기\n\n"
           "실제 자료 없이 실행됩니다. 작은 가상 자료를 만들고, 연구에 쓴 것과 같은 풀이 코드(`r1_solvers.py`)로 "
           "벌점 두 개를 바꿔 봅니다.\n\n"
           "- **η (soft prior):** 경로 밖 방향을 얼마나 더 세게 누를지. 0이면 보통 Ridge, 아주 크면 경로 방향만 남습니다(hard).\n"
           "- **τ (약물 공유):** 약물들의 계수가 소수의 공통 방향을 쓰도록 누르는 강도. 클수록 해의 rank가 낮아집니다.\n\n"
           "가상 세계를 두 개 만듭니다. **세계 A**는 신호가 경로 안에 있고, **세계 B**는 신호가 경로 밖에 있습니다. "
           "실제 자료가 어느 쪽에 가까운지는 01 노트북의 결과가 말해 줍니다."),
    ("code", 'import os, sys\nroot = os.getcwd()\nwhile not os.path.isdir(os.path.join(root, "configs", "softprior_multitask_r1")):\n'
             '    parent = os.path.dirname(root); assert parent != root; root = parent\n'
             'sys.path.insert(0, os.path.join(root, "src", "softprior_multitask_r1"))\n'
             'import numpy as np, matplotlib.pyplot as plt\nimport r1_solvers as S\n'
             'rs = np.random.RandomState(0)\nN, N_TEST, G, D, K = 100, 400, 200, 30, 6\n'
             'sets = [np.sort(rs.choice(G, 20, replace=False)) for _ in range(K)]     # 가상 "경로" 6개, 유전자 20개씩\n'
             'X, X_test = rs.normal(size=(N, G)), rs.normal(size=(N_TEST, G))\n'
             'M0 = np.zeros((G, K))\nfor k, idx in enumerate(sets): M0[idx, k] = 1 / len(idx)\n'
             'Qtrue = S.orth_basis(M0)\n'
             'shared = rs.normal(size=(2, D))                                          # 약물들이 공유하는 방향 2개\n'
             'W_A = Qtrue[:, :2] @ shared * 2.0                                        # 세계 A: 신호가 경로 방향에 있음\n'
             'out_dir = rs.normal(size=(G, 2)); out_dir -= Qtrue @ (Qtrue.T @ out_dir); out_dir /= np.linalg.norm(out_dir, axis=0)\n'
             'W_B = out_dir @ shared * 2.0                                             # 세계 B: 신호가 경로 밖에 있음\n'
             'def world(W):\n    return X @ W + rs.normal(size=(N, D)), X_test @ W\n'
             'print("표본", N, "유전자", G, "약물", D, "경로", K)'),
    ("md", "## 풀이 함수\n\n연구 코드와 같은 순서입니다. 학습 자료만으로 표준화하고, 경로 부분공간 Q를 만들고, 축약 공간에서 풉니다."),
    ("code", 'def fit_predict(Y, sets, ratio, lam, tau_frac):\n'
             '    """ratio = η/λ (np.inf면 hard), tau_frac = τ/τ_max. 시험 예측과 해의 rank를 돌려줍니다."""\n'
             '    wt = np.full(N, 1 / N)\n'
             '    mu, sd = S.wmean_sd(X, wt); Xs, Xt = S.standardise(X, mu, sd), S.standardise(X_test, mu, sd)\n'
             '    ymu, ysd = S.wmean_sd(Y, wt); Z = (Y - ymu) / ysd\n'
             '    Q = S.orth_basis(S.membership_std(sets, sd))\n'
             '    if np.isinf(ratio):                       # hard: 경로 방향만 사용\n'
             '        Xs, Xt, Q, ratio = Xs @ Q @ Q.T, Xt @ Q @ Q.T, Q, 0.0\n'
             '    B = S.reduced_basis(Xs, Q); XB, XtB = Xs @ B, Xt @ B\n'
             '    H, Rm = S.smooth_terms(XB, wt, Z, Q.shape[1], lam, ratio * lam)\n'
             '    if tau_frac == 0:\n        C = np.linalg.solve(H, Rm); rank = min(C.shape)\n'
             '    else:\n        C, info = S.admm_nuclear(H, Rm, tau_frac * S.tau_max(XB, wt, Z)); rank = info["rank"]\n'
             '    return (XtB @ C) * ysd + ymu, rank\n\n'
             'def test_error(Y, truth, sets, ratio, lam, tau_frac):\n'
             '    P, rank = fit_predict(Y, sets, ratio, lam, tau_frac)\n'
             '    return float(np.sqrt(((P - truth) ** 2).mean()) / truth.std()), rank     # 1.0 = 아무것도 배우지 못함'),
    ("md", "## 1. η를 바꾸기: hard, soft, prior 없음\n\nλ는 고정하고 η/λ만 바꿉니다. 맨 오른쪽(∞)이 hard입니다."),
    ("code", 'ratios = [0, 1, 10, 100, 1000, np.inf]\nYA, TA = world(W_A); YB, TB = world(W_B)\n'
             'errA = [test_error(YA, TA, sets, r, 0.01, 0)[0] for r in ratios]\n'
             'errB = [test_error(YB, TB, sets, r, 0.01, 0)[0] for r in ratios]\n'
             'print("eta/lambda:", ratios)\nprint("세계 A(신호가 경로 안):", np.round(errA, 3))\nprint("세계 B(신호가 경로 밖):", np.round(errB, 3))\n'
             'fig, ax = plt.subplots(figsize=(5.5, 3.2)); xs = np.arange(len(ratios))\n'
             'ax.plot(xs, errA, "o-", color="#2a78d6", label="world A: signal inside pathways")\n'
             'ax.plot(xs, errB, "o-", color="#eb6834", label="world B: signal outside pathways")\n'
             'ax.set_xticks(xs); ax.set_xticklabels(["0\\n(Ridge)", "1", "10", "100", "1000", "inf\\n(hard)"]); ax.set_xlabel("eta / lambda"); ax.set_ylabel("test error / signal SD")\n'
             'ax.legend(frameon=False); plt.tight_layout(); plt.show()'),
    ("md", "오차는 신호의 표준편차로 나눈 값입니다. 1.0이면 아무것도 배우지 못한 것입니다. 세계 A에서는 η를 키울수록 좋아집니다. 세계 B에서는 hard가 신호를 통째로 잃습니다. "
           "soft prior의 장점은 **η를 자료로 고를 수 있다**는 것입니다. 세계 B에서는 η=0을 고르면 Ridge로 돌아갑니다. "
           "다만 그때 soft prior가 Ridge보다 좋아지는 것은 아닙니다."),
    ("md", "## 2. 경로 이름표를 섞으면\n\n세계 A에서 유전자 이름표를 섞은 무작위 '경로'를 prior로 쓰면 이득이 사라져야 합니다. "
           "실제 자료에서도 같은 대조를 20번 했습니다."),
    ("code", 'real = test_error(YA, TA, sets, 100, 0.01, 0)[0]\n'
             'perm = []\nfor s in range(10):\n    p = np.random.RandomState(100 + s).permutation(G)\n'
             '    perm.append(test_error(YA, TA, [np.sort(p[idx]) for idx in sets], 100, 0.01, 0)[0])\n'
             'print("실제 경로 prior: %.3f / 섞은 경로 prior 10개: 평균 %.3f (최소 %.3f, 최대 %.3f) / prior 없음: %.3f" % (real, np.mean(perm), min(perm), max(perm), errA[0]))'),
    ("md", "## 3. τ를 바꾸기: 약물 사이 공유\n\n두 세계 모두 약물 30개가 방향 2개를 공유하도록 만들었습니다. τ를 키우면 해의 rank가 내려갑니다. "
           "적당한 τ에서 오차가 조금 줄고, 너무 크면 계수 전체가 0 쪽으로 눌려 다시 나빠집니다. 이득의 크기는 자료에 달려 있습니다."),
    ("code", 'fracs = [0, 0.03, 0.1, 0.2, 0.3, 0.5]\n'
             'res = [test_error(YB, TB, sets, 0, 0.01, f) for f in fracs]\n'
             'for f, (e, r) in zip(fracs, res): print("tau/tau_max = %.2f  rank %2d  상대 오차 %.3f" % (f, r, e))\n'
             'fig, ax = plt.subplots(figsize=(5.5, 3.2)); ax.plot(fracs, [e for e, _ in res], "o-", color="#1baf7a")\n'
             'for f, (e, r) in zip(fracs, res): ax.annotate("rank %d" % r, (f, e), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)\n'
             'ax.set_xlabel("tau / tau_max"); ax.set_ylabel("test error / signal SD (world B)"); plt.tight_layout(); plt.show()'),
    ("md", "## 4. 공유할 것이 없는 세계\n\n약물마다 계수가 서로 무관하면 τ는 도움이 되지 않습니다. 실제 자료에서 공유의 이득이 작았다면, "
           "약물별로 따로 맞춘 Ridge가 이미 필요한 정보를 대부분 쓰고 있었다는 뜻일 수 있습니다."),
    ("code", 'W_C = rs.normal(size=(G, D)) * 0.2                                    # 약물마다 무관한 계수\nYC, TC = world(W_C)\n'
             'for f in [0, 0.1, 0.3, 0.5]:\n    e, r = test_error(YC, TC, sets, 0, 0.01, f); print("tau/tau_max = %.2f  rank %2d  상대 오차 %.3f" % (f, r, e))'),
    ("md", "## 정리\n\n- hard prior는 신호가 경로 밖에 있을 때 크게 잃습니다. soft prior는 그 위험을 없애지만, 경로가 맞지 않으면 이득도 없습니다.\n"
           "- 무작위 경로와의 비교가 'prior가 실제로 기여했는가'를 가릅니다.\n"
           "- 약물 공유는 약물들이 정말로 방향을 공유할 때만 돕습니다.\n"
           "- 이 가상 자료는 벌점의 동작을 보여 줄 뿐, 실제 간암 자료의 결론이 아닙니다."),
]


def main():
    for name, cells in NB.items():
        nb = dict(nbformat=4, nbformat_minor=5,
                  metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                language_info=dict(name="python")),
                  cells=[])
        for i, (kind, src) in enumerate(cells):
            cell = dict(cell_type="markdown" if kind == "md" else "code", id=f"c{i:02d}", metadata={}, source=src)
            if kind == "code":
                cell.update(outputs=[], execution_count=None)
            nb["cells"].append(cell)
        with open(os.path.join(HERE, name), "w") as fh:
            json.dump(nb, fh, ensure_ascii=False, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
