"""Build the R5 tutorial notebook. Executed on the server by experiments/perturbation_design_r5/notebooks.sbatch.
It replays ONE source (meta-validation) episode step by step and plots the committed result tables. No HCC treated value is read."""
import os
import nbformat as nbf

OUT = os.environ.get("R5_NB_OUT", os.path.dirname(os.path.abspath(__file__)))
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# R5 튜토리얼 — 다음에 어떤 약물을 측정할까?

**질문:** 새 세포주에서 약물 반응(978개 유전자의 RNA 변화)을 몇 개만 측정할 수 있다면, 어떤 약물을 먼저 측정해야 나머지 약물의 반응을 더 잘 예측할까?

이 노트북은 **이미 공개된 CMap 2020 자료를 봉인했다가 하나씩 공개하는 모의 실험**이다. 새 wet-lab 실험이 아니다.
HCC 세포주의 처리 후 값은 여기서 읽지 않는다. source(다른 암종) 세포주 22RV1 한 개의 episode를 따라간다.

순서: (1) 초기 4개 공개 → (2) 정책이 다음 4개 선택 → (3) 그 4개 공개 → (4) 고정된 예측기로 나머지(query) 예측 → K=4, 8, 16, 32에서 오차 비교."""),
    code("""%matplotlib inline
import os, sys, json
sys.path.insert(0, os.environ.get("R5_ROOT", "."))
os.chdir(os.environ.get("R5_ROOT", "."))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.perturbation_design_r5.data import Store, make_episode
from src.perturbation_design_r5.models import EpisodeKernels, nu_vector
from src.perturbation_design_r5.replay import load_hyper, run_selection, score_snapshots, readouts_from
st = Store(); hp = load_hyper()
ep = make_episode(st, "meta_val", "22RV1", "22RV1", seed=1)
print("약물 그룹", len(ep.groups), "| 공개 가능 pool", len(ep.pool), "| 봉인 query", len(ep.query))"""),
    md("""## 1. 한 번의 '측정'은 무엇인가?

약물 그룹마다 미리 정한 **처리 well 1개**(그 plate의 DMSO 평균을 뺀 값)가 공개된다. 같은 약물의 다른 반복은 돌려주지 않는다.
plate의 DMSO well들이 얼마나 흔들리는지(`ctl_sd`)는 처리 전에 알 수 있는 정보라서 정책이 쓸 수 있다."""),
    code("""ek = EpisodeKernels(ep, st.bases)
nu = nu_vector(ep, ek, hp)
first4 = ep.rep.iloc[ep.initial][["drug_group", "det_plate", "ctl_sd", "n_ctl"]].copy()
first4["predicted_noise_nu"] = nu[ep.initial]
first4"""),
    md("""## 2. 정책별 선택 비교 (같은 초기 4개에서 시작)

- `random`: 무작위
- `fixed_vr`: 고정 Gaussian 모형에서 pool 전체의 예측 불확실성을 가장 많이 줄이는 4개(관측 값과 무관)
- `proposed_nam`: 측정 잡음 예측(ν)과 여러 모형 가설의 사후 가중치를 함께 쓰는 제안 정책"""),
    code("""budgets = [4, 8, 16, 32]
res = {}
for pol in ["random", "fixed_vr", "proposed_nam"]:
    sel = run_selection(st, ep, pol, hp, budgets, record_rounds=True)
    rows, _ = score_snapshots(st, ep, sel, readouts_from(hp), budgets)
    res[pol] = (sel, pd.DataFrame(rows))
    nxt = sel["S"][4:8]
    print(f"{pol:13s} 다음 4개:", list(ep.groups[nxt]), " 평균 예측 ν =", round(float(nu[nxt].mean()), 3))"""),
    md("""## 3. 공개 수 K에 따른 query 오차 (이 episode 하나)

오차는 query 약물의 978개 유전자 ΔRMSE 평균이다. 낮을수록 좋다. 한 episode는 우연의 영향이 크므로, 결론은 아래 4절의 전체 표로 본다."""),
    code("""fig, ax = plt.subplots(figsize=(5, 3.5))
for pol, (sel, df) in res.items():
    d = df[df.readout == "main"]
    ax.plot(d.K, d.rmse, marker="o", label=pol)
ax.set_xscale("log", base=2); ax.set_xticks(budgets); ax.set_xticklabels(budgets)
ax.set_xlabel("revealed treated wells K"); ax.set_ylabel("query delta-RMSE"); ax.legend(); fig.tight_layout(); plt.show()
pd.concat({p: d[d.readout == "main"].set_index("K").rmse for p, (s, d) in res.items()}, axis=1).round(4)"""),
    md("""## 4. 전체 결과 (commit된 표)

meta-validation 8개 계열 × 시작 seed 10개, 그리고 동결 후 한 번 실행한 HCC 세 계열. AUC는 K=4–32 오차 곡선의 정규화 면적."""),
    code("""base = "manifests/perturbation_design_r5/runs"
mv = pd.read_csv(f"{base}/mval_v1/summary.tsv", sep="\\t")
mv = mv[mv.readout == "main"].set_index("policy")
hc = pd.read_csv(f"{base}/hcc_eval_v1/equal_mean.tsv", sep="\\t")
hc = hc[hc.readout == "main"].set_index("policy")
tab = pd.DataFrame({"meta-val AUC": mv.auc, "HCC3 AUC": hc.auc}).sort_values("meta-val AUC")
tab["meta-val − random"] = tab["meta-val AUC"] - tab.loc["random", "meta-val AUC"]
tab["HCC3 − random"] = tab["HCC3 AUC"] - tab.loc["random", "HCC3 AUC"]
tab.round(4)"""),
    code("""fig, ax = plt.subplots(figsize=(6, 3.5))
t = tab.drop(index="random")
y = np.arange(len(t))
ax.barh(y - 0.2, t["meta-val − random"], 0.4, label="meta-val (source)")
ax.barh(y + 0.2, t["HCC3 − random"], 0.4, label="HCC3")
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y); ax.set_yticklabels(t.index)
ax.set_xlabel("AUC minus random (negative = better)"); ax.legend(); fig.tight_layout(); plt.show()"""),
    md("""## 5. 읽는 법

- 불확실성이 큰 약물만 고르는 정책(`latent_variance`, `total_variance`)과 오차 예측형(`learned_loss`)은 **random보다 나빴다.** 측정 잡음이 큰 well을 고르는 경향과 같은 방향이다(원인은 확정하지 않음).
- `fixed_vr`와 `proposed_nam`은 random보다 약간 낮았지만 차이는 AUC의 약 0.3–0.6%다.
- 사전 등록한 채택 조건(source에서 random·diversity·fixed_vr 모두를 구간 0 미포함으로 넘김)을 제안 정책이 충족하지 못했다. 제안 정책과 더 단순한 `fixed_vr`는 구분되지 않았다.
- 이 결과는 CMap 2020 Level 3, 10 µM·24 h, 978개 유전자, 고정 readout에 한정된다."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
os.makedirs(OUT, exist_ok=True)
nbf.write(nb, os.path.join(OUT, "01_one_episode_and_results.ipynb"))
print("written", OUT)
