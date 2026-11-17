"""Generate the two R2 notebooks. They are executed on the server (nbconvert) from the repository root so that the
stored outputs are real. Notebook 01 needs only the small tables committed under manifests/episodic_residual_r2.
Notebook 02 needs the server-side source unit store and one frozen checkpoint; it uses a SOURCE virtual target
(a held-out meta-validation cell line), never an HCC target, and trains nothing."""
import os
import nbformat as nbf

OUT = os.environ.get("R2_NB_OUT", os.path.dirname(os.path.abspath(__file__)))
SETUP = '''import os, json, sys
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
ROOT = os.environ.get("R2_ROOT", os.path.abspath(os.path.join(os.getcwd(), "..", "..")) if os.path.basename(os.getcwd()) == "episodic_residual_r2" else os.getcwd())
os.chdir(ROOT); sys.path.insert(0, ROOT)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8983", "font.size": 10,
                     "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
MAN = "manifests/episodic_residual_r2"
LABEL = {"no_change": "no change", "source_same_drug_mean": "source same-drug mean (f0)", "mean_offset": "f0 + mean offset",
         "shrunk_same_drug_mean": "shrunk f0 (added)", "shrunk_mean_offset": "shrunk f0 + shrunk offset (added)",
         "ecfp_ridge": "ECFP residual ridge", "ecfp_krr": "ECFP Tanimoto KRR", "allfeat_krr": "same-input untrained KRR",
         "generic_cnp": "generic CNP", "residual_adapter": "proposed adapter", "residual_adapter__zero_residual": "proposed, residual set to 0",
         "residual_adapter__shuffle_residual": "proposed, residual rows swapped", "transigen_as_is": "TranSiGen as is",
         "transigen_offset": "TranSiGen + offset", "transigen_full_ft": "TranSiGen full fine-tuning"}
MAIN = ["no_change", "source_same_drug_mean", "mean_offset", "shrunk_same_drug_mean", "shrunk_mean_offset", "ecfp_ridge", "ecfp_krr", "allfeat_krr",
        "generic_cnp", "residual_adapter", "transigen_as_is", "transigen_offset", "transigen_full_ft"]'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


n1 = nb([
("md", """# 01. 결과 표 읽기 (재학습 없음)

이 노트북은 저장소에 올라간 작은 결과 표만 읽는다. 서버 자료나 GPU가 필요 없고 아무것도 다시 학습하지 않는다.

- **source 가상 target**: 161개 source 세포주 중 학습에 쓰지 않은 9개 계열을 "처음 보는 세포주"처럼 취급해 채점한 결과다. 모든 설정 선택은 여기서 끝났다.
- **HCC**: 방법을 동결한 뒤 HUH7·JHH5·JHH7에서 한 번 채점한 결과다. 이 자료는 앞선 연구 B에서 이미 채점된 적이 있어 독립 확인이 아니다(`reports/episodic_residual_r2/EXPOSURE.md`).
- 점수 **ΔRMSE**는 "예측한 RNA 변화와 실제 RNA 변화의 차이"이며 **낮을수록 좋다.**
- K는 새 세포주에서 실제로 실험해 본 약물 수(8, 16, 32)다."""),
("code", SETUP),
("md", "## 1. source 가상 target: 방법별 ΔRMSE"),
("code", '''src = pd.read_csv(f"{MAN}/source_eval_summary.tsv", sep="\\t")
t = src[src.method.isin(MAIN)].pivot(index="method", columns="K", values="delta_rmse").reindex(MAIN)
t.index = [LABEL[m] for m in t.index]
print("meta-validation, 9 held-out cell families, family-equal mean of query delta RMSE (lower is better)")
print(t.to_string())'''),
("code", '''fig, ax = plt.subplots(figsize=(7.2, 4.2))
show = ["source_same_drug_mean", "mean_offset", "shrunk_mean_offset", "generic_cnp", "residual_adapter", "transigen_full_ft"]
cols = [GREY, ORANGE, YELLOW, AQUA, BLUE, "#b05cc6"]
for m, c in zip(show, cols):
    d = src[src.method == m].sort_values("K")
    ax.plot(d.K, d.delta_rmse, marker="o", color=c, label=LABEL[m])
ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32]); ax.set_xticklabels(["8", "16", "32"])
ax.set_xlabel("K (treated drug groups measured in the new cell line)"); ax.set_ylabel("query delta RMSE (lower is better)")
ax.set_title("Source virtual targets (meta-validation)", loc="left")
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)); plt.tight_layout(); plt.show()'''),
("md", """## 2. 약물별 보정이 실제로 무엇을 더했나

제안법의 예측은 세 조각의 합이다: (1) source 같은 약물 평균 f0, (2) support 평균 잔차(공통 offset), (3) 약물별 보정.
아래 표는 (3)을 0으로 두었을 때와, support 잔차의 짝을 약물끼리 바꿔 넣었을 때의 점수를 함께 보여 준다.
plate를 support와 공유하는 query와 공유하지 않는 query도 나눠 본다."""),
("code", '''rows = ["mean_offset", "residual_adapter__zero_residual", "residual_adapter", "residual_adapter__shuffle_residual", "allfeat_krr"]
cols = ["delta_rmse", "delta_rmse_shared_plate", "delta_rmse_disjoint_plate", "centred_pearson", "retrieval_mrr_centred"]
for K in (8, 16, 32):
    d = src[(src.K == K) & src.method.isin(rows)].set_index("method").reindex(rows)[cols]
    d.index = [LABEL[m] for m in d.index]
    print(f"K = {K}"); print(d.to_string()); print()'''),
("md", "## 3. 동결 후 HCC 평가: 세포주별 ΔRMSE와 95% 구간"),
("code", '''hcc = pd.read_csv(f"{MAN}/hcc_results.tsv", sep="\\t")
h = hcc[(hcc.query_set == "test") & hcc.method.isin(MAIN)]
for K in (8, 16, 32):
    d = h[h.K == K]
    cell = d.apply(lambda r: f"{r.delta_rmse:.4f} [{r.delta_rmse_lo:.4f}, {r.delta_rmse_hi:.4f}]", axis=1)
    t = d.assign(cell=cell).pivot(index="method", columns="target", values="cell").reindex(MAIN)
    t.index = [LABEL[m] for m in t.index]
    print(f"K = {K}  (test query drug groups; interval = drug-group bootstrap)"); print(t.to_string()); print()'''),
("code", '''fig, axes = plt.subplots(1, 4, figsize=(13, 4.4), sharey=False)
show = ["source_same_drug_mean", "mean_offset", "shrunk_mean_offset", "generic_cnp", "residual_adapter", "transigen_full_ft"]
cols = [GREY, ORANGE, YELLOW, AQUA, BLUE, "#b05cc6"]
for ax, target in zip(axes, ["HUH7", "JHH5", "JHH7", "EQUAL_MEAN_3"]):
    for m, c in zip(show, cols):
        d = h[(h.target == target) & (h.method == m)].sort_values("K")
        ax.errorbar(d.K, d.delta_rmse, yerr=[d.delta_rmse - d.delta_rmse_lo, d.delta_rmse_hi - d.delta_rmse], marker="o", color=c, capsize=2, label=LABEL[m], linewidth=1.5)
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32]); ax.set_xticklabels(["8", "16", "32"])
    ax.set_title(target if target != "EQUAL_MEAN_3" else "equal-weight mean of 3", loc="left"); ax.set_xlabel("K")
axes[0].set_ylabel("test delta RMSE (lower is better)")
axes[-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5)); plt.tight_layout(); plt.show()'''),
("md", """## 4. 기준선과의 차이 (같은 query 약물에서의 짝 비교)

`diff`가 음수면 그 방법이 기준선보다 낮다(좋다). 구간이 0을 포함하면 방향을 정할 수 없다."""),
("code", '''con = pd.read_csv(f"{MAN}/hcc_contrasts.tsv", sep="\\t")
c = con[(con.query_set == "test") & (con.metric == "delta_rmse") & (con.reference == "mean_offset") & con.method.isin(MAIN)]
for K in (8, 16, 32):
    d = c[c.K == K]
    cell = d.apply(lambda r: f"{r['diff']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]", axis=1)
    t = d.assign(cell=cell).pivot(index="method", columns="target", values="cell").reindex([m for m in MAIN if m != "mean_offset"])
    t.index = [LABEL[m] for m in t.index]
    print(f"K = {K}: method minus 'f0 + mean offset'"); print(t.to_string()); print()'''),
("md", """## 5. 공통 이동을 뺀 지표

ΔRMSE는 모든 약물에 공통인 이동을 맞히기만 해도 좋아진다. 아래는 예측과 정답에서 각각 query 평균을 뺀 뒤의 상관(centred Pearson)과,
"예측이 자기 약물의 정답을 다른 약물의 정답보다 잘 찾는가"(centred MRR, 1에 가까울수록 좋음)다."""),
("code", '''for v in ("centred_pearson", "mrr_centred"):
    t = h[h.K == 16].pivot(index="method", columns="target", values=v).reindex(MAIN)
    t.index = [LABEL[m] for m in t.index]
    print(f"K = 16, {v} (higher is better)"); print(t.to_string()); print()'''),
("md", """## 6. 예측 배열은 어디에 있나

발현 배열은 Git에 올리지 않았다. 서버 경로와 SHA-256만 기록했다."""),
("code", '''pf = json.load(open("predictions/episodic_residual_r2/prediction_files.json"))
print("frozen code commit:", pf["frozen_code_commit"][:10], "| eval job:", pf["slurm_job"], "| failures:", len(pf["failures"]))
for name, e in list(pf["prediction_files"].items()):
    print(f"{name:28s} {e['bytes'] / 1e6:6.1f} MB  n_query={e['n_query']:4d}  sha256={e['sha256'][:16]}...")'''),
])

n2 = nb([
("md", """# 02. 작은 예제: support 16개 → query 약물 하나의 ΔRNA

새 세포주에서 약물 16개만 실험했다고 하자. 17번째 약물의 RNA 변화를 어떻게 예측하는지 실제 숫자로 따라간다.

- 여기서 "새 세포주"는 **source 가상 target**(학습에 쓰지 않은 meta-validation 세포주)이다. HCC 자료는 열지 않는다.
- 아무것도 학습하지 않는다. 동결된 모델 하나(seed 42)를 불러와 계산만 한다.
- 서버의 source 단위 저장소가 필요하다(`experiments/episodic_residual_r2/env.sh`를 읽은 환경에서 실행).

용어:
- **Δ**: 같은 plate의 대조군 평균 대비 약물 처리 후 978개 유전자의 변화.
- **f0**: 다른 세포주들에서 같은 약물이 보인 Δ의 평균(이 세포주와 같은 계열은 뺀다).
- **잔차**: 이 세포주의 실제 Δ에서 f0를 뺀 것. support 16개에서만 알 수 있다."""),
("code", SETUP),
("code", '''import torch
from src.episodic_residual_r2.data import Source, Features
from src.episodic_residual_r2 import methods as M
from src.episodic_residual_r2.suite import load_learned
src = Source("cpu")
CELL, SEED, K = "HCC515", 1, 16
ci = list(src.cell_name).index(CELL)
assert src.cell_role[ci] == "val"                       # a held-out source cell line, not a training one
ep = src.eval_episode(ci, SEED)
inp = ep.inputs(K)                                      # everything a method may see: no query label inside
names = src.groups.drug_group.to_numpy()
print("virtual target:", CELL, "| family:", ep.family, "| support K =", K, "| query drugs:", len(ep.g_q))
print("support drugs:", ", ".join(names[ep.g_s[:K]]))'''),
("md", "## 1. 출발점: 다른 세포주에서 본 같은 약물의 평균(f0)"),
("code", '''rmse = lambda p, y: torch.sqrt(((p - y) ** 2).mean(-1))
truth = ep.y_q                                           # used for scoring only
f0 = M.same_drug_mean(inp)
order = torch.argsort(rmse(f0, truth))
q = int(order[len(order) // 2])                          # the query drug with the median f0 error: a typical case
print("example query drug:", names[ep.g_q[q]], "| source cell lines behind its f0:", int(inp["n0_q"][q]))
print(f"no change      : RMSE {rmse(torch.zeros_like(truth[q]), truth[q]):.4f}")
print(f"f0             : RMSE {rmse(f0[q], truth[q]):.4f}   correlation with truth {np.corrcoef(f0[q].numpy(), truth[q].numpy())[0, 1]:.3f}")'''),
("md", """## 2. support 16개가 알려 주는 것: 잔차와 그 평균(offset)

support 약물마다 `실제 Δ − f0`를 계산한다. 16개의 평균이 "이 세포주는 전반적으로 이만큼 다르다"는 공통 offset이다."""),
("code", '''e = inp["y_s"] - inp["f0_s"]                           # (16, 978) support residuals
b, R = M.split_residual(inp)                             # b: mean residual, R: residual minus the mean
print("support residual size (RMS per drug):", np.round(torch.sqrt((e ** 2).mean(1)).numpy(), 3))
print(f"common offset size (RMS over genes): {torch.sqrt((b ** 2).mean()):.4f}")
print(f"share of support residual energy carried by the common offset: {float((b ** 2).sum() * K / (e ** 2).sum()):.3f}")
mo = M.mean_offset(inp)
print(f"f0 + offset    : RMSE {rmse(mo[q], truth[q]):.4f}")'''),
("md", """## 3. 축소(shrinkage): 적은 표본의 평균은 덜 믿는다

f0는 몇 개 세포주의 평균이고 offset은 16개 약물의 평균이라 둘 다 잡음이 섞여 있다.
source에서 미리 정한 비율로 0 쪽으로 당긴 것이 추가 기준선이다(이 연구에서 계획 뒤에 덧붙인 진단)."""),
("code", '''sh = json.load(open(f"{MAN}/shrink_selection.json"))["selected"][str(K)]
kf, kb = sh["shrunk_mean_offset"]["kappa_f"], sh["shrunk_mean_offset"]["kappa_b"]
n = float(inp["n0_q"][q])
print(f"f0 weight for this drug  = n/(n+{kf:g}) = {n / (n + kf):.3f}     offset weight = K/(K+{kb:g}) = {K / (K + kb):.3f}")
sf = M.shrunk_prediction(inp, sh["shrunk_same_drug_mean"]["kappa_f"], "inf")
so = M.shrunk_prediction(inp, kf, kb)
print(f"shrunk f0              : RMSE {rmse(sf[q], truth[q]):.4f}")
print(f"shrunk f0 + offset     : RMSE {rmse(so[q], truth[q]):.4f}")'''),
("md", """## 4. 제안법: 닮은 support 약물의 잔차를 빌려 온다

동결된 모델은 약물마다 좌표(embedding)를 만든다. query 약물과 좌표가 가까운 support 약물의 "평균을 뺀 잔차"에 큰 가중치를 주어 더한다.
가중치는 정답을 보지 않고 좌표와 λ만으로 닫힌 식으로 계산된다."""),
("code", '''sel = json.load(open(f"{MAN}/model_selection.json"))["final"]["residual_adapter"]["42"]
model = load_learned(sel, src, "cpu")
fmap = Features.load(os.path.join(os.environ["R2_PROC"], "feature_basis.npz"), "cpu")
with torch.no_grad():
    ft = fmap.build(inp, src.FP)
    zs, zq = model.embed(ft, "s"), model.embed(ft, "q")
    m = zs.mean(0, keepdim=True); zs, zq = zs - m, zq - m
    W = (zq @ zs.T).double() @ torch.linalg.inv((zs @ zs.T).double() + model.lam().double() * torch.eye(K, dtype=torch.float64))
    pred = model(inp, ft)
    pred0 = model(inp, ft, zero_residual=True)
w = W[q].numpy()
top = np.argsort(-np.abs(w))[:5]
print(f"lambda = {float(model.lam()):.3f} | variant = {sel['variant']}")
print("five support drugs with the largest weight for", names[ep.g_q[q]])
for i in top:
    print(f"  {names[ep.g_s[i]]:28s} weight {w[i]:+.3f}")
print(f"sum of |weights| = {np.abs(w).sum():.3f}")
print(f"proposed, per-drug part set to 0 : RMSE {rmse(pred0[q], truth[q]):.4f}")
print(f"proposed                         : RMSE {rmse(pred[q], truth[q]):.4f}")'''),
("code", '''fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
for ax, (lab, p, c) in zip(axes, [("f0", f0[q], GREY), ("f0 + mean offset", mo[q], ORANGE), ("proposed", pred[q], BLUE)]):
    ax.scatter(truth[q].numpy(), p.numpy(), s=5, color=c, alpha=0.6)
    lim = float(truth[q].abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#8a8983", linewidth=0.8)
    ax.set_title(f"{lab}: RMSE {rmse(p, truth[q]):.3f}", loc="left"); ax.set_xlabel("measured change (978 genes)")
axes[0].set_ylabel("predicted change"); plt.suptitle(f"{CELL}, query drug {names[ep.g_q[q]]}, K = {K}", x=0.01, ha="left"); plt.tight_layout(); plt.show()'''),
("md", """## 5. 한 약물이 아니라 이 세포주의 모든 query 약물에서

약물 하나의 결과는 우연일 수 있다. 같은 계산을 이 episode의 모든 query 약물에 적용한 평균이다.
9개 계열 전체의 평균은 노트북 01과 `RESULTS.md`에 있다."""),
("code", '''with torch.no_grad():
    cnp = load_learned(json.load(open(f"{MAN}/model_selection.json"))["final"]["generic_cnp"]["42"], src, "cpu")
    pc = cnp(inp, ft)
allq = {"no change": torch.zeros_like(truth), "f0": f0, "f0 + mean offset": mo, "shrunk f0": sf, "shrunk f0 + shrunk offset": so,
        "generic CNP (seed 42)": pc, "proposed, residual 0 (seed 42)": pred0, "proposed (seed 42)": pred}
tab = pd.DataFrame({"mean query RMSE": {k: float(rmse(v, truth).mean()) for k, v in allq.items()},
                    "this drug": {k: float(rmse(v[q], truth[q])) for k, v in allq.items()}})
print(f"{CELL}, support seed {SEED}, K = {K}, {len(ep.g_q)} query drugs"); print(tab.to_string())'''),
("code", '''fig, ax = plt.subplots(figsize=(6.4, 3.4))
v = tab["mean query RMSE"]
ax.barh(range(len(v)), v.values, color=[GREY, GREY, ORANGE, YELLOW, YELLOW, AQUA, BLUE, BLUE], height=0.62)
ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index); ax.invert_yaxis()
ax.set_xlim(v.min() * 0.97, v.max() * 1.01); ax.set_xlabel("mean query RMSE in this one episode (lower is better)")
for i, x in enumerate(v.values):
    ax.text(x, i, f" {x:.4f}", va="center", fontsize=8)
ax.grid(axis="y", visible=False); plt.tight_layout(); plt.show()'''),
("md", """## 읽는 법

- 이 예제는 세포주 하나, support 추출 한 번의 결과다. 방법 간 순서는 `RESULTS.md`의 9개 계열 평균과 HCC 표로 판단한다.
- "제안법"과 "제안법(약물별 항 0)"의 차이가 약물별 보정이 더한 몫이다. 나머지는 f0와 offset(및 그 축소)에서 온다."""),
])

for name, n in (("01_read_results.ipynb", n1), ("02_worked_example_16_supports.ipynb", n2)):
    nbf.write(n, os.path.join(OUT, name))
    print("wrote", name)
