"""Generate the four beginner notebooks. They are executed on the server (nbconvert) from the repository root,
so that the outputs stored in the .ipynb files are real. Notebooks 01-03 need the server data and checkpoints;
notebook 04 needs only the tables committed under reports/transigen_adaptation_b1/tables."""
import os
import nbformat as nbf

OUT = os.path.dirname(os.path.abspath(__file__))
SETUP = '''import os, json, sys
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
ROOT = os.environ.get("TB_ROOT", os.path.abspath(os.path.join(os.getcwd(), "..", "..")) if os.path.basename(os.getcwd()) == "transigen_adaptation_b1" else os.getcwd())
os.chdir(ROOT); sys.path.insert(0, ROOT)
pd.set_option("display.width", 160); pd.set_option("display.float_format", lambda v: f"{v:.3f}")
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8983", "font.size": 10,
                     "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
MAN = "manifests/transigen_adaptation_b1"
summ = json.load(open(f"{MAN}/splits_summary.json"))["targets"]
TARGET = next(t for t in ["HUH7", "JHH7", "JHH5"] if summ.get(t, {}).get("status", "excluded") != "excluded")
SPLIT = 1
print("example target:", TARGET, "| split seed:", SPLIT)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


n1 = nb([
("md", """# 01. 대조군 하나, 약물 하나: 정답 Δ는 어떻게 만들어지는가

이 노트북은 연구 B의 "정답"이 무엇인지 실제 숫자로 보여 준다.

- **X0**: 약물을 넣지 않고 용매(DMSO)만 넣은 well들의 발현을 같은 plate 안에서 평균한 값이다.
- **X1**: 같은 plate에서 약물 하나를 10 µM로 24시간 처리한 well들의 발현 평균이다.
- **Δ = X1 − X0**: 약물 때문에 생긴 변화다. 모델이 맞혀야 하는 값이다.

발현 값은 L1000 Level 3의 978개 landmark 유전자이며 추가 정규화를 하지 않았다.
여기서는 대상 HCC 세포주의 **적응용(adapt) 약물** 하나를 예로 쓴다. 시험(test) 약물은 열지 않는다."""),
("code", SETUP),
("code", '''from src.transigen_adaptation_b1.data import Data
D = Data(verify_hashes=False)
rows = D.target_rows(TARGET, SPLIT, "adapt", k=8)          # 적응용 약물 8개에 해당하는 행
meta = D.rows.iloc[rows]
print(meta[["cell", "pert_id", "det_plate", "n_trt_wells", "n_ctl_wells"]].head(6).to_string(index=False))'''),
("md", """표의 한 줄이 **(세포주, 약물, plate)** 하나다. `n_ctl_wells`는 그 plate에서 평균에 들어간 DMSO well 수, `n_trt_wells`는 약물 well 수다.

이제 첫 번째 약물 하나를 골라 X0, X1, Δ를 본다."""),
("code", '''pid = meta.pert_id.iloc[0]
r = meta[meta.pert_id == pid].row_id.to_numpy()                 # 이 약물의 plate별 행
x0 = np.asarray(D.x0[r])                                         # (plate 수, 978)
x1 = D.target_x1(r, TARGET, SPLIT, "adapt")                      # 적응용 행만 허용된다
name = D.drugs.set_index("pert_id").cmap_name[pid]
print("약물:", pid, "/", name, "| plate 수:", len(r), "| X0 모양:", x0.shape, "| X1 모양:", x1.shape)
delta_plate = x1 - x0
delta = delta_plate.mean(0)                                      # plate별 Δ를 똑같은 비중으로 평균 -> 조건의 정답
top = np.argsort(-np.abs(delta))[:10]
tab = pd.DataFrame({"gene": D.genes.gene_symbol.to_numpy()[top], "X0 (첫 plate)": x0[0, top], "X1 (첫 plate)": x1[0, top],
                    "Δ (첫 plate)": delta_plate[0, top], "Δ (plate 평균 = 정답)": delta[top]})
tab'''),
("md", """- 값 8~12 근처의 숫자가 발현 수준이고, Δ는 그 차이다. 대부분의 유전자에서 Δ는 0에 가깝고 일부 유전자만 크게 변한다.
- 위 10개는 이 약물에서 변화의 절댓값이 가장 큰 유전자다. 통계적으로 검정한 DEG가 아니다.
- plate가 여러 장이면 plate마다 Δ를 구한 다음 평균한다. plate가 한 장뿐인 세포주에서는 그 한 장의 Δ가 곧 정답이다."""),
("code", '''fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.barh(tab.gene[::-1], tab["Δ (plate 평균 = 정답)"][::-1], color=BLUE, height=0.6)
ax.axvline(0, color=GREY, lw=1); ax.grid(axis="y", visible=False)
ax.set_xlabel("delta = X1 - X0 (Level 3 units)"); ax.set_title(f"{TARGET}, {name}: 10 genes with the largest change", loc="left", fontsize=10)
plt.tight_layout(); plt.show()
print("978개 유전자 전체에서 |Δ|의 중앙값:", round(float(np.median(np.abs(delta))), 3), "| 최댓값:", round(float(np.abs(delta).max()), 3))'''),
("md", """## 정리

- 모델의 입력은 **X0와 약물 구조** 두 가지다. 출력은 X1의 예측이고, 거기서 X0를 빼면 예측한 Δ가 된다.
- 채점은 예측한 Δ와 위에서 만든 정답 Δ를 978개 유전자에서 비교한다(다음 노트북).
- X0를 계산할 때 약물 처리 well의 정보는 쓰지 않는다. X0는 처리 결과를 모르는 상태에서도 구할 수 있어야 하기 때문이다."""),
])

n2 = nb([
("md", """# 02. 입력 두 개만으로 처리 후 RNA 예측하기

여기서는 **B0 모델**을 쓴다. B0는 공개 TranSiGen 구조를 대상 HCC 세포주를 모두 뺀 다른 세포주 자료(source)로 학습한 모델이다. 이 HCC 세포주의 약물 반응은 한 번도 보지 않았다.

추론에 들어가는 것은 두 가지뿐이다.

1. 대조군 발현 X0 (978개 숫자)
2. 약물 구조를 0과 1로 적은 ECFP4 지문 (2,048개 숫자)

측정된 처리군 X1은 모델에 넣지 않는다. X1은 채점할 때 정답으로만 쓴다.
예시는 대상 HCC의 **검증(val) 약물**이다. 시험(test) 약물은 열지 않는다."""),
("code", SETUP),
("code", '''import torch, inspect
from src.transigen_adaptation_b1 import tsg_adapter as T
from src.transigen_adaptation_b1.data import Data, condition_delta
from src.transigen_adaptation_b1.adapt_target import source_checkpoint
from src.transigen_adaptation_b1.evaluate import per_condition
D = Data(verify_hashes=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = source_checkpoint(42)
model, meta_ck = T.load_checkpoint(ck["checkpoint"], dev, seed=42)
print("B0 checkpoint:", ck["run_id"], "| best epoch:", ck["best_epoch"])
print("추론 함수의 인자:", list(inspect.signature(T.predict_x1_mean_path).parameters))'''),
("md", "추론 함수의 인자에 처리군(X1)이 없다는 것을 확인했다(`batch`는 한 번에 계산할 행 수일 뿐이다). 이제 검증 약물들의 X0와 지문을 넣는다."),
("code", '''rows = np.sort(D.target_rows(TARGET, SPLIT, "val"))
meta = D.rows.iloc[rows]
x0 = np.asarray(D.x0[rows]); fp = D.fp[meta.drug_idx.to_numpy()]
print("입력 X0:", x0.shape, "| 입력 지문:", fp.shape, "| 지문에서 1인 자리 수(첫 약물):", int(fp[0].sum()))
x1_hat = T.predict_x1_mean_path(model, torch.tensor(x0, device=dev), torch.tensor(fp, dtype=torch.float32, device=dev)).cpu().numpy()
print("출력 X1 예측:", x1_hat.shape)
cond, d_hat = condition_delta(meta, x1_hat - x0)                  # plate 평균 -> 조건별 예측 Δ
x1 = D.target_x1(rows, TARGET, SPLIT, "val")                       # 여기서 처음 정답을 읽는다 (채점용)
_, d_true = condition_delta(meta, x1 - x0)
pc = per_condition(d_hat, d_true)
pc.insert(0, "pert_id", cond.pert_id)
print("검증 약물 수:", len(pc), "| 평균 ΔRMSE:", round(pc.delta_rmse.mean(), 4), "| 평균 조건별 ΔPearson:", round(pc.delta_pearson.mean(), 4))
print("같은 약물에서 무변화(Δ=0)의 평균 ΔRMSE:", round(float(np.sqrt((d_true ** 2).mean(1)).mean()), 4))
pc[["pert_id", "delta_rmse", "delta_pearson", "top20_direction", "truth_rms", "pred_rms"]].head(8)'''),
("md", """- `delta_rmse`는 약물 하나에서 978개 유전자의 예측 Δ와 정답 Δ가 평균적으로 얼마나 다른지다. 작을수록 좋다.
- `delta_pearson`은 같은 약물 안에서 유전자별 변화의 모양이 얼마나 비슷한지다. 클수록 좋다.
- `truth_rms`와 `pred_rms`는 실제 반응과 예측 반응의 크기다. 예측이 실제보다 훨씬 작으면 "거의 안 변한다"고 답한 것이다.
- 무변화 기준선과 꼭 비교한다. Δ가 원래 작은 자료에서는 아무것도 예측하지 않아도 RMSE가 낮게 나온다."""),
("code", '''i = int(np.argmax(pc.truth_rms.to_numpy()))                       # 실제 반응이 가장 큰 검증 약물
top = np.argsort(-np.abs(d_true[i]))[:10]
print("약물:", cond.pert_id[i], "/", D.drugs.set_index("pert_id").cmap_name[cond.pert_id[i]])
display(pd.DataFrame({"gene": D.genes.gene_symbol.to_numpy()[top], "정답 Δ": d_true[i, top], "예측 Δ": d_hat[i, top]}))
fig, ax = plt.subplots(figsize=(4.2, 4.2))
ax.scatter(d_true[i], d_hat[i], s=9, color=BLUE, alpha=0.55, linewidths=0)
lim = float(np.abs(d_true[i]).max()) * 1.05
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1); ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_xlabel("measured delta"); ax.set_ylabel("predicted delta (B0)")
ax.set_title(f"978 genes, one validation drug (Pearson {pc.delta_pearson[i]:.2f})", loc="left", fontsize=10)
plt.tight_layout(); plt.show()'''),
("md", """## 재구성과 예측을 섞지 않기

TranSiGen에는 측정된 X1을 encoder에 넣어 다시 만들어 내는 **재구성 경로**가 있다. 학습에는 필요하지만, 그 출력은 정답을 보고 만든 것이라 예측이 아니다.
이 연구의 채점은 위에서 쓴 `predict_x1_mean_path`의 출력만 사용한다."""),
])

n3 = nb([
("md", """# 03. 작은 적응 실험: 범용(B0) · HCC-only(B1) · 미세조정(B2)

같은 HCC 세포주에서 **약물 8개**의 실측 반응만 학습에 쓸 수 있다고 하자. 세 가지 방법을 비교한다.

| 이름 | 출발점 | HCC 약물 8개로 학습 |
|---|---|---|
| B0 | source로 학습한 모델 | 하지 않음 |
| B1 | 무작위 초기값 | 함 |
| B2 | B0 | 함 (전체 파라미터 미세조정) |

이 노트북은 방법을 보여 주는 축소판이다(300 update, seed 하나, 검증 약물로 채점). 본 실험의 수치는 `RESULTS.md`에 있다. 시험(test) 약물은 여기서도 열지 않는다."""),
("code", SETUP),
("code", '''import torch
from src.transigen_adaptation_b1 import tsg_adapter as T
from src.transigen_adaptation_b1.data import Data
from src.transigen_adaptation_b1.adapt_target import TargetSet, source_checkpoint
D = Data(verify_hashes=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ts = TargetSet(D, TARGET, SPLIT, 8, dev)                          # 적응 약물 8개 + 검증 약물
sel = json.load(open(f"{MAN}/adapt_selection.json"))["selected"]
lr = {arm: sel[f"{TARGET}|{SPLIT}|{arm}"]["lr"] for arm in ["B1_scratch", "B2_finetune"]}
print("적응 행:", len(ts.rows_a), "(약물", ts.n_adapt_conditions, "개) | 검증 약물:", len(ts.cond_v), "| 본 실험에서 고른 학습률:", lr)'''),
("code", '''def run(arm, updates=300, every=25, seed=42):
    if arm == "B2_finetune":
        model, _ = T.load_checkpoint(source_checkpoint(seed)["checkpoint"], dev, seed=seed)
    else:
        model = T.build_model(seed, dev)
    T.seed_all(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr[arm], weight_decay=T.WEIGHT_DECAY)
    n = len(ts.rows_a); bs = min(32, n); gen = torch.Generator().manual_seed(seed)
    curve = [(0, ts.val_score(model)[0])]
    for step in range(1, updates + 1):
        b = torch.randperm(n, generator=gen)[:bs].to(dev)
        T.upstream_train_step(model, opt, ts.x0_a[b], ts.x1_a[b], ts.fp_a[b])   # 공개 코드의 loss를 그대로 사용
        if step % every == 0:
            curve.append((step, ts.val_score(model)[0]))
    return pd.DataFrame(curve, columns=["update", "val_delta_rmse"])

c1, c2 = run("B1_scratch"), run("B2_finetune")
b0 = c2.val_delta_rmse.iloc[0]                                    # update 0의 B2는 B0와 같은 모델이다
no_change = float(np.sqrt((ts.truth_v ** 2).mean(1)).mean())
pd.DataFrame({"update": c1["update"], "B1 (처음부터)": c1.val_delta_rmse, "B2 (미세조정)": c2.val_delta_rmse}).iloc[::2]'''),
("code", '''fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(c1["update"], c1.val_delta_rmse, color=ORANGE, label="B1 scratch")
ax.plot(c2["update"], c2.val_delta_rmse, color=BLUE, label="B2 fine-tune")
ax.axhline(b0, color=AQUA, lw=1.5, ls="--", label="B0 (no adaptation)")
ax.axhline(no_change, color=GREY, lw=1.5, ls=":", label="no change (delta = 0)")
ax.set_yscale("log"); ax.set_xlabel("optimizer updates"); ax.set_ylabel("validation delta RMSE (log scale)")
ax.set_title(f"{TARGET}: adaptation with 8 drugs, scored on validation drugs", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.show()
print(f"B0 {b0:.4f} | 무변화 {no_change:.4f} | B1 최저 {c1.val_delta_rmse.min():.4f} | B2 최저 {c2.val_delta_rmse.min():.4f}")'''),
("md", """## 읽는 법

- 세로축은 로그 척도다. B1은 발현의 절댓값(약 8~12)을 처음부터 다시 배워야 해서 출발점의 오차가 매우 크다.
- B2의 update 0 값이 B0다. 선이 그 아래로 내려가면 HCC 자료로 적응해서 좋아진 것이고, 올라가면 나빠진 것이다.
- 회색 점선(무변화)보다 위에 있는 모델은 "아무 변화도 없다"고 답하는 것보다 못하다.
- 본 실험에서는 검증 점수가 가장 좋은 update의 모델을 고른다. 그래서 검증 약물도 실제로 측정해야 하는 비용이다. "8개만 측정했다"고 말하려면 검증 약물 수를 함께 적어야 한다.
- 한 번의 축소 실행이다. 결론은 3개 분할 × 3개 seed의 시험 결과(`RESULTS.md`)로 판단한다."""),
])

n4 = nb([
("md", """# 04. 저장된 결과로 표와 학습곡선 다시 만들기

이 노트북은 저장소에 commit된 표만 읽는다. 서버 자료나 GPU가 없어도 실행된다.

- `reports/transigen_adaptation_b1/tables/test_per_condition.tsv.gz`: 시험 약물 하나, 모델 하나, seed 하나마다 한 줄씩 있는 점수표
- `tables/primary_delta_rmse.tsv`: 보고서의 주지표 표
- `tables/learning_curve_k.tsv`: 적응 약물 수 k에 따른 점수

978개 유전자의 예측 배열 전체는 용량 때문에 서버에 있다. 경로와 SHA-256은 `manifests/transigen_adaptation_b1/final_predictions.json`에 있다."""),
("code", '''import os, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
if os.path.basename(os.getcwd()) == "transigen_adaptation_b1": os.chdir(os.path.join("..", ".."))
os.chdir(os.environ.get("TB_ROOT", os.getcwd()))
T = "reports/transigen_adaptation_b1/tables"
pd.set_option("display.width", 160); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df",
                     "grid.linewidth": 0.6, "font.size": 10, "lines.linewidth": 2, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
pc = pd.read_csv(f"{T}/test_per_condition.tsv.gz", sep="\\t", dtype={"seed": str})
print(len(pc), "줄 |", pc.target.nunique(), "개 대상 |", pc.model.nunique(), "개 모델·기준선")
pc.head(3)'''),
("md", """## 1. 주지표를 직접 다시 계산하기

순서가 중요하다. **약물별 ΔRMSE → 그 분할·seed의 시험 약물 평균 → seed 평균 → 분할 평균.**
seed 세 개의 예측을 먼저 평균한 ensemble은 다른 값이므로 이름에 `_ensemble3`가 붙은 줄로 따로 저장돼 있다. 아래 계산에서는 뺀다."""),
("code", '''main = pc[(pc.main == 1) & ~pc.model.str.endswith("_ensemble3")]
per_run = main.groupby(["target", "model", "split_seed", "seed"]).delta_rmse.mean()      # 분할·seed마다 시험 약물 평균
mine = per_run.groupby(["target", "model", "split_seed"]).mean().groupby(["target", "model"]).mean().unstack(0)
saved = pd.read_csv(f"{T}/primary_delta_rmse.tsv", sep="\\t").pivot(index="model", columns="target", values="score")
order = [m for m in ["no_change", "target_adapt_mean", "source_same_drug_mean", "source_ridge", "target_ridge", "B0", "b0_mean_offset",
                     "b0_mean_offset_shrunk", "B1_scratch", "B2_finetune"] if m in mine.index]
print("다시 계산한 값과 저장된 표의 최대 차이:", float((mine.loc[order] - saved.loc[order]).abs().max().max()))
mine.loc[order]'''),
("md", """## 2. 두 가지 주 대비: B2 − B0, B2 − B1

값이 **음수면 B2가 더 좋다**(RMSE는 작을수록 좋다). 구간은 같은 대상 안에서 약물 그룹을 다시 뽑은 bootstrap 95% 구간이다.
이 구간은 "이 세포주에서 다른 약물을 뽑았다면"에 대한 것이며, 다른 HCC 세포주로의 일반화를 말해 주지 않는다."""),
("code", '''con = pd.read_csv(f"{T}/contrasts.tsv", sep="\\t")
con[con.metric == "delta_rmse"][["target", "contrast", "diff", "ci_lo", "ci_hi"]].reset_index(drop=True)'''),
("md", "## 3. 상관은 약물별로 계산해서 평균한다\n\n모든 약물과 유전자를 한 줄로 이어 붙여 계산한 pooled 상관은 다른 숫자다. 여기서는 약물 하나마다 978개 유전자의 상관을 구한 뒤 평균한다. 무변화(Δ=0)처럼 예측이 상수이면 상관이 정의되지 않으므로 빈칸(NaN)으로 둔다."),
("code", '''sec = main.groupby(["target", "model", "split_seed", "seed"])[["delta_pearson", "centred_pearson", "top20_direction"]].mean()
sec.groupby(["target", "model", "split_seed"]).mean().groupby(["target", "model"]).mean().unstack(0).loc[order].round(3)'''),
("md", "## 4. 학습곡선: 적응 약물 수 k를 바꾸면\n\n첫 번째 분할에서만 k = 8, 16, 32를 실행했다(pool에 없는 크기는 실행하지 않았다). 학습률은 k=16에서 고른 것을 그대로 썼다."),
("code", '''cur = pd.read_csv(f"{T}/learning_curve_k.tsv", sep="\\t")
targets = sorted(cur.target.unique())
fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.6), squeeze=False)
style = {"B2_finetune": (BLUE, "-", "B2 fine-tune"), "B1_scratch": (ORANGE, "-", "B1 scratch"), "b0_mean_offset": (YELLOW, "-", "B0 + mean offset"),
         "target_adapt_mean": (AQUA, "-", "target adapt mean"), "B0": (GREY, "--", "B0 (no adaptation)")}
for ax, t in zip(axes[0], targets):
    d = cur[cur.target == t]
    for m, (c, ls, lab) in style.items():
        x = d[d.model == m].sort_values("k")
        ax.plot(x.k, x.delta_rmse, color=c, ls=ls, marker="o", ms=5, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(d.k.unique())); ax.set_xticklabels(sorted(d.k.unique()))
    ax.set_yscale("log"); ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("adapt drugs k")
axes[0][0].set_ylabel("test delta RMSE (log scale)"); axes[0][-1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
cur.pivot_table(index=["target", "model"], columns="k", values="delta_rmse").round(4)'''),
("md", """## 읽을 때 주의할 점

- JHH7 그림에서 B2 선은 B0 점선과 완전히 겹친다. 이 세포주에서는 미세조정이 검증 점수를 한 번도 개선하지 못해 update 0의 모델, 즉 B0가 그대로 선택됐기 때문이다.
- 세로축은 로그 척도이고 세포주마다 범위가 다르다. 세포주 사이의 높낮이를 직접 비교하지 않는다.
- 표의 약물은 "그 HCC 세포주에서 측정되지 않은 것으로 취급한" 약물이다. 다른 세포주(source)에서는 측정된 약물이다. 세상에 처음 나온 약물이 아니다.
- k개 적응 약물 외에 검증 약물도 실제 측정이 필요하다. 그 수는 `RESULTS.md`의 분할 표에 함께 적혀 있다.
- 대상 세포주가 몇 개뿐이므로 평균을 "HCC 전체"의 성능으로 읽지 않는다."""),
])

for name, n in [("01_one_control_one_drug", n1), ("02_public_model_inference", n2), ("03_adaptation_demo", n3), ("04_read_results", n4)]:
    nbf.write(n, os.path.join(OUT, name + ".ipynb"))
    print("wrote", name)
