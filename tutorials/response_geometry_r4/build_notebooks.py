"""Generate the two R4 notebooks. Executed on the server (nbconvert) from the repository root so the stored outputs are real.
01 reads only the small committed tables. 02 uses the server unit store and one frozen checkpoint on VALIDATION drugs
(never test responses) to show what "predict an unseen drug from its structure and the baseline RNA" means. Nothing is trained."""
import os
import nbformat as nbf

OUT = os.environ.get("R4_NB_OUT", os.path.dirname(os.path.abspath(__file__)))
SETUP = '''import os, json, sys
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
ROOT = os.environ.get("R4_ROOT", os.getcwd())
os.chdir(ROOT); sys.path.insert(0, ROOT)
pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8983"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8983", "font.size": 10,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
MAN = "manifests/response_geometry_r4"'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


n1 = nb([
("md", """# 01. R4 결과 표 읽기 (재학습 없음)

저장소에 올라간 작은 표만 읽는다. 서버 자료·GPU가 필요 없다.

- 과제: **학습에 한 번도 쓰지 않은 약물**(모든 세포주에서 제외)의 처리 후 RNA 변화(978개 유전자)를, 처리 전 RNA(X0)와 약물 구조(ECFP)만으로 예측한다.
- 점수 **ΔRMSE**: 예측한 변화와 실제 변화의 차이. **낮을수록 좋다.**
- W0 = 공개 WAVE 원형, W1 = 같은 부품으로 변화량을 직접 예측하도록 바꾼 모델, W2 = W1 + "실측 반응이 비슷한 약물끼리 가깝게" 학습하는 보조 목적."""),
("code", SETUP),
("md", "## 1. 시험 약물 결과 (동결 후 한 번 채점)"),
("code", '''t = pd.read_csv(f"{MAN}/test_summary.tsv", sep="\\t")
piv = t.pivot(index="model", columns="panel", values="rmse_mean")[["source", "HUH7", "JHH5", "JHH7", "hcc_mean"]]
print("test delta RMSE (seed mean; source = family-equal mean over non-HCC families)")
print(piv.sort_values("source").to_string())'''),
("code", '''fig, ax = plt.subplots(figsize=(7.5, 4))
p = piv.sort_values("source")
ax.barh(p.index, p["source"], color=[BLUE if m.startswith("W2") else GREY for m in p.index])
ax.set_xlim(p["source"].min() - 0.03, p["source"].max() + 0.01)
ax.set_xlabel("test delta RMSE, source panel (lower is better)")
plt.tight_layout(); plt.show()'''),
("md", "## 2. 짝 비교 (약물 그룹 bootstrap 95% 구간)\n\n차이 = a − b. 구간 전체가 0보다 작아야 a가 더 낫다고 말한다."),
("code", '''c = pd.read_csv(f"{MAN}/test_contrasts.tsv", sep="\\t")
print(c[c.panel.isin(["source", "hcc_mean"])].to_string(index=False))'''),
("md", "## 3. 반응 크기별 오차\n\n실제 반응이 약한 약물과 강한 약물에서 오차가 어떻게 다른지 본다(채점용 구분이며 설정 선택에 쓰지 않았다)."),
("code", '''s = pd.read_csv(f"{MAN}/test_by_strength.tsv", sep="\\t")
print(s.pivot(index="model", columns="tertile", values="rmse")[["weak", "mid", "strong"]].to_string())'''),
])

n2 = nb([
("md", """# 02. 처음 보는 약물 하나를 예측해 보기 (validation 약물, 재학습 없음)

서버의 unit 저장소와 동결된 checkpoint 하나를 쓴다. **시험(test) 약물의 정답은 열지 않는다.** 대신 개발용 validation 약물에서

1. 예측 함수가 받는 입력이 처리 전 RNA(X0)와 약물 지문뿐임을 확인하고,
2. 예측한 변화와 실제 변화를 나란히 그려 본다."""),
("code", SETUP + '''
os.environ.setdefault("R4_PROC", os.path.join(ROOT, "assets/processed/units_v1"))
os.environ.setdefault("TB_PROC", "/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_transigen_adaptation_b1/assets/processed/cmap2020_v1")
os.environ.setdefault("R4_WAVE", os.path.join(ROOT, "assets/external/WAVE"))
import torch, inspect
from src.response_geometry_r4.data import Store, TrainStats
from src.response_geometry_r4.models import W1DeltaAnchor
fr = json.load(open(f"{MAN}/freeze_manifest.json"))'''),
("code", '''print(inspect.signature(W1DeltaAnchor.predict_delta))   # (self, x0, fp): no treated profile, no teacher
S = Store(); st = TrainStats(S)
name = [k for k in fr["models"] if fr["models"][k][0]["arm"] in ("w1", "w2")]
print("frozen models with the W1/W2 architecture:", name)
r = fr["models"][name[-1]][0]
m = W1DeltaAnchor(); m.load_state_dict(torch.load(r["checkpoint"], map_location="cpu")); m.eval()
print("using", name[-1], r["run_id"])'''),
("code", '''u = S.units
val_jhh5 = S.val[u.cell.to_numpy()[S.val] == "JHH5"]
rms = np.sqrt((S.delta(val_jhh5) ** 2).mean(1))
pick = val_jhh5[np.argsort(-rms)[:1]]            # the validation drug with the largest measured JHH5 response
x0 = torch.from_numpy(S.x0[pick]); fp = torch.from_numpy(S.fp(pick))
pred = m.predict_delta(x0, fp).numpy()[0]; true = S.delta(pick)[0]
print(u.drug_group.iloc[pick[0]], "| RMSE", np.sqrt(((pred - true) ** 2).mean()).round(3), "| Pearson", np.corrcoef(pred, true)[0, 1].round(3),
      "| cell-mean-only RMSE", np.sqrt(((st.cell_mean[S.cell_idx[pick[0]]] - true) ** 2).mean()).round(3))'''),
("code", '''fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(true, pred, s=6, color=BLUE, alpha=0.6)
lim = np.abs(np.r_[true, pred]).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], color=GREY, lw=1); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("measured delta (978 genes)"); ax.set_ylabel("predicted delta")
plt.tight_layout(); plt.show()'''),
("md", "예측 크기가 실제보다 작게 나오는 경향(점이 가로로 눕는 모양)은 이 과제에서 흔하다. 새 약물의 구조만으로는 강한 반응의 크기를 다 맞추기 어렵기 때문이다(가설; 이 노트북에서 검증하지 않음)."),
])

os.makedirs(OUT, exist_ok=True)
nbf.write(n1, os.path.join(OUT, "01_read_results.ipynb"))
nbf.write(n2, os.path.join(OUT, "02_predict_unseen_drug.ipynb"))
print("written", OUT)
