"""Evaluate an exact 10 µM/24 h HepaRG state branch with compound and plate held out."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge

from src.r11_chemical_probe.common import read_json, sha256, write_json
from .state_data import exact_structure, pinned, table
from .state_eval import bootstrap_gain, directory_manifest, grouped_squares, macro_rmse
from .state_model import fingerprints, max_similarity

DATA_SHA256 = "658090ed4e6425f22b578bf2f4a050ff82535c594a96048439e466706c22e61d"
PANEL_SHA256 = "a604ee936c790784215af9f626916b959c6f897965c7e90f501297eb3ba1d59e"
META_SHA256 = "06abe17a0c15992442e627850c7e9be30c107f7b3e213fa1c5024a3623c87712"
ARMS = ("zero_change", "fit_global", "structure_nearest", "tanimoto_ridge")


def bucket(text):
    key = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(key[:4], "little") % 5


def partition_meta(source):
    cols = ["physical_well_id", "plate", "position", "parent_key", "smiles", "time_h", "dose_uM", "scope"]
    rows = table(pinned(source / "rows.tsv", META_SHA256), usecols=cols)
    rows = rows.loc[rows.dose_uM.astype(float).eq(10) & rows.time_h.eq("24") & rows.scope.eq("donor")
                    & rows.parent_key.ne("") & rows.smiles.ne("")].copy()
    all_smiles = {s: exact_structure(s) for s in rows.smiles.unique()}
    structure = pd.DataFrame(rows.smiles.map(all_smiles).tolist(), index=rows.index, columns=["compound", "group"])
    rows = rows.join(structure)
    rows = rows.loc[rows.compound.ne("")].copy()
    if rows.physical_well_id.duplicated().any() or rows.duplicated(["plate", "position"]).any():
        raise ValueError("Duplicated HepaRG physical samples")
    held_plates = rows.plate.map(lambda p: bucket(f"VCB02-HepaRG-plate-{p}")).eq(0)
    held_parents = rows.group.map(lambda p: bucket(f"VCB02-HepaRG-parent-{p}")).eq(0)
    role = np.select([~held_parents & ~held_plates, held_parents & held_plates], ["fit", "score"], default="unused")
    return rows.assign(role=role, matrix_row=rows.index)


def read(source, rows):
    axis = table(pinned(source / "genes.tsv", PANEL_SHA256))
    # Indexed into the previously public count-derived contrasts, never a learned array.
    matrix = np.load(source / "delta.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (31536, len(axis)) or matrix.dtype != np.float32:
        raise ValueError("HepaRG raw count-derived matrix disagrees with its fixed receipt")
    values = matrix[rows.matrix_row.astype(int).to_numpy()]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite HepaRG state contrasts")
    return values, axis.ensembl_id.to_numpy()


def train(args):
    rows = partition_meta(args.source)
    fit = rows.loc[rows.role.eq("fit")]
    query = rows.loc[rows.role.eq("score")]
    if min(fit.group.nunique(), query.group.nunique()) < 20:
        raise ValueError("Insufficient distinct held-plate and held-compound HepaRG support")
    if set(fit.plate) & set(query.plate) or set(fit.group) & set(query.group):
        raise ValueError("HepaRG split leaks compounds or plate control blocks")
    source_ref = read_json(args.source / "receipt.json")
    if source_ref["files"]["delta.npy"] != DATA_SHA256 or sha256(args.source / "delta.npy") != DATA_SHA256:
        raise ValueError("Count-derived HepaRG contrasts changed")
    x, genes = read(args.source, fit)
    by_compound = pd.DataFrame(x, index=pd.MultiIndex.from_frame(fit[["compound", "plate"]]))
    by_compound = by_compound.groupby(level=[0, 1], sort=True).mean().groupby(level=0, sort=True).mean()
    pca = PCA(n_components=8, svd_solver="randomized", random_state=1622).fit(by_compound)
    scale = np.sqrt(pca.explained_variance_).clip(.001)
    y = (by_compound.to_numpy() @ pca.components_.T) / scale
    f = fingerprints(by_compound.index)
    q = fingerprints(query.compound)
    kernel = KernelRidge(alpha=.2, kernel="precomputed").fit(max_similarity(f, f), y)
    similarity = max_similarity(q, f)
    pred = {
        "zero_change": np.zeros((len(q), len(scale)), dtype=np.float32),
        "fit_global": np.tile(y.mean(0), (len(q), 1)),
        "structure_nearest": y[similarity.argmax(axis=1)],
        "tanimoto_ridge": kernel.predict(similarity),
    }
    args.run.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.run.parent, prefix=".heparg-predict-") as temp:
        stage = Path(temp)
        rows.to_csv(stage / "split.tsv", sep="\t", index=False)
        np.savez_compressed(stage / "forecast.npz", **pred)
        pd.DataFrame({"ensembl_id": genes}).to_csv(stage / "genes.tsv", sep="\t", index=False)
        joblib.dump({"pca": pca, "scale": scale, "fingerprints": f, "response": y, "kernel": kernel},
                    stage / "checkpoint.joblib", compress=3)
        write_json(stage / "receipt.json", {
            "source_receipt_sha256": sha256(args.source / "receipt.json"), "protocol_sha256": sha256(args.protocol),
            "code_sha256": sha256(__file__), "query_expression_accessed": False,
            "target": "TempO-Seq 24 h log1p(CP10k) same-plate vehicle contrast",
            "fit_wells": len(fit), "fit_groups": fit.group.nunique(), "fit_plates": fit.plate.nunique(),
            "query_wells": len(query), "query_groups": query.group.nunique(), "query_plates": query.plate.nunique(),
            "files": directory_manifest(stage),
        })
        if args.run.exists():
            raise FileExistsError(args.run)
        stage.rename(args.run)


def freeze_or_score(args):
    stamp = read_json(args.run / "receipt.json")
    if stamp["query_expression_accessed"]:
        raise ValueError("Expression was accessed before the HepaRG forecast freeze")
    for field, digest in stamp["files"].items():
        pinned(args.run / field, digest)
    receipt = sha256(args.run / "receipt.json")
    if args.step == "freeze":
        if (args.run / "freeze.json").exists():
            raise FileExistsError("HepaRG prediction already frozen")
        write_json(args.run / "freeze.json", {"prediction_sha256": receipt})
        return
    if read_json(args.run / "freeze.json")["prediction_sha256"] != receipt:
        raise ValueError("HepaRG forecast mutated after freeze")
    if sha256(args.source / "receipt.json") != stamp["source_receipt_sha256"]:
        raise ValueError("HepaRG score refers to different raw count-derived values")
    query = table(args.run / "split.tsv").loc[lambda rows: rows.role.eq("score")]
    y, genes = read(args.source, query)
    panel = table(args.run / "genes.tsv").ensembl_id.to_numpy()
    if not np.array_equal(genes, panel):
        raise ValueError("HepaRG scoring gene order changed")
    model = joblib.load(args.run / "checkpoint.joblib")
    actual = (y @ model["pca"].components_.T) / model["scale"]
    with np.load(args.run / "forecast.npz", allow_pickle=False) as stored:
        pred = {k: stored[k] for k in ARMS}
    squares = {k: grouped_squares(actual - v, query.group.to_numpy()) for k, v in pred.items()}
    best = min(ARMS[:3], key=lambda k: macro_rmse(squares[k]))
    metrics = []
    for arm in ARMS:
        gain, lo, hi = bootstrap_gain(squares[arm], squares[best])
        metrics.append({
            "arm": arm, "reference": best, "programme_rmse": macro_rmse(squares[arm]),
            "relative_gain": gain, "ci95_low": lo, "ci95_high": hi,
            "programme_sign_agreement": (np.sign(pred[arm]) == np.sign(actual)).mean(),
        })
    out = args.run / "scored"
    out.mkdir(exist_ok=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.tsv", sep="\t", index=False)
    write_json(out / "receipt.json", {"freeze_sha256": sha256(args.run / "freeze.json"), "files": directory_manifest(out)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("predict", "freeze", "score"))
    for name in ("source", "run", "protocol"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    (train if args.step == "predict" else freeze_or_score)(args)
