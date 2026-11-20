"""Prepare frozen chemical pooling and catalogued DepMap annotations."""

import argparse
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import LINE, table
from src.r11_chemical_probe.models import Interaction


DRIVERS = ("APC", "ARID1A", "ATM", "ATR", "BRAF", "BRCA1", "BRCA2", "CDH1", "CDK4", "CDKN2A",
           "CTNNB1", "EGFR", "ERBB2", "ESR1", "FBXW7", "FGFR1", "FGFR2", "FGFR3", "HRAS", "IDH1",
           "IDH2", "JAK2", "KEAP1", "KIT", "KRAS", "MET", "MLH1", "MSH2", "MTOR", "MYC", "NF1",
           "NFE2L2", "NOTCH1", "NRAS", "PIK3CA", "PIK3R1", "PTEN", "RB1", "RET", "SMAD4", "SMARCA4",
           "STK11", "TERT", "TP53", "TSC1", "TSC2", "VHL")
FLAGS = ("HessDriver", "OncogeneHighImpact", "TumorSuppressorHighImpact")
MODEL = "ModelID"
DEFAULT = "IsDefaultEntryForModel"


def positive(series):
    return series.astype(str).str.casefold().isin(["true", "1", "yes"])


def thaw_depmap(cfg, out):
    start = time.monotonic()
    root = Path(cfg["depmap_root"])
    meta = Path(cfg["r11_assets"]) / "metadata/lines.tsv"
    lines = table(meta)
    model = pd.read_csv(root / "Model.csv", usecols=[MODEL, "RRID"], keep_default_na=False)
    model[LINE] = model.RRID.str.extract(r"(CVCL_[A-Z0-9]+)", flags=re.IGNORECASE)[0].str.upper()
    mapped = model[model[LINE].isin(lines[LINE])].drop_duplicates([MODEL, LINE])
    multiple = mapped.loc[mapped.duplicated(LINE, keep=False), LINE]
    mapped = mapped.loc[~mapped[LINE].isin(multiple)]
    ids = mapped[MODEL].tolist()
    m_file = root / "OmicsSomaticMutations.csv"
    c_file = root / "OmicsCNGeneWGS.csv"
    mutation_frames = []
    for frame in pd.read_csv(m_file, usecols=[MODEL, DEFAULT, "HugoSymbol", *FLAGS], chunksize=125000,
                             low_memory=False):
        selected = frame[frame[MODEL].isin(ids) & positive(frame[DEFAULT])]
        if not selected.empty:
            mutation_frames.append(selected)
    records = pd.concat(mutation_frames, ignore_index=True)
    monitored = records[MODEL].unique()
    flagged = records.loc[records[list(FLAGS)].apply(positive).any(axis=1) & records.HugoSymbol.isin(DRIVERS)]
    flags = pd.crosstab(flagged[MODEL], flagged.HugoSymbol).reindex(
        index=monitored, columns=DRIVERS, fill_value=0).gt(0).astype(float)
    cn_cols = [column for column in pd.read_csv(c_file, nrows=0).columns if column.split(" ")[0] in DRIVERS]
    selected_cn = []
    for frame in pd.read_csv(c_file, usecols=[MODEL, DEFAULT, *cn_cols], chunksize=128):
        selected_cn.append(frame.loc[frame[MODEL].isin(ids) & positive(frame[DEFAULT])])
    cn = pd.concat(selected_cn, ignore_index=True).drop(columns=DEFAULT)
    if cn[MODEL].duplicated().any():
        raise ValueError("DepMap default WGS copy-number entries are ambiguous")
    flags = flags.add_prefix("driver_")
    cn = cn.set_index(MODEL).add_prefix("cn_")
    annotations = lines.merge(mapped[[LINE, MODEL]], on=LINE, how="left", validate="one_to_one")
    annotations = annotations.join(flags, on=MODEL).join(cn, on=MODEL)
    if annotations[list(flags.columns)].nunique().max() < 2 or len(cn) < 3:
        raise ValueError("DepMap modality has insufficient mapping or variation")
    out.mkdir(parents=True, exist_ok=False)
    annotations.to_csv(out / "annotations.tsv", sep="\t", index=False, na_rep="NA")
    write_json(out / "receipt.json", dict(release=root.name, mapped_models=len(ids),
               mutation_profiled=len(monitored), cnv_profiled=len(cn), ambiguous_lines=sorted(multiple.unique()),
               driver_flags=list(FLAGS), driver_genes=list(DRIVERS), seconds=time.monotonic() - start,
               sha256={path.name: sha256(path) for path in [root / "Model.csv", m_file, c_file,
                                                          out / "annotations.tsv"]}))
    print(f"Mapped {len(ids)} DepMap lines; {len(monitored)} mutations, {len(cn)} CNV", flush=True)


def depmap_features(fit, kind):
    frame = pd.read_csv(fit.source.assets / "depmap/annotations.tsv", sep="\t", na_values="NA").set_index(LINE)
    train_lines = sorted({key[1] for key in fit.fit_units})
    control_keys = list(fit.ctrl)
    bank, project = fit.project("pca")
    rna = project(np.stack([bank[key] for key in control_keys]))
    if kind == "C9_id":
        eye = {line: value for line, value in zip(train_lines, np.eye(len(train_lines)))}
        annotation = np.stack([eye.get(key[1], np.zeros(len(eye))) for key in control_keys])
    else:
        copy = frame.filter(like="cn_")
        driver = frame.filter(like="driver_")
        train_copy = copy.loc[train_lines].dropna(how="all")
        neutral = train_copy.mean(axis=0).fillna(0)
        missing = copy.isna().to_numpy()
        pca, scale = Interaction._fit(train_copy.fillna(neutral).to_numpy(), 8)
        copy_z = scale.transform(pca.transform(copy.fillna(neutral).to_numpy()))
        # Missing entries carry an explicit indicator; fill values are numerical only.
        driver_missing = driver.isna().to_numpy()
        vec = np.column_stack((driver.fillna(0).to_numpy(), driver_missing, copy_z, missing))
        annotation = vec[frame.index.get_indexer([key[1] for key in control_keys])]
    return dict(zip(control_keys, np.column_stack((rna, annotation))))


def chem_pool(cfg, checkpoint, out):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    start = time.monotonic()
    r11 = Path(cfg["r11_assets"])
    features = r11 / "features"
    receipt = read_json(features / "receipt.json")
    for name, expected in receipt["checkpoint_files"].items():
        if sha256(checkpoint / name) != expected:
            raise ValueError(f"Chemical checkpoint differs from frozen R11: {name}")
    entities = table(features / "entities.tsv")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, local_files_only=True, add_pooling_layer=False)
    model.eval().requires_grad_(False)
    mean, cls, token_counts = [], [], []
    with torch.inference_mode():
        for i in range(0, len(entities), 32):
            batch = tokenizer(entities.smiles.iloc[i:i + 32].tolist(), padding=True, return_tensors="pt",
                              return_special_tokens_mask=True, truncation=False)
            include = batch["attention_mask"].bool() & ~batch.pop("special_tokens_mask").bool()
            if not include.any(dim=1).all():
                raise ValueError("No nonspecial tokens available")
            hidden = model(**batch).last_hidden_state
            mean.append(((hidden * include[:, :, None]).sum(dim=1) / include.sum(dim=1)[:, None]).numpy())
            cls.append(hidden[:, 0].numpy())
            token_counts.extend(include.sum(dim=1).tolist())
    mean = np.concatenate(mean)
    with np.load(features / "features.npz") as cache:
        delta = float(np.max(np.abs(np.concatenate(cls) - cache["C"])))
        entity_ids = cache["entity"]
    if delta > 1e-5 or not np.array_equal(entity_ids, entities.compound_entity_id):
        raise ValueError("Mean pooling run does not reproduce the source CLS backbone")
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "features.npz", M=mean, entity=entity_ids)
    write_json(out / "receipt.json", dict(checkpoint=receipt["checkpoint_files"], revision=receipt["revision"],
               n_entities=len(entities), tokens_min=min(token_counts), tokens_max=max(token_counts),
               cls_reproduction_max_abs=delta, seconds=time.monotonic() - start, sha256=sha256(out / "features.npz")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chemical-checkpoint", type=Path)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.chemical_checkpoint is not None:
        chem_pool(cfg, args.chemical_checkpoint, args.out)
    else:
        thaw_depmap(cfg, args.out)


if __name__ == "__main__":
    main()
