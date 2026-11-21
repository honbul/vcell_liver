"""Fit and seal the fixed R14 cross-organ study before scoring outcomes."""

import argparse
from hashlib import sha256 as hash_bytes
from pathlib import Path

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r11_chemical_probe.dataset import CONTROL, GROUP, KEY, LINE, load_data, table
from src.r11_chemical_probe.metrics import Contrast
from src.r11_chemical_probe.report import align, align_pool
from src.r12_multiaxis.data import select_panel
from src.r12_multiaxis.split import eligible_pools
from src.r12_multiaxis.stage import load_staged
from src.r13_optimization.objectives import GainScores, center

from .transfer_models import ALPHAS, ARMS, GAINS, NEIGHBORS, chemistry, predict_grid, scaffold


SNU = "CVCL_0366"
C3A = "CVCL_1098"
WELL = KEY[0]
DS = ("sciplex3", "mixseq", "heparg", "heparg_284321")
EXTERNAL = ("mixseq", "gse306429", "emeraldbay")
PROTOCOL = "09774371b5965afbee825e37f955756e90b84700985605a01d1f6c6230c7f63c"
MIN = 3


def files(folder):
    return {p.name: sha256(p) for p in folder.iterdir() if p.is_file() and p.name != "receipt.json"}


def checked(root, receipt, name):
    path = root / name
    if sha256(path) != receipt["files"][str(name)]:
        raise ValueError(f"Hash mismatch: {path}")
    return path


def label(rows, book):
    rows = rows.copy()
    smiles = {s: chemistry(s) for s in rows.smiles.unique()}
    rows["parent_key"] = rows.smiles.map(lambda x: smiles[x][0] if smiles[x] else "")
    if rows.parent_key.eq("").any():
        raise ValueError("Frozen Tahoe sample with unresolved chemistry")
    rows["source"] = np.select([rows[LINE].eq(SNU), rows[LINE].eq(C3A)], ["snu", "tahoe_hepatic"], "tahoe_other")
    rows["assay_id"] = "tahoe"
    rows["position"] = book.loc[rows[WELL], "barcode_position"].to_numpy()
    rows["dose_pool"] = rows.dose_uM
    if not np.array_equal(book.loc[rows[WELL], "plate"], rows.plate_id):
        raise ValueError("Tahoe plate metadata disagree with the coordinate receipt")
    return rows


def eligible(rows):
    return eligible_pools(rows, MIN)


def baseline(train, observed, query):
    by_line = {k: np.mean(observed[idx], axis=0) for k, idx in train.groupby(LINE).indices.items()}
    default = np.mean(observed, axis=0)
    return np.stack([by_line.get(line, default) for line in query[LINE]])


def matrix_by_gene(values, mask, genes, panel):
    match = genes.get_indexer(panel)
    good = np.flatnonzero(match >= 0)
    kept = np.flatnonzero(mask)
    matrix = np.full((len(kept), len(panel)), np.nan, dtype=np.float32)
    matrix[:, good] = values[np.ix_(kept, match[good])]
    return matrix


class Assay:
    def __init__(self, root, name):
        self.name = name
        self.root = root / name
        self.receipt = self.root / "receipt.json"
        self.hash = sha256(self.receipt)
        self.ledger = read_json(self.receipt)
        for name in ("rows.tsv", "genes.tsv"):
            checked(self.root, self.ledger, name)
        self.rows = table(self.root / "rows.tsv")
        self.genes = pd.Index(table(self.root / "genes.tsv").ensembl_id)
        self.values = np.load(self.root / "delta.npy", mmap_mode="r", allow_pickle=False)
        self.control = (np.load(self.root / "control.npy", mmap_mode="r", allow_pickle=False)
                        if (self.root / "control.npy").is_file() else None)
        if self.values.shape != (len(self.rows), len(self.genes)) or not self.genes.is_unique:
            raise ValueError(f"Assay gene/row alignment failed: {name}")
        if self.control is not None and self.control.shape != self.values.shape:
            raise ValueError(f"Control matrix alignment failed: {name}")
        if self.ledger.get("protocol_sha256", PROTOCOL) != PROTOCOL:
            raise ValueError(f"Assay staged under another protocol: {name}")

    def select(self, mask, panel, responses=True):
        rows = self.rows.loc[mask].copy().reset_index(drop=True)
        rows["source"] = "heparg" if self.name.startswith("heparg") else self.name
        rows["assay_id"] = self.name
        rows["time_h"] = rows.time_h.astype(float)
        if self.name.startswith("heparg"):
            # Nominal concentrations may differ by compound at the same designed tier.
            field = next((key for key in ("dose_rank", "dose_slot", "dose_tier", "dose_order", "dose_index")
                          if key in rows), None)
            if field is not None:
                rows["dose_pool"] = rows[field]
            else:
                rows["dose_pool"] = rows.groupby(["dataset", "parent_key"]).dose_uM.rank(method="dense").astype(int)
        else:
            rows["dose_pool"] = rows.dose_uM
        if "dataset" in rows:
            rows["plate_id"] = rows.dataset.astype(str) + "/" + rows.plate_id.astype(str)
        x = matrix_by_gene(self.values, mask, self.genes, panel) if responses else None
        known = self.genes.get_indexer(panel)
        if self.control is None or not (known >= 0).any():
            control = np.zeros(len(rows))
        else:
            control = self.control[np.ix_(np.flatnonzero(mask), known[known >= 0])].mean(axis=1)
        return rows, x, control

    @property
    def external(self):
        return self.rows.scope.isin(("external", "both"))

    @property
    def donor(self):
        return self.rows.scope.isin(("donor", "both")) & self.rows.parent_key.ne("") & self.rows.smiles.ne("")


def position_prediction(train, reference, query):
    src = train.loc[train[LINE].eq(SNU)].reset_index(drop=True)
    if src.empty:
        return np.zeros((len(query), reference.shape[1]), dtype=np.float32), 0
    z = reference[train[LINE].eq(SNU).to_numpy()]
    metric = Contrast(src, MIN)
    z = center(metric, z).astype(np.float32)
    bank = {}
    for i in metric.active:
        key = int(src.position.iloc[i])
        bank.setdefault(key, []).append(i)
    output = np.zeros((len(query), z.shape[1]), dtype=np.float32)
    supported = 0
    for i, (pos, plate, line) in enumerate(zip(query.position, query.plate_id, query[LINE])):
        if line != SNU:
            continue
        donors = [j for j in bank.get(pos, []) if src.plate_id.iloc[j] != plate]
        if donors:
            output[i] = np.mean(z[donors], axis=0)
            supported += 1
    return output, supported


def own_control(r11, role, rows, columns):
    ctl_rows, ctl = load_data(r11 / "source", role, columns)
    ix = align_pool(rows, ctl_rows, ["plate_id", LINE])
    if not np.array_equal(ctl_rows.iloc[ix][WELL], rows[f"control_{role}"]):
        raise ValueError("Role-specific Tahoe control differs from history")
    return ctl[ix].toarray()


def merge(parts):
    rows, values, ctl = zip(*parts)
    return pd.concat(rows, ignore_index=True), np.concatenate(values), np.concatenate(ctl)


def id_hash(key):
    return hash_bytes(str(key).encode()).hexdigest()


def subsets(train, query, holding, arms, fixed, exclude_scaffold=False):
    rows, values, controls = train
    allowed = ~rows.parent_key.isin(set(holding.parent_key) | set(query.parent_key))
    allowed &= ~(rows.source.isin(("snu", "tahoe_other", "tahoe_hepatic")) & rows[WELL].isin(holding[WELL]))
    if exclude_scaffold:
        shared = set(holding.smiles.map(scaffold)) - {""}
        allowed &= ~rows.smiles.map(scaffold).isin(shared)
    same_position = rows.source.eq("tahoe_other") & rows.position.isin(holding.position)
    use = allowed & rows.source.isin(ARMS[arms.removesuffix("_nopos").removesuffix("_matched")])
    dropped = int((use & same_position).sum())
    if arms.endswith("_nopos"):
        use &= ~same_position
    if arms.endswith("_matched"):
        cap = int(rows.loc[allowed & rows.source.eq("snu"), "parent_key"].nunique())
        source = rows.loc[use].groupby("source")
        for name, part in source:
            if name != "snu":
                selected = sorted(part.parent_key.unique(), key=id_hash)[:cap]
                use.loc[part.index] &= part.parent_key.isin(selected)
    if fixed is not None:
        use &= ~rows.parent_key.isin(fixed)
    m = use.to_numpy()
    return (rows.loc[m].reset_index(drop=True), values[m], controls[m]), dropped


def validation_loss(predictions, actual, rows):
    g = GainScores(rows, actual, np.zeros_like(actual), MIN)
    return {key: g.points(value, GAINS)[0] for key, value in predictions.items()}


def scoreable_mask(rows):
    return eligible(rows.loc[rows[LINE].eq(SNU)].reset_index(drop=True))


def output_task(dest, rows, panel, predictions, spec, provenance):
    dest.mkdir(parents=True, exist_ok=False)
    rows.to_csv(dest / "rows.tsv", sep="\t", index=False)
    np.save(dest / "panel.npy", panel, allow_pickle=False)
    archive = {}
    for key, value in predictions.items():
        name = f"{key}.npy"
        if value.shape != (len(rows), len(panel)) or not np.isfinite(value).all():
            raise ValueError(f"Invalid frozen output {key}")
        np.save(dest / name, value.astype(np.float32), allow_pickle=False)
        archive[key] = name
    write_json(dest / "receipt.json", dict(protocol_sha256=PROTOCOL, predictions=archive, selected=spec,
                                          implementation={name: sha256(Path(__file__).with_name(name))
                                                          for name in ("transfer.py", "transfer_models.py")},
                                          inputs=provenance, files=files(dest)))


def study(args):
    fold, role = args.fold, args.role
    stage, split_name = (("staged", "partition") if args.split == "compound" else ("staged-scaffold", "scaffold"))
    source = args.assets / stage
    receipt = read_json(source / "receipt.json")
    fit_path = source / f"outer{fold}"
    for name in [f"outer{fold}/{x}" for x in ("fit-data.npy", "fit-indices.npy", "fit-indptr.npy", "fit-rows.tsv", "score-rows.tsv")]:
        checked(source, receipt, name)
    split_file = args.assets / split_name / "split.json"
    split = read_json(split_file)
    pos = read_json(args.positions / "receipt.json")
    position_path = args.positions / "positions.tsv"
    if sha256(position_path) != pos["positions_sha256"]:
        raise ValueError("Tahoe coordinate manifest was replaced")
    book = table(position_path).set_index("sample")
    train, counts = load_staged(fit_path, "fit", receipt["genes"])
    hold = label(table(fit_path / "score-rows.tsv"), book)
    train = label(train, book)
    train["inner"] = train[GROUP].map({int(k): int(v) for k, v in split["inner"][str(fold)].items()})
    if train.inner.isna().any() or set(train.parent_key) & set(hold.parent_key):
        raise ValueError("Incomplete or overlapping historical parent split")
    query = eligible(hold).assign(**{CONTROL: lambda x: x[f"control_{role}"]})
    genes = table(args.r11 / "metadata/genes.tsv").set_index("token_id").ensembl_id
    assays = {name: Assay(args.data, name) for name in DS if (args.data / name / "receipt.json").is_file()}
    if set(assays) != set(DS):
        raise ValueError(f"Required external sources unavailable: {set(DS) - set(assays)}")
    mix = assays["mixseq"]
    external_parents = set(mix.rows.loc[mix.external, "parent_key"])
    external_wells = set(mix.rows.loc[mix.external, WELL])
    if args.split == "compound":
        for key in ("heparg", "heparg_284321"):
            column = assays[key].rows.outer_fold
            if not column.loc[assays[key].donor].isin((0, 1, 2)).all():
                raise ValueError("HepaRG donor lacks a shared held-out fold")

    donor_mask = {}
    for key, assay in assays.items():
        keep = assay.donor & ~assay.rows.parent_key.isin(hold.parent_key)
        if key == "mixseq":
            keep &= ~assay.rows.parent_key.isin(external_parents) & ~assay.rows[WELL].isin(external_wells)
        if key.startswith("heparg") and args.split == "compound":
            keep &= assay.rows.outer_fold.ne(fold)
        donor_mask[key] = keep
    modes = [*ARMS, *(f"{arm}_nopos" for arm in ARMS if arm != "D0"), "D4_matched"]
    trials = {arm: {} for arm in modes}
    positions = np.zeros(len(GAINS))
    for inner in (0, 1, None):
        take = train.inner.ne(inner).to_numpy() if inner is not None else np.ones(len(train), dtype=bool)
        fit_rows = train.loc[take].reset_index(drop=True)
        validation = scoreable_mask(train.loc[~take]) if inner is not None else query
        excluded = train.loc[~take] if inner is not None else hold
        panel = select_panel(counts[take & train[LINE].eq(SNU).to_numpy()], 2000)
        vectors = counts[take][:, panel].toarray()
        own_fit = own_control(args.r11, role, fit_rows, panel)
        own_query = own_control(args.r11, role, validation, panel)
        delta = vectors - own_fit
        parts = [(fit_rows, delta, own_fit.mean(axis=1))]
        parts.extend(assay.select(donor_mask[key], pd.Index(genes.loc[panel])) for key, assay in assays.items()
                     if donor_mask[key].any())
        combined = merge(parts)
        p_pos, support = position_prediction(fit_rows, delta, validation)
        query_control = own_query.mean(axis=1)
        if inner is not None:
            opposite = own_control(args.r11, "B" if role == "A" else "A", validation, panel)
            actual = counts[align(validation, train)][:, panel].toarray() - opposite
            positions += validation_loss({"position": p_pos}, actual, validation)["position"]
        else:
            predictions = {"B0": np.zeros_like(own_query), "B_pos": GAINS[np.argmin(positions)] * p_pos,
                           "B_abs": baseline(fit_rows, delta, validation)}
            selected = {}
            coverage = {}
        for arm in modes:
            (ref, y, ctl), count = subsets(combined, validation, excluded, arm, None, args.split == "scaffold")
            if not ref.source.eq("snu").any():
                raise ValueError("Empty SNU response training set")
            z, similarity = predict_grid(ref, validation, y, ctl, query_control)
            if inner is not None:
                for key, loss in validation_loss(z, actual, validation).items():
                    trials[arm][key] = trials[arm].get(key, 0) + loss
            else:
                coverage[arm] = dict(same_position_rows=count, sources=ref.groupby("source").parent_key.nunique().to_dict(),
                                     rows_by_source=ref.source.value_counts().to_dict(),
                                     measured_genes=sorted(set(np.isfinite(y).sum(axis=0).tolist())))
                for family, settings in (("ridge", ALPHAS), ("neighbor", NEIGHBORS)):
                    key, scale = min(((key, g) for key in settings for g in range(len(GAINS))),
                                     key=lambda item: trials[arm][family, item[0]][item[1]])
                    name = f"{arm}_{family}"
                    loss = float(trials[arm][family, key][scale] / 2)
                    selected[name] = dict(method=family, setting=key, gain=float(GAINS[scale]), inner_loss=loss)
                    predictions[name] = z[family, key] * GAINS[scale]
                    validation[f"{name}_similarity"] = similarity
            print(f"{args.split} fold={fold} role={role} inner={inner} {arm}", flush=True)
    winner = min((a for a in selected if a.split("_")[0] in ARMS and a.count("_") == 1 and not a.startswith("D0")),
                 key=lambda arm: selected[arm]["inner_loss"])
    family = selected[winner]["method"]
    sensitivity = [f"{winner.split('_')[0]}_nopos_{family}", f"D0_{family}"]
    predictions.update(integrated=predictions[winner], matched_D0=predictions[sensitivity[1]],
                       integrated_nopos=predictions[sensitivity[0]])
    selected.update(integrated=winner, integrated_nopos=sensitivity[0], matched_D0=sensitivity[1],
                    position_gain=float(GAINS[np.argmin(positions)]))
    output_task(args.work / "predict" / args.split / f"task{2*fold+int(role=='B')}", validation, panel, predictions, selected,
                dict(split_sha256=sha256(split_file), stage_sha256=sha256(source / "receipt.json"),
                     assay_receipts={k: v.hash for k, v in assays.items()}, same_well_and_parent_excluded=True,
                     mixseq_reserved_parent_groups=len(external_parents), mixseq_reserved_wells=len(external_wells),
                     matched_snu_parent_cap=True, position_supported_rows=support, coverage=coverage))


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "assets", "r11", "positions", "data", "protocol"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--split", choices=("compound", "scaffold"), default="compound")
    parser.add_argument("--fold", type=int, choices=range(3), required=True)
    parser.add_argument("--role", choices=("A", "B"), required=True)
    return parser


if __name__ == "__main__":
    args = cli().parse_args()
    if sha256(args.protocol) != PROTOCOL:
        raise ValueError("R14 protocol differs from prospective commitment")
    study(args)
