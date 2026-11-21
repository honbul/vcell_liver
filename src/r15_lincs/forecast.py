"""Fit R15 on inner parent splits and persist predictions without held outcomes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.goal_tracks.t2_predict import R12_EPOCHS, R12_LRS, run_chemberta
from src.goal_tracks.t2_prnet import fit_model as fit_prnet
from src.goal_tracks.t2_upstream import EPOCHS, SEED, upstream
from src.r11_chemical_probe.common import config, read_json, sha256, write_json
from src.r11_chemical_probe.dataset import GROUP, KEY
from src.r12_multiaxis.data import Fit
from src.r12_multiaxis.network import fit_network

from .design import Bank, FitSource, fold_id
from .estimators import GAINS, candidates, chemical_grid, cross_position, loss


LEGACY_CPU = ("ChemBERTa_R11", "FiLM_R12", "Concat_R12")
LEGACY_GPU = ("chemCPA_P0", "TranSiGen_Tahoe", "XPert_morgan", "PRnet-well-adaptation")
LEGACY = (*LEGACY_CPU, *LEGACY_GPU)
PROPOSED = "R15"


class StructureMap:
    def __init__(self, folder):
        stamp = read_json(folder/"features/receipt.json")
        artifact = folder/"features/chemberta.npz"
        if sha256(artifact) != stamp["chemberta_sha256"]:
            raise ValueError("The structure-only embedding cache changed")
        with np.load(artifact, allow_pickle=False) as z:
            self.keys = pd.Index(z["parents"])
            self.matrix = z["embedding"]

    def pos(self, parents):
        ix = self.keys.get_indexer(parents)
        if (ix < 0).any():
            raise ValueError("Missing chemistry-only embedding")
        return self.matrix[ix]


def split(bank, fold, external=False):
    rows = bank.target()
    if external:
        training = rows.loc[~rows.parent.isin(bank.blocked)]
        outer = rows.iloc[:0]
    else:
        assigned = rows.parent.map(lambda p: fold_id(p, "R15-outer", 3))
        training, outer = rows.loc[assigned.ne(fold)], rows.loc[assigned.eq(fold)]
        extra = bank.auxiliary()
        supplemental = extra.loc[extra.parent.map(lambda p: fold_id(p, "R15-outer", 3)).eq(fold)]
        outer = pd.concat((outer, supplemental), ignore_index=True)
    validation = training.parent.map(lambda p: fold_id(p, f"R15-inner-{fold}", 5)).eq(0)
    return tuple(x.reset_index(drop=True) for x in (training.loc[~validation], training.loc[validation], training, outer))


def pick(metric):
    return min(metric, key=lambda key: (round(metric[key], 9), key))


def position(bank, train, query, role):
    y = bank.response(train, role)
    tr, te = cross_position(train, query, y)
    return y, tr, te


def setup(bank, inner, validation, role):
    y, fit_pos, val_pos = position(bank, inner, validation, role)
    truth = bank.response(validation, role)
    chosen = pick({gain: loss(validation, truth, gain*val_pos) for gain in GAINS})
    return y, truth, fit_pos, val_pos, chosen


def grid_chemical(bank, train, query, y, rules):
    return chemical_grid(train, query, bank.chemistry(train), bank.chemistry(query), y,
                         bank.scenario, bank.transfer(train), bank.transfer(query), rules)


def choose_proposed(bank, train, query, labels, truth, fit_pos, val_pos, position_gain):
    a, b = bank.transfer(train), bank.transfer(query)
    anchor = position_gain*val_pos+b
    rules = candidates(bank.scenario)
    pred = grid_chemical(bank, train, query, labels-position_gain*fit_pos-a, rules)
    settings = {}
    for rule in rules:
        for gain in GAINS:
            settings[(rule, gain)] = loss(query, truth, anchor+gain*pred[rule])
    winner = pick(settings)
    return winner, settings[winner]


def fit_known(args, bank, train, query, y, truth, anchor, model, work, selected=None):
    import torch

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    embeddings = StructureMap(args.assets) if model == "ChemBERTa_R11" else None
    source = FitSource(bank, train, config(args.config), args.assets, args.role, embeddings, y)
    fit = Fit(source, train, query, panel=bank.landmarks.pr_gene_id.to_numpy())

    def score(p):
        return loss(query, truth, p+anchor)

    if model == "ChemBERTa_R11":
        value, _ = run_chemberta(fit, False)
        return ({(): score(value)} if selected is None else value)
    if model == "PRnet-well-adaptation":
        value, _ = fit_prnet(fit, 10., args.prnet)
        return ({(): score(value)} if selected is None else value)
    if model in EPOCHS:
        epochs = EPOCHS[model] if selected is None else (selected[0],)
        values, _ = upstream(model)(fit, work, epochs, score if selected is None else None)
        return ({(e,): v for e, v in values.items()} if selected is None else values)
    rates = R12_LRS if selected is None else (selected[0],)
    rows = {}
    head = "film" if model == "FiLM_R12" else "concat"
    for lr in rates:
        torch.manual_seed(SEED)
        epochs = max(R12_EPOCHS) if selected is None else selected[1]
        fit.cfg = dict(fit.cfg, min_epochs=epochs+1)
        if selected is None:
            fit.query.treated = fit.query.control+truth-anchor
        best, _, curve, _ = fit_network(fit, fit.inputs("pca"), head, "L0", lr, SEED, "cpu",
                                        epochs, patience=selected is None)
        if selected is not None:
            return best["contrast"][2]
        rows.update({(lr, r["epoch"]): r["contrast"] for r in curve if r["epoch"] in R12_EPOCHS})
    return rows


def forecast(args):
    bank = Bank(args.cache, args.protocol, args.scenario)
    inner, validation, training, outer = split(bank, args.fold)
    if set(training.parent) & set(outer.parent):
        raise ValueError("An outer target parent leaked into its fitting universe")
    labels, truth, fit_pos, val_pos, position_gain = setup(bank, inner, validation, args.role)
    raw, final_pos, query_pos = position(bank, training, outer, args.role)
    a, b = bank.transfer(training), bank.transfer(outer)
    final_anchor = position_gain*query_pos+b
    final_labels = raw-position_gain*final_pos-a
    model = args.model
    if model == PROPOSED:
        chosen, objective = choose_proposed(bank, inner, validation, labels, truth, fit_pos, val_pos, position_gain)
        rule, gain = chosen
        estimates = grid_chemical(bank, training, outer, final_labels, [rule])[rule]
        forecast = final_anchor+gain*estimates
        arrays = {"B0": np.zeros_like(forecast), "B_pos": position_gain*query_pos, model: forecast}
        if args.scenario == "B":
            arrays.update(B_cell=b, B_pos_cell=final_anchor)
    else:
        with TemporaryDirectory(prefix=f"r15-{model}-") as work:
            t, v = bank.transfer(inner), bank.transfer(validation)
            observed, anchor = labels-position_gain*fit_pos-t, position_gain*val_pos+v
            scores = fit_known(args, bank, inner, validation, observed, truth, anchor, model, Path(work))
            chosen = pick(scores)
            objective = scores[chosen]
            estimates = fit_known(args, bank, training, outer, final_labels, None, final_anchor, model, Path(work), chosen)
        arrays = {model: final_anchor+estimates}
    selection = dict(model=model, parameters=chosen, position_gain=position_gain, objective=float(objective),
                     train_parents=training.parent.nunique(), fit_groups=sorted(set(training.parent)))
    path = args.out/"predict"/args.scenario/f"f{args.fold}-{args.role}"/model
    write_forecast(args, bank, path, outer, arrays, selection)


def write_forecast(args, bank, path, rows, arrays, selection):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".forecast-") as t:
        out = Path(t)
        metadata = ["study","index",*KEY,GROUP,"hours","quality","is_exemplar","prior_exact","prior_parent"]
        rows[metadata].to_csv(out/"rows.tsv",sep="\t",index=False)
        for name, matrix in arrays.items():
            if matrix.shape != (len(rows), len(bank.landmarks)) or not np.isfinite(matrix).all():
                raise ValueError(f"Nonfinite or misaligned prediction: {name}")
        np.savez_compressed(out/"forecast.npz", **arrays)
        write_json(out/"receipt.json", dict(protocol_sha256=sha256(args.protocol), cache=bank.receipts,
            code_sha256=sha256(__file__), scenario=args.scenario, fold=args.fold, role=args.role, selection=selection,
            files={p.name:sha256(p) for p in out.iterdir()}))
        if path.exists():
            raise FileExistsError(path)
        out.rename(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("cache","config","protocol","assets","prnet","out"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--model",choices=(PROPOSED,*LEGACY),required=True)
    parser.add_argument("--scenario",choices=("A","B"),required=True)
    parser.add_argument("--fold",type=int,choices=range(3),required=True)
    parser.add_argument("--role",choices="AB",required=True)
    forecast(parser.parse_args())


if __name__ == "__main__":
    main()
