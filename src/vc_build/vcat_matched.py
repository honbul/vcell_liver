"""Compare the upstream VCAT SMILES architecture on registered continuous HCC panels."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

from .expanded_panel_cv import RECEIPTS, registered_inputs, same_split
from .external import KEY, SOURCE, _validate
from .io import freeze_table
from .limore_corrected import shared_support
from .pdo_transfer_cv import device_cuda
from .response_component_benchmark import (
    BASELINES, EXTERNAL_TRACKS, FORECAST_SCHEMA, PUBLIC, external_inputs, external_reference, source_reference,
)
from .response_components import component_scores, interaction_intervals
from .rna_state_projection import assess, score_external, score_source
from .structured_cv import ASSAYS, with_assay
from .structured_kernel import Standardization, ranked_expression
from .v0plus_cv import panel_targets, validation_roster
from .v0plus_neural import load_auxiliary, masked_standardize
from .yang_hcc_external import require_hash

REVISION = "c122e8c3515bb1281e7210d40e585341f36ca956"
AUTHOR_FILES = {
    "LICENSE": "1822446fe6a60891dc028db940ed55a82905cb6b8753ddbd3fb30311e761a4cb",
    "src/vcat/__init__.py": "1669663209477529709a34f35f353fd0304fe1c9c67d17efcb2b51980ea83f76",
    "src/vcat/model.py": "5032e3c1a4d9ffc3fb8b3258d4bae8d6ccf58f2354eb59cca6b307d84f1cdc36",
    "src/vcat/config.py": "bdb495b04c120f890e87b2c22860a09bc9de2ebe83d2ee86b91a87875651b7dd",
    "src/vcat/data.py": "bd787b6ce2838c8b650b748bed73260937a86b0b42632abccade74e06dfb2bde",
    "src/vcat/utils.py": "0959f1045e310062a76d022b376cf3c49c135c416e3c4fe3ee03d0ef070a50f9",
}
FULL = "vcat_smiles"
ABLATION = "vcat_concat"
PRIMARY = "vcat_structured_average"
ARMS = (FULL, ABLATION, PRIMARY)
CONTROLS = ("structured", FULL, ABLATION)
VPM_EPOCHS = 40
RESPONSE_EPOCHS = 18


def upstream(root):
    for name, checksum in AUTHOR_FILES.items():
        require_hash(root / name, checksum)
    sys.path.insert(0, str(root / "src"))
    from vcat.data import SmilesTokenizer
    from vcat.model import VCATModel

    return VCATModel, SmilesTokenizer


class CachedDrugTokens(nn.Module):
    """Run the upstream GRU once per distinct drug in a response minibatch."""

    def __init__(self, encoder, tokens):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("tokens", tokens)

    def forward(self, drug_ids):
        unique, inverse = drug_ids.unique(sorted=True, return_inverse=True)
        with torch.autocast(device_type="cuda", enabled=False):
            encoded = self.encoder(self.tokens[unique])
        return encoded[inverse]


def model_inputs(x, held, source, auxiliary, tokenizer_type, device):
    measured = auxiliary["chronos"].reindex(x.index)
    supported = measured.columns[measured.notna().sum().ge(40) & measured.std().gt(1e-7)]
    genes = x.columns.intersection(supported).sort_values()
    if len(genes) < 70:
        raise ValueError("Insufficient fit-only CRISPR gene support")
    raw, query = [ranked_expression(frame)[:, x.columns.get_indexer(genes)] for frame in (x, held)]
    reference = Standardization.fit(raw, .05)
    fit_x, held_x = [reference.apply(values, device).unsqueeze(1) for values in (raw, query)]
    chronos, mask = masked_standardize(measured, x.index, genes, device)
    compounds = source[["compound_id", "isomeric_smiles"]].drop_duplicates().set_index(
        "compound_id", verify_integrity=True,
    ).isomeric_smiles.sort_index()
    smiles = compounds.astype(str).tolist()
    length = max(map(len, smiles))
    if length > 256 or compounds.isna().any():
        raise ValueError("Unsupported registered SMILES")
    tokenizer = tokenizer_type(smiles_list=smiles, max_length=length)
    tokens = torch.as_tensor(np.stack([tokenizer.encode(item) for item in smiles]), device=device)
    return fit_x, held_x, -chronos, mask, compounds.index, tokenizer, tokens


def train_vpm(model, expression, truth, available):
    vpm = model.cell_encoder.vpm
    optimizer = torch.optim.AdamW(vpm.parameters(), lr=.001, weight_decay=.001)
    rows = available.any(dim=1).nonzero().flatten()
    vpm.train()
    for epoch in range(VPM_EPOCHS):
        totals = []
        for selection in rows[torch.randperm(len(rows), device=rows.device)].split(64):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, pred = vpm(expression[selection])
            loss = (pred.float()[available[selection]] - truth[selection][available[selection]]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vpm.parameters(), 1.)
            optimizer.step()
            totals.append(loss.detach())
        if not torch.stack(totals).isfinite().all():
            raise RuntimeError("Nonfinite CRISPR pretraining loss")
    return float(torch.stack(totals).mean())


def train_response(model, x, cells, drugs, labels, weights, device):
    vpm = model.cell_encoder.vpm
    vpm.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=.0004, weight_decay=.001,
    )
    history = []
    for epoch in range(RESPONSE_EPOCHS):
        model.train()
        vpm.eval()
        order = torch.randperm(len(cells), device=device)
        total = labels.new_zeros(())
        for at in order.tensor_split((len(order)+255)//256):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(x[cells[at]], drugs[at]).logits
            loss = ((pred.float() - labels[at]).square() * weights[at]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.detach() * len(at)
        value = float(total / len(cells))
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite VCAT response loss")
        history.append(value)
        if (epoch + 1) % 6 == 0:
            print(f"{model.ablation} epoch={epoch+1}/{RESPONSE_EPOCHS} mse={value:.5f}", flush=True)
    return history


@torch.no_grad()
def query_predictions(model, x, cells, drugs):
    model.eval()
    cached = model.drug_encoder
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        example = drugs[:32]
        using_cache = cached(example)
        native = cached.encoder(cached.tokens[example])
        difference = float((using_cache - native).abs().max())
        if difference > 2e-5:
            raise ValueError(f"Coalesced SMILES differs from upstream evaluation: {difference}")
        model.drug_encoder = cached.encoder
        scores = []
        for selection in torch.arange(len(cells), device=cells.device).split(192):
            pred = model(x[cells[selection]], cached.tokens[drugs[selection]]).logits
            scores.append(pred.cpu().numpy())
    finally:
        model.drug_encoder = cached
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)
    return np.concatenate(scores) / 5, difference


def fit_architectures(fit, query, rna, auxiliary, model_type, tokenizer_type):
    _validate(fit, query)
    device = device_cuda()
    torch.set_float32_matmul_precision("high")
    x = rna.loc[sorted(fit.cell_id.unique())]
    q = rna.loc[sorted(query.cell_id.unique())]
    _, anchor, source, _ = panel_targets(fit)
    fit_x, held_x, target, mask, compounds, tokenizer, tokens = model_inputs(
        x, q, source, auxiliary, tokenizer_type, device,
    )

    def index(assay, expression):
        cells = expression.index.get_indexer(assay.cell_id)
        drugs = compounds.get_indexer(assay.compound_id)
        if (cells < 0).any() or (drugs < 0).any():
            raise ValueError("VCAT query needs a registered cell and compound")
        return tuple(torch.as_tensor(v, device=device) for v in (cells, drugs))

    train_cells, train_drugs = index(source, x)
    held_cells, held_drugs = index(query, q)
    labels = torch.as_tensor(source.residual.to_numpy(dtype=np.float32), device=device) * 5
    counts = source.groupby([*SOURCE, "compound_id"]).residual.transform("size")
    weight = 1 / counts
    weight = torch.as_tensor((weight / weight.mean()).to_numpy(dtype=np.float32), device=device)
    result, audit = [], {"fit_genes": fit_x.shape[-1], "fit_cells": len(fit_x), "fit_drugs": len(compounds),
                         "max_smiles_tokens": tokenizer.max_length, "measured_chronos_entries": int(mask.sum())}
    for name, variant in ((FULL, "full"), (ABLATION, "concat_mlp")):
        torch.manual_seed(53)
        model = model_type(
            d_model=64, num_heads=4, num_layers=1, encoder_layers=1, ffn_factor=2., dropout=.1,
            max_genes=fit_x.shape[-1], expr_dim=fit_x.shape[-1], tcs_dim=0,
            drug_feature="smiles", smiles_vocab_size=tokenizer.vocab_size,
            smiles_embedding_dim=64, smiles_gru_layers=2, ablation=variant,
        ).to(device)
        model.drug_encoder = CachedDrugTokens(model.drug_encoder, tokens)
        if name == FULL:
            audit["final_pretrain_loss"] = train_vpm(model, fit_x, target, mask)
            weights = {key: value.cpu().clone() for key, value in model.cell_encoder.vpm.state_dict().items()}
        else:
            model.cell_encoder.vpm.load_state_dict(weights)
        torch.manual_seed(53)
        losses = train_response(model, fit_x, train_cells, train_drugs, labels, weight, device)
        prediction, replay = query_predictions(model, held_x, held_cells, held_drugs)
        result.append(query[KEY].assign(method=name, predicted_value=query.compound_id.map(anchor) + prediction))
        audit[name] = {"train_mse": losses, "cached_vs_native_smiles_max_error": replay}
    return pd.concat(result, ignore_index=True), audit


def predict(args):
    model_type, tokenizer_type = upstream(args.upstream)
    auxiliary, aux_receipt = load_auxiliary(args.auxiliary)
    external = args.mode in EXTERNAL_TRACKS
    if external:
        fit, query, rna, registered = external_inputs(args, args.mode)
        previous, reference = external_reference(args, args.mode, query)
    else:
        original, clean, rna, _ = registered_inputs(args)
        _, fit, check = same_split(original, clean, args.fold, args.mode)
        query = validation_roster(check)
        previous, reference = source_reference(args.cv_archive, args.mode, args.fold)
        registered = RECEIPTS["core"]
    current, diagnostics = fit_architectures(fit, query, rna, auxiliary, model_type, tokenizer_type)
    current = current.assign(track=args.mode, fold=0) if external else with_assay(current, check, args.fold)
    keys = [*KEY, "track", "fold"]
    base = previous[previous.method.eq("structured")][[*keys, "predicted_value"]]
    blend = current[current.method.eq(FULL)].merge(base, on=keys, validate="one_to_one")
    blend = blend.assign(method=PRIMARY, predicted_value=.5 * (blend.predicted_value_x + blend.predicted_value_y))
    rows = pd.concat([previous, current, blend[FORECAST_SCHEMA]], ignore_index=True)[FORECAST_SCHEMA]
    if external:
        shared_support(rows, query, (*BASELINES, *PUBLIC, *ARMS))
    else:
        panel = rows.pivot(index=keys, columns="method", values="predicted_value")
        if set(panel) != {*BASELINES, *ARMS} or not np.isfinite(panel).all(axis=None):
            raise ValueError("VCAT models differ from registered source support")
    freeze_table(rows, args.out, provenance={
        "mode": args.mode, "fold": args.fold, "inputs_sha256": registered, "core_sha256": RECEIPTS["core"],
        "reference_sha256": reference, "diagnostics": diagnostics, "auxiliary": aux_receipt,
        "vcat_revision": REVISION, "upstream_files": AUTHOR_FILES,
    })


def score_vcat(frame, *, external):
    report = assess(frame, external=external, primary=PRIMARY, controls=CONTROLS, require_precision=True)
    if external:
        pairs = {(row["baseline"], row["metric"]): row["ci95"] for row in report["metrics"]["paired_contrasts"]}
        for model in PUBLIC:
            ci = report["across_cell_ci95"][model]
            report["gates"][model] = all(ci[x][0] > 0 for x in ("spearman", "pearson")) and all(
                pairs[model,x][0] > -.005 for x in ("spearman", "ndcg_at_5", "top5_overlap")
            )
        report["interaction_ci95"] = interaction_intervals(frame, PRIMARY, "structured")
        report["gates"]["structured"] &= report["interaction_ci95"][0] > 0
        report["components"] = component_scores(frame, cell_axis="lineage_group")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    inputs = {
        "predict": ("licob", "archive", "cv-archive", "auxiliary", "upstream"),
        "source": ("forecasts",), "external": ("licob", "forecasts", "limore-labels", "licob-labels"),
    }
    for action, files in inputs.items():
        stage = stages.add_parser(action)
        for name in ("original", "core", "out", *files):
            stage.add_argument("--" + name, type=Path, required=True)
        if action == "predict":
            stage.add_argument("--mode", choices=(*ASSAYS,*EXTERNAL_TRACKS), required=True)
            stage.add_argument("--fold", choices=range(3), type=int, default=0)
    args = parser.parse_args()
    if args.stage == "predict":
        predict(args)
    elif args.stage == "source":
        score_source(args, assessor=score_vcat)
    else:
        score_external(args, assessor=score_vcat, arms=ARMS)


if __name__ == "__main__":
    main()
