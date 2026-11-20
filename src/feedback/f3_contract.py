#!/usr/bin/env python
"""F3 step 1: data contract for the 4-line CRISPRi panel (K562, RPE1, HepG2, Jurkat).

Reads obs/var only via h5py (no expression matrix), plus the per-line pseudobulk QC/npz written
by f3_pseudobulk.py (raw-count check, library size, knockdown). Writes
reports/feedback/F3/data_contract.tsv, data_contract.json, target_overlap.tsv.
"""
import argparse
import itertools
import json
from pathlib import Path

import h5py
import numpy as np

from f3_pseudobulk import CONTROL, LINES, _dec, read_cat

META = {
    "K562": dict(study="Replogle et al. 2022 Cell (essential-gene CRISPRi Perturb-seq)", accession="figshare 20029387 (file 35773219)", modality="CRISPRi", cell_type="chronic myelogenous leukemia"),
    "RPE1": dict(study="Replogle et al. 2022 Cell (essential-gene CRISPRi Perturb-seq)", accession="figshare 20029387 (file 35775606)", modality="CRISPRi", cell_type="hTERT-immortalised retinal pigment epithelium (non-cancer)"),
    "HepG2": dict(study="Nadig et al. 2025 Nat Genet (essential-gene CRISPRi Perturb-seq)", accession="GEO GSE264667", modality="CRISPRi", cell_type="hepatoblastoma-derived (not adult HCC); previously viewed in V2 project"),
    "Jurkat": dict(study="Nadig et al. 2025 Nat Genet (essential-gene CRISPRi Perturb-seq)", accession="GEO GSE264667", modality="CRISPRi", cell_type="T-cell acute lymphoblastic leukemia"),
}
MD5 = {
    "K562": "4f1122ce1c7f13299a68df6459a266d3 (matches figshare)",
    "RPE1": "6a2a9d0d2bf4ec147f4d1104043b268c (local; no public md5)",
    "HepG2": "dfc676e4186b8c8c173ff14e1194cc8a (local; GEO has no checksum)",
    "Jurkat": "dae9fab2a8b0430099a76ce60d0c5405 (local; GEO has no checksum)",
}


def q(a, p):
    return [float(x) for x in np.percentile(a, p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudobulk", default="assets/f3/pseudobulk")
    ap.add_argument("--out", default="reports/feedback/F3")
    ap.add_argument("--min-cells", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pb = Path(args.pseudobulk)
    rows, tsets, tids, gsets, sym2id = {}, {}, {}, {}, {}
    for line, path in LINES.items():
        with h5py.File(path, "r") as f:
            ncell, ngene = f["X"].shape
            obs_cols = [str(c) for c in f["obs"].attrs["column-order"]]
            g_codes, g_cats = read_cat(f, "gene")
            gid_codes, gid_cats = read_cat(f, "gene_id")
            gt_codes, gt_cats = read_cat(f, "gene_transcript")
            gem = f["obs/gem_group"][:]
            umi = f["obs/UMI_count"][:]
            var_ids = _dec(f["var/gene_id"][:])
        tsym = g_cats[g_codes]; tgid = gid_cats[gid_codes]
        ctrl = tsym == CONTROL
        targets, tcount = np.unique(tsym[~ctrl], return_counts=True)
        guides_per_target = {}
        for t, gt in zip(tsym[~ctrl], gt_cats[gt_codes][~ctrl]):
            guides_per_target.setdefault(t, set()).add(gt)
        gpt = np.array([len(guides_per_target[t]) for t in targets])
        ctrl_guides = np.unique(gt_cats[gt_codes][ctrl])
        gems, cpg = np.unique(gem[ctrl], return_counts=True)
        all_gems = np.unique(gem)
        targets_ok = targets[tcount >= args.min_cells]
        tsets[line] = set(targets_ok)
        for t, i in zip(tsym[~ctrl], tgid[~ctrl]):
            sym2id.setdefault(t, {}).setdefault(line, i)
        tids[line] = {sym2id[t][line] for t in targets_ok}
        gsets[line] = set(var_ids)
        r = dict(cell_line=line, **META[line], file=path, md5=MD5[line],
                 n_cells=int(ncell), n_genes_panel=int(ngene), gene_id_type="Ensembl (var index gene_id) + symbol (var gene_name)",
                 panel_rule="author-filtered: genes with mean >0.1 UMI/cell (var mean min observed ~0.10)",
                 obs_columns=",".join(obs_cols), target_column="gene (symbol); gene_id (Ensembl)",
                 guide_column="gene_transcript / sgID_AB (dual-sgRNA pair per TSS)", control_label=CONTROL,
                 n_control_cells=int(ctrl.sum()), n_control_guides=int(len(ctrl_guides)),
                 batch_column="gem_group (10x GEM/lane)", n_gem_groups=int(len(all_gems)),
                 n_gem_groups_with_controls=int(len(gems)),
                 controls_per_gem_group_min_med_max=[int(cpg.min()), float(np.median(cpg)), int(cpg.max())],
                 day_time="not in file metadata (single harvest per screen per publication; not verified here)",
                 umi_count_median=float(np.median(umi)),
                 n_targets_any=int(len(targets)), n_targets_ge_min_cells=int(len(targets_ok)),
                 cells_per_target_min_q25_med_q75_max=q(tcount, [0, 25, 50, 75, 100]),
                 guides_per_target_counts={int(k): int(v) for k, v in zip(*np.unique(gpt, return_counts=True))},
                 n_targets_multi_tss=int((gpt > 1).sum()))
        qc_p = pb / f"{line}_qc.json"
        if qc_p.exists():
            qc = json.loads(qc_p.read_text())
            r.update(raw_integer_counts=qc["raw_integer_counts"], library_size_used=qc["library_size_used"],
                     rowsum_over_UMI_count_pct=qc["rowsum_over_UMI_count_pct"])
        npz_p = pb / f"{line}.npz"
        if npz_p.exists():
            d = np.load(npz_p, allow_pickle=False)
            groups = d["groups"]; n = d["n"]; SL = d["SL"]; self_col = d["self_col"]; c = int(d["ctrl_index"])
            ctrl_lin = SL[c] / n[c]
            kd = []
            for gi, g in enumerate(groups):
                if gi == c or n[gi] < args.min_cells or self_col[gi] < 0:
                    continue
                kd.append(1 - (SL[gi, self_col[gi]] / n[gi]) / ctrl_lin[self_col[gi]])
            kd = np.array(kd)
            r.update(knockdown_source="computed: 1 - mean normalised target-gene expr (perturbed) / (control); not provided in file",
                     n_targets_self_measured=int(len(kd)), knockdown_median=float(np.median(kd)),
                     knockdown_frac_ge_50pct=float((kd >= 0.5).mean()), knockdown_frac_lt_20pct=float((kd < 0.2).mean()))
        rows[line] = r

    lines = list(LINES)
    inter4 = set.intersection(*tsets.values())
    pair = {f"{a}&{b}": len(tsets[a] & tsets[b]) for a, b in itertools.combinations(lines, 2)}
    triple = {"&".join(c): len(set.intersection(*[tsets[x] for x in c])) for c in itertools.combinations(lines, 3)}
    unique = {l: len(tsets[l] - set.union(*[tsets[m] for m in lines if m != l])) for l in lines}
    id_conflicts = sorted(t for t, m in sym2id.items() if len(set(m.values())) > 1)
    ginter = set.intersection(*gsets.values())
    targets_in_genes = {l: len(tsets[l] & {s for s in tsets[l] if sym2id[s][l] in ginter}) for l in lines}
    # double-OOD natural counts: targets in held-out line never perturbed in the other 3
    natural_double_ood = {l: len(tsets[l] - set.union(*[tsets[m] for m in lines if m != l])) for l in lines}
    seen_in_sources = {l: len(tsets[l] & set.union(*[tsets[m] for m in lines if m != l])) for l in lines}
    overlap = dict(min_cells_per_target=args.min_cells, targets_per_line={l: len(tsets[l]) for l in lines},
                   four_way_intersection=len(inter4), pairwise=pair, three_way=triple, line_unique=unique,
                   loco_seen_target_counts=seen_in_sources, loco_natural_double_ood_counts=natural_double_ood,
                   symbol_to_ensembl_conflicts=id_conflicts[:50], n_symbol_conflicts=len(id_conflicts),
                   gene_panel_sizes={l: len(gsets[l]) for l in lines}, gene_panel_4way_intersection=len(ginter),
                   gene_panel_pairwise={f"{a}&{b}": len(gsets[a] & gsets[b]) for a, b in itertools.combinations(lines, 2)},
                   targets_measured_in_4way_gene_panel=targets_in_genes)
    crit = {
        "post_perturbation_scRNA_with_controls_all_lines": all(rows[l]["n_control_cells"] > 1000 for l in lines),
        "raw_counts_all_lines": all(rows[l].get("raw_integer_counts", False) for l in lines),
        "batch_metadata_all_lines": all(rows[l]["n_gem_groups_with_controls"] == rows[l]["n_gem_groups"] for l in lines),
        "same_modality_CRISPRi_all_lines": True,
        "four_way_target_overlap_ge_300": len(inter4) >= 300,
        "gene_panel_intersection_ge_2000": len(ginter) >= 2000,
        "day_time_metadata_in_file": False,
        "heldout_line_never_viewed_before": False,
    }
    verdict = "conditional"
    reasons = ["Day/time not in file metadata (must be taken from the publications)",
               "HepG2 (GSE264667) was viewed in the earlier V2 project: development_previously_exposed, not a blind cohort",
               "Line identity is confounded with lab/library/sequencing depth/screen design (Replogle vs Nadig); LOCO measures transfer across that bundle",
               "K562 essential library is smaller (fewer targets) than RPE1/HepG2/Jurkat"]
    if not all(v for k, v in crit.items() if k not in ("day_time_metadata_in_file", "heldout_line_never_viewed_before")):
        verdict = "ineligible"
    contract = dict(lines=rows, overlap=overlap, criteria=crit, loco_verdict=verdict, verdict_reasons=reasons)
    (out / "data_contract.json").write_text(json.dumps(contract, indent=1, default=str))
    keys = list(rows["K562"].keys()) + [k for k in rows["RPE1"] if k not in rows["K562"]]
    with open(out / "data_contract.tsv", "w") as fh:
        fh.write("field\t" + "\t".join(lines) + "\n")
        for k in keys:
            fh.write(k + "\t" + "\t".join(json.dumps(rows[l].get(k), default=str).strip('"') for l in lines) + "\n")
    with open(out / "target_overlap.tsv", "w") as fh:
        fh.write("set\tn_targets\n")
        for l in lines:
            fh.write(f"{l}\t{len(tsets[l])}\n")
        for k, v in {**pair, **triple}.items():
            fh.write(f"{k}\t{v}\n")
        fh.write(f"ALL4\t{len(inter4)}\n")
        for l in lines:
            fh.write(f"{l}_only\t{unique[l]}\n")
    print(json.dumps(overlap, indent=1)); print(verdict, crit)


if __name__ == "__main__":
    main()
