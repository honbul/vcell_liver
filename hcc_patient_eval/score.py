"""Score frozen forecasts against independent tumor annotations and matched controls."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

from .cohort import digest, write_receipt
from .annotations import CODING_CLASSES

matplotlib.use("Agg")
from matplotlib import pyplot as plt


PURITY = ("absolute_purity", "immune_score", "stromal_score")


def checked_files(root, mapping):
    for name, expected in mapping.items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError(f"Frozen artifact mismatch for {name}")


def estimate_annotations(directory):
    scores = pd.read_csv(directory / "estimate_score.gct", sep="\t", skiprows=2, index_col=0)
    scores = scores.drop(columns="Description").T.rename(
        columns={"ImmuneScore": "immune_score", "StromalScore": "stromal_score",
                 "ESTIMATEScore": "estimate_score"})
    # R's make.names may replace TCGA barcode hyphens with dots.
    scores.index = scores.index.str.replace(".", "-", regex=False)
    return scores


def effect(y, exposure, covariates=None):
    combined = pd.concat([y.rename("value"), exposure.rename("exposure"), covariates], axis=1).dropna()
    n = len(combined)
    if n < 20 or combined.exposure.nunique() < 2 or np.ptp(combined.value) < 1e-12:
        return dict(n=n, effect=np.nan, low99=np.nan, high99=np.nan, p=np.nan)
    if combined.exposure.nunique() == 2 and combined.exposure.value_counts().min() < 5:
        return dict(n=n, effect=np.nan, low99=np.nan, high99=np.nan, p=np.nan)
    design = combined.drop(columns="value").astype(float)
    continuous = [x for x in design if x != "exposure"]
    design[continuous] = design[continuous].sub(design[continuous].mean()).div(
        design[continuous].std().replace(0., 1.))
    design = add_constant(design)
    fit = OLS(combined.value, design).fit(cov_type="HC3")
    lo, hi = fit.conf_int(alpha=.01).loc["exposure"]
    return dict(n=n, effect=float(fit.params.exposure), low99=float(lo), high99=float(hi),
                p=float(fit.pvalues.exposure))


def tests(forecasts, annotation, compounds):
    target_text = compounds.drug_target_source_labels.fillna("")
    drug_sets = {
        "FGFR_panel": target_text.index[target_text.str.contains(r"\bFGFR\b", case=False, regex=True)],
        "MET_panel": target_text.index[target_text.str.contains(r"\bMET\b", case=False, regex=True)],
        "VEGFR_panel": target_text.index[target_text.str.contains(r"\bVEGFR\b", case=False, regex=True)],
        "CDK4_6_panel": target_text.index[target_text.str.contains(r"CDK4|CDK6|CDK4/6", case=False, regex=True)],
    }
    cna_pairs = [("FGF19", "FGFR_panel", 2), ("CCND1", "CDK4_6_panel", 2),
                 ("MET", "MET_panel", 2), ("VEGFA", "VEGFR_panel", 2),
                 ("CDKN2A", "CDK4_6_panel", -2)]
    panel_tests, drug_tests, strata, purity = [], [], [], []
    covariates = annotation[list(PURITY)]
    for (arm, prep), table in forecasts.groupby(["arm", "preprocessing"], sort=True):
        scores = table.pivot(index="participant", columns="compound_id", values="value").reindex(annotation.index)
        centered = scores.sub(scores.mean(axis=1), axis=0)
        for gene, panel, status in cna_pairs:
            tested = annotation[f"{gene}_gistic"]
            flag = tested.eq(status).where(tested.notna()).astype(float)
            drugs = drug_sets[panel].intersection(centered.columns)
            if not len(drugs):
                continue
            contrast = centered[drugs].mean(axis=1)
            for adjust, adjusters in (("unadjusted", None), ("purity_immune_stroma", covariates)):
                panel_tests.append(dict(arm=arm, preprocessing=prep, contrast=f"{gene}:{panel}",
                    adjustment=adjust, drugs=len(drugs), positive=int(flag.sum()), **effect(contrast, flag, adjusters)))
        tp53, ctnnb1 = [annotation[f"{gene}_coding_mutation"].astype("boolean") for gene in ("TP53", "CTNNB1")]
        exclusives = ctnnb1.astype("Float64").where(tp53.notna() & ctnnb1.ne(tp53)).astype(float)
        hoshida = annotation.hoshida_subtype
        for drug in centered:
            values = centered[drug]
            for group in ("S1", "S2", "S3", "unassigned"):
                values_in_group = values[hoshida.eq(group)].dropna()
                strata.append(dict(arm=arm, preprocessing=prep, compound_id=drug, subtype=group,
                    n=len(values_in_group), mean=values_in_group.mean(), median=values_in_group.median()))
            subtype_flag = hoshida.map({"S1": 1., "S2": 1., "S3": 0.})
            for name, flag in (("CTNNB1_only_vs_TP53_only", exclusives), ("S1S2_vs_S3", subtype_flag)):
                drug_tests.append(dict(arm=arm, preprocessing=prep, contrast=name, compound_id=drug,
                                       **effect(values, flag, covariates)))
            for marker in PURITY:
                data = pd.concat([values.rename("value"), covariates], axis=1).dropna()
                # Partial Spearman using residuals from linear projections of ranks.
                ranks = data.rank()
                observed = spearmanr(data.value, data[marker]).statistic if data.value.nunique() > 1 else np.nan
                controls = add_constant(ranks.drop(columns=["value", marker]))
                residual_y = OLS(ranks.value, controls).fit().resid
                residual_marker = OLS(ranks[marker], controls).fit().resid
                rho = residual_y.corr(residual_marker) if data.value.nunique() > 1 else np.nan
                purity.append(dict(arm=arm, preprocessing=prep, compound_id=drug, marker=marker,
                                   n=len(data), spearman=observed, partial_rank_rho=rho))
    tested = pd.DataFrame(drug_tests)
    tested["fdr"] = np.nan
    for _, group in tested.dropna(subset=["p"]).groupby(["arm", "preprocessing", "contrast"]):
        tested.loc[group.index, "fdr"] = multipletests(group.p, method="fdr_bh")[1]
    panels = pd.DataFrame(panel_tests)
    panels["fdr"] = np.nan
    for _, group in panels.dropna(subset=["p"]).groupby(["arm", "preprocessing", "adjustment"]):
        panels.loc[group.index, "fdr"] = multipletests(group.p, method="fdr_bh")[1]
    return panels, tested, pd.DataFrame(strata), pd.DataFrame(purity), drug_sets


def plots(output, points, domain, annotations, panels, drug_tests):
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    preps = list(points.preprocessing.unique())
    fig, axes = plt.subplots(len(preps), 2, figsize=(9, 2.7 * len(preps)), layout="constrained")
    for (pca, umap), prep in zip(axes, preps, strict=True):
        data = points.loc[points.preprocessing.eq(prep)]
        for ax, fields in ((pca, ("PC1", "PC2")), (umap, ("UMAP1", "UMAP2"))):
            for name, group in reversed(list(data.groupby("origin"))):
                ax.scatter(group[fields[0]], group[fields[1]], label=name, s=11, alpha=.55)
            ax.set(xlabel=fields[0], ylabel=fields[1], title=prep.replace("_", " "))
    axes[0, 0].legend(frameon=False)
    save(fig, output, "domain_shift")

    joined = domain.join(annotations[list(PURITY)])
    fig, axes = plt.subplots(1, len(PURITY), figsize=(11, 3.5), layout="constrained")
    for ax, marker in zip(axes, PURITY, strict=True):
        for mode, data in joined.groupby("preprocessing"):
            ax.scatter(data[marker], data.nearest_fit_distance, label=mode.replace("_", " "), s=9, alpha=.5)
        ax.set(xlabel=marker.replace("_", " "), ylabel="Nearest model-fit distance")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, output, "purity_distance")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    data = panels.loc[panels.arm.eq("candidate") & panels.preprocessing.str.startswith("native_")].dropna(subset=["effect"])
    for name, group in data.groupby("adjustment"):
        offset = -.12 if name == "unadjusted" else .12
        centers = sorted(data.contrast.unique())
        y = [centers.index(c) + offset for c in group.contrast]
        axes[0].errorbar(group.effect, y, xerr=[group.effect-group.low99, group.high99-group.effect],
                         fmt="o", ms=4, capsize=2, label=name.replace("_", " "))
        axes[0].set_yticks(range(len(centers)), centers)
    axes[0].axvline(0, color="0.6", linestyle=":")
    axes[0].set(xlabel="Panel-centered model score difference", title="Target-panel concordance")
    if len(data):
        axes[0].legend(frameon=False, fontsize=8)
    subset = drug_tests.loc[drug_tests.preprocessing.str.startswith("native_") & drug_tests.arm.ne("drug_mean")]
    for (arm, contrast), group in subset.groupby(["arm", "contrast"]):
        axes[1].hist(group.effect.dropna(), bins=20, histtype="step", label=f"{arm} {contrast}", linewidth=1.2)
    axes[1].set(xlabel="Adjusted per-drug model score difference", ylabel="Drugs", title="Subtype and driver contrasts")
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, output, "biologic_concordance")


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(folder / f"{name}.png", dpi=160)
    plt.close(fig)


def limore_driver_reference(bundle, maf, patient_tests):
    variants = pd.read_csv(maf, sep="\t", usecols=["Tumor_Sample_Barcode", "Hugo_Symbol", "Variant_Classification"])
    rows = pd.read_parquet(bundle / "rows.parquet",
                           columns=["source_cell_label", "cell_group", "compound_id", "endpoint_value"])
    data = rows.groupby(["source_cell_label", "compound_id"]).endpoint_value.mean().unstack()
    data = data.sub(data.mean(axis=1), axis=0)
    tested = set(variants.Tumor_Sample_Barcode) & set(data.index)
    status = {}
    for gene in ("TP53", "CTNNB1"):
        select = variants.Hugo_Symbol.eq(gene) & variants.Variant_Classification.isin(CODING_CLASSES)
        status[gene] = set(variants.loc[select, "Tumor_Sample_Barcode"]) & tested
    reference = data.loc[sorted(status["CTNNB1"] - status["TP53"])].mean() - data.loc[
        sorted(status["TP53"] - status["CTNNB1"])].mean()
    reference = reference.rename("limore_measured_difference")
    selected = patient_tests.loc[patient_tests.contrast.eq("CTNNB1_only_vs_TP53_only")]
    joined = selected.join(reference, on="compound_id").dropna(subset=["effect", reference.name])
    metrics = []
    for (arm, prep), group in joined.groupby(["arm", "preprocessing"]):
        rho = spearmanr(group.effect, group[reference.name]).statistic
        metrics.append(dict(arm=arm, preprocessing=prep, paired_drugs=len(group),
            spearman=float(rho), sign_agreement=float(np.mean(np.sign(group.effect) == np.sign(group[reference.name])))))
    support = dict(covered_cells=len(tested), ctnnb1_only=len(status["CTNNB1"] - status["TP53"]),
                   tp53_only=len(status["TP53"] - status["CTNNB1"]))
    return joined, metrics, support


def release_rank_agreement(forecast):
    native = forecast.loc[forecast.preprocessing.str.startswith("native_")]
    candidate = native.loc[native.arm.eq("candidate")].pivot(index="participant", columns="compound_id", values="value")
    baseline = native.loc[native.arm.eq("baseline")].pivot(index="participant", columns="compound_id", values="value")
    joint = candidate.columns.intersection(baseline.columns)
    if not len(joint):
        return None
    agreeing = candidate[joint].rank(axis=1).corrwith(baseline[joint].rank(axis=1), axis=1).dropna()
    return dict(shared_compounds=len(joint), complete_patients=len(agreeing), median_spearman=float(agreeing.median()))


def write_report(output, freeze, summary, panels):
    domain = summary["domain"]
    name = freeze["contract"]["candidate"]["contract"]["model_version"]
    lines = [
        f"# {name.upper()} TCGA-LIHC translation audit", "",
        f"The frozen release was evaluated on {freeze['tumors']} adult primary tumors and "
        f"{freeze['compound_pool']} registered compounds. Gene mapping covered "
        f"{freeze['native_gene_overlap']}/{freeze['native_model_genes']} checkpoint genes.", "",
        "| Input | RNA OOD | Abstentions | Median nearest fit distance |",
        "|---|---:|---:|---:|",
    ]
    for mode, data in domain.items():
        lines.append(f"| {mode} | {data['ood']}/{data['total']} | {data['abstain']}/{data['total']} "
                     f"| {data['distance_quantiles'][1]:.3f} |")
    lines.extend(["", "| Fixed-label Hoshida subtype | Tumors |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in sorted(summary["subtypes"].items()))
    lines.extend(["", "| Adjusted copy-number contrast | N | Effect | 99% CI | FDR |",
                  "|---|---:|---:|---:|---:|"])
    primary = panels.loc[panels.arm.eq("candidate") & panels.preprocessing.str.startswith("native_")
                         & panels.adjustment.eq("purity_immune_stroma")]
    for row in primary.itertuples():
        lines.append(f"| {row.contrast} | {row.n} | {row.effect:+.4f} | [{row.low99:+.4f}, {row.high99:+.4f}] | {row.fdr:.3f} |")
    lines.extend(["", "See the accompanying metrics and figures for purity, comparator and "
                  "gene-program sensitivity. Reported scores represent the released model's screening endpoint. "
                  "TCGA-LIHC contributes tumor annotations and lacks matched response labels for this panel.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))


def score(args, freeze_sha):
    predicted = args.out
    frozen = predicted / "freeze.json"
    if digest(frozen) != freeze_sha:
        raise ValueError("Supplied forecast freeze hash differs")
    freeze = json.loads(frozen.read_text())
    checked_files(predicted, freeze["outputs"])
    candidate = freeze["contract"]["candidate"]
    checked_files(args.model, candidate["files"])
    a_path = args.annotations / "receipt.json"
    a_receipt = json.loads(a_path.read_text())
    checked_files(args.annotations, a_receipt["outputs"])
    if a_receipt["cohort_receipt_sha256"] != freeze["sources"]["cohort"]:
        raise ValueError("Annotation cohort differs from frozen expression")
    estimate_manifest = args.annotations / "estimate.sha256"
    checked_files(args.annotations, {name: expected for expected, name in
                                    (line.split() for line in estimate_manifest.read_text().splitlines())})
    annotation = pd.read_parquet(args.annotations / "annotations.parquet").join(estimate_annotations(args.annotations))
    forecast = pd.read_parquet(predicted / "predictions.parquet")
    if set(forecast.participant) != set(annotation.index):
        raise ValueError("Frozen prediction and annotation IDs differ")
    panels, drug_tests, strata, purity, targets = tests(
        forecast, annotation, pd.read_csv(predicted / "compounds.tsv", sep="\t", index_col="compound_id"))
    source, concordance, concordance_support = limore_driver_reference(args.limore, args.limore_maf, drug_tests)
    domain = pd.read_parquet(predicted / "domain.parquet")
    points = pd.read_parquet(predicted / "coordinates.parquet")
    output = args.score_out or (args.out / "score")
    output.mkdir(exist_ok=False)
    for name, values in (("panels", panels), ("drug_contrasts", drug_tests), ("subtype_strata", strata),
                         ("purity_confounding", purity), ("limore_driver_concordance", source)):
        values.to_csv(output / f"{name}.tsv", sep="\t", index=False, na_rep="NA")
    plots(output, points, domain, annotation, panels, drug_tests)

    mode_stats, correlations = {}, {}
    for mode, data in domain.groupby("preprocessing"):
        mode_stats[mode] = dict(total=len(data), ood=int(data.ood.sum()), abstain=int(data.abstain.sum()),
            insufficient_genes=int(data.insufficient_genes.sum()),
            distance_quantiles=data.nearest_fit_distance.quantile([.05, .5, .95]).tolist())
        correlations[mode] = {}
        merged = data.join(annotation[list(PURITY)])
        for marker in PURITY:
            pairs = merged[["nearest_fit_distance", marker]].dropna()
            correlations[mode][marker] = dict(n=len(pairs), rho=float(spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1]).statistic))
    purity_stats = purity.loc[purity.arm.eq("candidate")].groupby(["preprocessing", "marker"]).partial_rank_rho.agg(
        median=lambda v: v.abs().median(), maximum=lambda v: v.abs().max()).reset_index()
    counts = annotation.hoshida_subtype.value_counts().to_dict()
    clinical = annotation[["OS", "OS.time", "PFI", "PFI.time", "ajcc_pathologic_tumor_stage"]].notna().sum().to_dict()
    summary = dict(domain=mode_stats, distance_marker_spearman=correlations,
        model_prediction_marker_partial_abs_rho=purity_stats.to_dict("records"), subtypes=counts,
        clinical_available=clinical, coding_mutations={c: dict(n=int(annotation[c].notna().sum()),
                    positives=int(annotation[c].sum())) for c in annotation if c.endswith("_coding_mutation")},
        copy_number={c: dict(n=int(annotation[c].notna().sum()), gains=int(annotation[c].eq(2).sum()),
                    losses=int(annotation[c].eq(-2).sum())) for c in annotation if c.endswith("_gistic")},
        prognosis_or_response_labels_used=False,
        limore_driver_reference=concordance, limore_driver_support=concordance_support,
        baseline_native_rank_agreement=release_rank_agreement(forecast),
        target_panel_compounds={k: list(v) for k, v in targets.items()},
        fdr_discoveries=drug_tests.assign(significant=drug_tests.fdr.lt(.05)).groupby(
            ["arm", "preprocessing", "contrast"]).significant.sum().reset_index().to_dict("records"))
    write_receipt(output / "metrics.json", summary)
    write_report(output, freeze, summary, panels)
    write_receipt(output / "receipt.json", dict(freeze_sha256=freeze_sha, annotation_receipt_sha256=digest(a_path),
        source_limore_maf_sha256=digest(args.limore_maf), score_code_sha256=digest(Path(__file__)),
        estimate_manifest_sha256=digest(estimate_manifest), outputs={p.name: digest(p) for p in sorted(output.iterdir())}))
