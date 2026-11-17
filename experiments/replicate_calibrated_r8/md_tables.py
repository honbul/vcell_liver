"""Print the Markdown tables used in RESULTS.md straight from the committed result tables (standard library only).

usage: python3 experiments/replicate_calibrated_r8/md_tables.py [source|hcc|hcc_mock] ...
"""
import csv, sys, os

MAN = "manifests/replicate_calibrated_r8"
ORDER = ["no_change", "same_drug_mean", "shrunk_same_drug_mean", "mean_offset", "C0_iid", "C1_iid", "B00", "B01", "B10", "B11", "B10v2", "B11v2", "G_concat", "G_dummy",
         "C0_cov", "C1_cov", "C0_cov2", "C1_cov2", "cnp", "transigen_as_is", "transigen_offset", "transigen_full_ft",
         "B00__shuffle_residual", "B01__shuffle_residual", "B10__shuffle_residual", "B11__shuffle_residual"]
MAIN_CONTRASTS = ["C1_iid-C0_iid", "B00-C0_iid", "B00-C1_iid", "B01-B00", "B10-B00", "B11-B00", "interaction(B11-B10-B01+B00)", "B10v2-B00", "B11v2-B00",
                  "G_concat-B00", "G_dummy-B00", "B00__shuffle_residual-B00", "B00-no_change", "B00-same_drug_mean", "B00-shrunk_same_drug_mean", "B00-mean_offset", "B00-cnp",
                  "B00-transigen_full_ft", "C0_iid-no_change", "C0_iid-same_drug_mean", "C0_iid-shrunk_same_drug_mean", "C0_iid-mean_offset", "cnp-C0_iid", "transigen_full_ft-C0_iid",
                  "C1_iid-transigen_full_ft", "C0_cov-C0_iid", "C0_cov2-C0_iid"]


def read(name):
    with open(os.path.join(MAN, name)) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f4(x, sign=False):
    if x in ("", None):
        return "—"
    return f"{float(x):+.4f}" if sign else f"{float(x):.4f}"


def source(tag="source_eval"):
    s = read(f"{tag}_summary.tsv")
    by = {(r["method"], r["K"]): r for r in s}
    print(f"### {tag}: family-balanced query delta RMSE\n\n| method | K=8 | K=16 | K=32 | K=16 centred RMSE | K=16 centred Pearson | K=16 shared plate | K=16 disjoint plate |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in ORDER:
        if (m, "16") not in by:
            continue
        r = by[(m, "16")]
        print(f"| {m} | {f4(by[(m, '8')]['delta_rmse'])} | {f4(r['delta_rmse'])} | {f4(by[(m, '32')]['delta_rmse'])} | {f4(r['centred_rmse'])} | {f4(r['centred_pearson'])} | "
              f"{f4(r['delta_rmse_shared_plate'])} | {f4(r['delta_rmse_disjoint_plate'])} |")
    c = read(f"{tag}_contrasts.tsv")
    cb = {(r["contrast"], r["K"], r["metric"]): r for r in c}
    print(f"\n### {tag}: paired contrasts (family bootstrap 95%)\n\n| contrast | K=8 | K=16 | K=16 interval | lower in | K=32 | K=16 centred RMSE |\n|---|---:|---:|---|---:|---:|---:|")
    for n in MAIN_CONTRASTS:
        if (n, "16", "delta_rmse") not in cb:
            continue
        r = cb[(n, "16", "delta_rmse")]
        g = lambda K, v="delta_rmse": cb.get((n, K, v), {}).get("diff", "")
        cr = cb.get((n, "16", "centred_rmse"))
        cr = f"{f4(cr['diff'], True)} [{f4(cr['lo'], True)}, {f4(cr['hi'], True)}]" if cr else "—"
        print(f"| {n} | {f4(g('8'), True)} | {f4(r['diff'], True)} | [{f4(r['lo'], True)}, {f4(r['hi'], True)}] | {r['n_families_negative']}/{r['n_families']} | {f4(g('32'), True)} | {cr} |")


def hcc(tag="hcc", qset="test"):
    s = [r for r in read(f"{tag}_results.tsv") if r["query_set"] == qset]
    targets = sorted({r["target"] for r in s}, key=lambda t: (t == "EQUAL_MEAN_3", t))
    by = {(r["target"], r["method"], r["K"]): r for r in s}
    for K in ("16", "8", "32"):
        print(f"\n### {tag} ({qset}): query delta RMSE, K={K}\n\n| method | " + " | ".join(targets) + " |\n|---|" + "---:|" * len(targets))
        for m in ORDER:
            if not any((t, m, K) in by for t in targets):
                continue
            cells = []
            for t in targets:
                r = by.get((t, m, K))
                cells.append("—" if not r else f"{f4(r['delta_rmse'])} [{f4(r['delta_rmse_lo'])}, {f4(r['delta_rmse_hi'])}]" if K == "16" else f4(r["delta_rmse"]))
            print(f"| {m} | " + " | ".join(cells) + " |")
    print(f"\n### {tag} ({qset}): centred metrics, K=16\n\n| method | " + " | ".join(f"{t} cRMSE | {t} cPearson" for t in targets) + " |\n|---|" + "---:|---:|" * len(targets))
    for m in ORDER:
        if not any((t, m, "16") in by for t in targets):
            continue
        print(f"| {m} | " + " | ".join(f"{f4(by[(t, m, '16')]['centred_rmse'])} | {f4(by[(t, m, '16')]['centred_pearson'])}" if (t, m, "16") in by else "— | —" for t in targets) + " |")
    c = [r for r in read(f"{tag}_contrasts.tsv") if r["query_set"] == qset]
    cb = {(r["target"], r["contrast"], r["K"], r["metric"]): r for r in c}
    for K, v in (("16", "delta_rmse"), ("16", "centred_rmse"), ("8", "delta_rmse"), ("32", "delta_rmse")):
        print(f"\n### {tag} ({qset}): paired contrasts, K={K}, {v} (drug-group bootstrap 95%)\n\n| contrast | " + " | ".join(targets) + " |\n|---|" + "---|" * len(targets))
        for n in MAIN_CONTRASTS:
            if not any((t, n, K, v) in cb for t in targets):
                continue
            cells = []
            for t in targets:
                r = cb.get((t, n, K, v))
                cells.append("—" if not r else f"{f4(r['diff'], True)} [{f4(r['lo'], True)}, {f4(r['hi'], True)}]")
            print(f"| {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["source"]:
        if a.startswith("hcc"):
            hcc(a)
        else:
            source(a if a != "source" else "source_eval")
