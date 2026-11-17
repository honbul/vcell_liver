"""Render the markdown tables used in RESULTS.md from the committed result tables (standard library only)."""
import csv, sys

MAN = "manifests/episodic_residual_r2/"
NAME = [("no_change", "무변화"), ("source_same_drug_mean", "f0 (source 같은 약물 평균)"), ("mean_offset", "f0 + mean-offset (판정 기준선)"),
        ("ecfp_ridge", "ECFP 잔차 Ridge"), ("ecfp_krr", "ECFP Tanimoto KRR"), ("allfeat_krr", "동일 입력 비학습 KRR"),
        ("shrunk_same_drug_mean", "축소 f0 (추가 기준선)"), ("shrunk_mean_offset", "축소 f0 + 축소 offset (추가 기준선)"),
        ("generic_cnp", "일반 CNP"), ("residual_adapter", "**제안법 (동결)**"), ("residual_adapter__zero_residual", "제안법, 잔차 0"),
        ("residual_adapter__shuffle_residual", "제안법, 잔차 행 교환"), ("adapter_train_shuffle", "교환 학습 대조"),
        ("adapter_no_ctrl", "대조군 feature 제외"), ("residual_adapter_initial", "초기 통합안 (수정 전)"), ("adapter_joint", "비분리형 (R2-3)"),
        ("transigen_as_is", "TranSiGen 그대로"), ("transigen_offset", "TranSiGen + offset"), ("transigen_full_ft", "TranSiGen full FT")]
TARGETS = ["HUH7", "JHH5", "JHH7", "EQUAL_MEAN_3"]


def read(f):
    return list(csv.DictReader(open(MAN + f), delimiter="\t"))


def results(qset="test", metric="delta_rmse", ci=True, digits=4):
    rows = read("hcc_results.tsv")
    idx = {(r["target"], r["K"], r["method"]): r for r in rows if r["query_set"] == qset}
    for K in ("8", "16", "32"):
        print(f"\n**K = {K}** ({qset}, {metric})\n")
        print("| 방법 | HUH7 | JHH5 | JHH7 | 세 계열 동등 평균 |\n|---|---:|---:|---:|---:|")
        for m, lab in NAME:
            cells = []
            for t in TARGETS:
                r = idx.get((t, K, m))
                if r is None or r[metric] == "":
                    cells.append("—")
                elif ci:
                    cells.append(f"{float(r[metric]):.{digits}f} [{float(r[metric + '_lo']):.{digits}f}, {float(r[metric + '_hi']):.{digits}f}]")
                else:
                    cells.append(f"{float(r[metric]):.{digits}f}")
            print(f"| {lab} | " + " | ".join(cells) + " |")


def contrasts(ref="mean_offset", qset="test", metric="delta_rmse"):
    rows = read("hcc_contrasts.tsv")
    idx = {(r["target"], r["K"], r["method"]): r for r in rows if r["query_set"] == qset and r["reference"] == ref and r["metric"] == metric}
    for K in ("8", "16", "32"):
        print(f"\n**K = {K}** ({qset}, {metric}: 방법 − {ref})\n")
        print("| 방법 | HUH7 | JHH5 | JHH7 | 세 계열 동등 평균 |\n|---|---:|---:|---:|---:|")
        for m, lab in NAME:
            if m == ref:
                continue
            cells = []
            for t in TARGETS:
                r = idx.get((t, K, m))
                cells.append("—" if (r is None or r['diff'] == '') else f"{float(r['diff']):+.4f} [{float(r['lo']):+.4f}, {float(r['hi']):+.4f}]")
            print(f"| {lab} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "results":
        results(*(sys.argv[2:4] or ["test", "delta_rmse"]), ci=(len(sys.argv) < 5))
    else:
        contrasts(*sys.argv[2:5])
