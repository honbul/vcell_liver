"""Independently check saved paper predictions without another training run."""
import argparse
from pathlib import Path
import time

import anndata as ad
import numpy as np
from scipy.stats import pearsonr

from artifact_io import digest, write
from paper_comparison import ROOT, load_json, tsv

PUBLIC = ROOT/'assets/public'
TRACKS = ('R1','R2')


def groups(bundle, split, genes, adata):
    if bundle['gene_names'].tolist() != genes:
        raise ValueError('Prediction output genes are not in the author order')
    names = bundle['pert_cat'].astype(str)
    ref = adata.obs['condition'].astype(str).to_numpy()
    unique = sorted(set(names))
    if unique != sorted(split['conditions']['test']):
        raise ValueError('Saved predictions do not cover the held-out conditions')
    pred, truth = bundle['pred'], bundle['truth']
    if pred.shape != truth.shape or pred.shape != (len(names),len(genes)):
        raise ValueError('Unexpected saved expression shape')
    for c in unique:
        rows = np.flatnonzero(names==c)
        expected = np.flatnonzero(ref==c)
        if len(rows) != len(expected):
            raise ValueError(f'Unexpected measured cell count for {c}')
        y = truth[rows].mean(0)
        obs = np.asarray(adata.X[expected].mean(0)).ravel()
        np.testing.assert_allclose(y,obs,atol=1e-4,rtol=2e-5)
        yield c,pred[rows].mean(0),y


def summarize_gears(run, adata, ctrl, gene_id):
    split = load_json(run/'split.json')
    output = load_json(run/'result.json')
    genes = adata.var.gene_name.astype(str).tolist()
    cond_key = dict(zip(adata.obs.condition, adata.obs.condition_name))
    by_condition = {r['condition']:r for r in tsv(run/'fig2f_conditions.tsv')}
    max_abs = 0.0
    with np.load(run/'prediction.npz', allow_pickle=False) as bundle:
        for name,p,y in groups(bundle,split,genes,adata):
            idx = [gene_id[g] for g in adata.uns['top_non_dropout_de_20'][cond_key[name]]]
            value = np.mean(((p-ctrl)[idx]-(y-ctrl)[idx])**2)
            ref = float(by_condition[name]['gears'])
            np.testing.assert_allclose(value,ref,atol=1e-6,rtol=2e-5)
            max_abs = max(max_abs,abs(value-ref))
    return dict(seed=output['split_seed'],verified_conditions=len(by_condition),
                max_mse_difference=float(max_abs),result_sha256=digest(run/'result.json'))


def summarize_scgpt(run, adata, ctrl, gene_id):
    split = load_json(run/'split.json')
    output = load_json(run/'result.json')
    genes = adata.var.gene_name.astype(str).tolist()
    key = next(iter(adata.uns['rank_genes_groups_cov_all'])).split('_')
    obs = {}
    variant = []
    with np.load(run/'prediction.npz',allow_pickle=False) as bundle:
        for name,p,y in groups(bundle,split,genes,adata):
            group = '_'.join([key[0],name,key[2]])
            idx = [gene_id[g] for g in adata.uns['rank_genes_groups_cov_all'][group][:20]]
            alt = [gene_id[g] for g in adata.uns['top_non_dropout_de_20'][group][:20]]
            py, ty = p-ctrl,y-ctrl
            values = {}
            if np.any(y != 0):
                values['pearson'] = float(pearsonr(p,y)[0])
                values['pearson_delta'] = float(pearsonr(py,ty)[0])
            if np.any(y[idx] != 0):
                values['pearson_de'] = float(pearsonr(p[idx],y[idx])[0])
                values['pearson_de_delta'] = float(pearsonr(py[idx],ty[idx])[0])
            if len(alt)>1 and np.any(y[alt] != 0):
                variant.append(pearsonr(py[alt],ty[alt])[0])
            obs[name] = values
    recomputed = {metric:float(np.mean([r[metric] for r in obs.values() if metric in r]))
                  for metric in ('pearson','pearson_delta','pearson_de','pearson_de_delta')}
    for k,v in recomputed.items():
        np.testing.assert_allclose(v,output['metrics'][k],atol=1e-7,rtol=2e-6)
    return dict(verified_conditions=len(obs),author_evaluator=recomputed,
                exploratory_top20_non_dropout_pearson_de_delta=float(np.mean(variant)),
                alternative_n=len(variant),result_sha256=digest(run/'result.json'))


def diagnose(runs, paper_status, r1_stem, r2_id, output):
    started = time.monotonic()
    status = load_json(paper_status)
    if any(status[k]['status']!='modified_reproduction_executed' for k in TRACKS):
        raise ValueError('Independent diagnostics require both paper tracks to have finished')
    path = PUBLIC/'norman/perturb_processed.h5ad'
    registry = {r['file']:r['sha256'] for r in load_json(ROOT/'manifests/public_receipt.json')}
    if digest(path) != registry['norman/perturb_processed.h5ad']:
        raise ValueError('Official Norman matrix hash mismatch')
    adata = ad.read_h5ad(path)
    ctrl = np.asarray(adata.X[np.asarray(adata.obs.condition=='ctrl')].mean(0)).ravel()
    gene_id = {g:i for i,g in enumerate(adata.var.index)}
    gears = [summarize_gears(runs/'R1'/f'{r1_stem}-s{s}',adata,ctrl,gene_id) for s in range(1,6)]
    scgpt = summarize_scgpt(runs/'R2'/r2_id,adata,ctrl,gene_id)
    if {d['result_sha256'] for d in gears} != {r['result_sha256'] for r in status['R1']['evidence']}:
        raise ValueError('GEARS diagnostics used different completed runs')
    if scgpt['result_sha256'] != status['R2']['evidence']['result_sha256']:
        raise ValueError('scGPT diagnostics used a different completed run')
    write(output,dict(paper_status_sha256=digest(paper_status),
                      scope='Independent evaluator check using complete test exports; no new training',
                      R1=gears,R2=scgpt,seconds=time.monotonic()-started,
                      limitations='Alternative DE score is diagnostic only; it cannot identify the publisher calculation.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'assets/runs')
    p.add_argument('--paper-status',type=Path,required=True)
    p.add_argument('--r1-stem',default='R1-paper-public-v1')
    p.add_argument('--r2-id',default='R2-manuscript-public-v1')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    diagnose(a.runs,a.paper_status,a.r1_stem,a.r2_id,a.output)


if __name__=='__main__':
    main()
