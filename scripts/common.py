"""Shared paths, provenance, resource measurement and donor-held-out evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATHS = yaml.safe_load((ROOT / 'configs/paths.yaml').read_text())
DATA = Path(PATHS['data_root'])
RAW = Path(PATHS['raw_root'])


def config():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    a = p.parse_args()
    return yaml.safe_load(Path(a.config).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


class Resources:
    def __init__(self):
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.peak_gpu = 0
        self.gpu_sampled = False
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        while not self.stop.is_set():
            try:
                s = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
                for row in s.splitlines():
                    pid, mem = row.split(',')
                    if int(pid.strip()) == os.getpid():
                        self.peak_gpu = max(self.peak_gpu, float(mem.strip()) / 1024)
                        self.gpu_sampled = True
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.stop.wait(0.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=6)
        r = dict(wall_seconds=time.monotonic()-self.started,
                 peak_cpu_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 peak_process_gpu_gib=self.peak_gpu if self.gpu_sampled else None)
        try:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                r.update(gpu_name=torch.cuda.get_device_name(),
                         peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                         peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        except ImportError:
            pass
        return r


def provenance(c, adata, model, resource_monitor, **extra):
    subset = c.get('subset', 'pilot')
    return dict(run_id=c['run_id'], status='executed', task='annotation',
                dataset_accession='GSE149614', model_name=model, seed=42,
                n_cells=adata.n_obs, data_hash=sha(DATA/'subsets'/f'{subset}.h5ad'),
                split_manifest_hash=sha(ROOT/'manifests/splits'/f'{subset}.tsv'),
                pretraining_overlap='unknown', **(resource_monitor if isinstance(resource_monitor,dict) else resource_monitor.finish()), **extra)


def evaluate(features, adata, run_id, outdir=None):
    """Fit scaler/head on train; select C/weight on validation; evaluate test once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.preprocessing import StandardScaler
    import joblib
    out = Path(outdir or ROOT/'results'/run_id)
    out.mkdir(parents=True, exist_ok=True)
    obs = adata.obs
    y = obs.cell_type_coarse.astype(str).to_numpy()
    tr, va, te = [obs.split.astype(str).to_numpy() == s for s in ('train', 'val', 'test')]
    labels = sorted(set(y[tr]))
    assert all(set(obs.loc[obs.split == a, 'patient_id']).isdisjoint(set(obs.loc[obs.split == b, 'patient_id'])) for a,b in [('train','val'),('train','test'),('val','test')])
    assert np.isfinite(features).all() and len(features) == len(obs)
    sc = StandardScaler().fit(features[tr])
    x = sc.transform(features)
    candidates = []
    best = None
    for C in (0.01, 0.1, 1.0):
        for w in (None, 'balanced'):
            clf = LogisticRegression(C=C, class_weight=w, max_iter=2000, random_state=42)
            clf.fit(x[tr], y[tr])
            score = f1_score(y[va], clf.predict(x[va]), labels=labels, average='macro', zero_division=0)
            candidates.append(dict(C=C, class_weight=w, validation_macro_f1=score))
            if best is None or score > best[0]:
                best = (score, clf, C, w)
    _, clf, C, w = best
    pred = clf.predict(x)
    known = te & np.isin(y, labels)
    metrics = dict(macro_f1=f1_score(y[known], pred[known], labels=labels, average='macro', zero_division=0),
                   balanced_accuracy=balanced_accuracy_score(y[known], pred[known]),
                   n_test_unknown=int((te & ~np.isin(y,labels)).sum()), n_evaluated=int(known.sum()),
                   selected_C=C, selected_class_weight=w,
                   split_counts=obs.split.value_counts().to_dict(),
                   label_classes=labels, validation_candidates=candidates)
    write_json(out/'metrics.json', metrics)
    pd.DataFrame(classification_report(y[known], pred[known], labels=labels, output_dict=True, zero_division=0)).T.to_csv(out/'per_class.tsv', sep='\t')
    pd.DataFrame(confusion_matrix(y[known], pred[known], labels=labels), index=labels, columns=labels).to_csv(out/'confusion_matrix.tsv',sep='\t')
    pats=[]
    for patient in sorted(obs.loc[te, 'patient_id'].unique()):
        ix = known & (obs.patient_id.to_numpy()==patient)
        pats.append(dict(patient_id=patient,n_cells=int(ix.sum()),macro_f1=f1_score(y[ix],pred[ix],labels=labels,average='macro',zero_division=0),balanced_accuracy=balanced_accuracy_score(y[ix],pred[ix])))
    pd.DataFrame(pats).to_csv(out/'by_patient.tsv',sep='\t',index=False)
    rawout = DATA/'outputs'/run_id
    rawout.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(cell_id=obs.index,patient_id=obs.patient_id,split=obs.split,truth=y,prediction=pred,confidence=clf.predict_proba(x).max(1))).to_csv(rawout/'predictions.tsv',sep='\t',index=False)
    joblib.dump(dict(scaler=sc,classifier=clf),rawout/'head.joblib')
    # A shuffled-label head is a leakage sanity check, never a model selector.
    rng=np.random.default_rng(42)
    shuffled=LogisticRegression(C=C,class_weight=w,max_iter=2000,random_state=42).fit(x[tr],rng.permutation(y[tr]))
    metrics['shuffled_train_labels_test_macro_f1']=f1_score(y[known],shuffled.predict(x[known]),labels=labels,average='macro',zero_division=0)
    write_json(out/'metrics.json',metrics)
    print(json.dumps(dict(run_id=run_id, **{k:v for k,v in metrics.items() if k!='validation_candidates'})),flush=True)
    return metrics


def save_features(adata, values, run_id):
    d=DATA/'features';d.mkdir(exist_ok=True)
    np.savez_compressed(d/f'{run_id}.npz',X=values,cell_ids=adata.obs_names.to_numpy(dtype=str))


def load_features(adata, run_id):
    x=np.load(DATA/'features'/f'{run_id}.npz')
    assert np.array_equal(x['cell_ids'],adata.obs_names.to_numpy(dtype=str))
    return x['X']
