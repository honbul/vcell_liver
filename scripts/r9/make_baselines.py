"""BASELINES.tsv: same-information-contract baselines (+ external full pipelines in a separate block) with cost."""
import pandas as pd

from r9common import *
from r9arms import DESCR

t = pd.read_csv(REPORT / 'a2_model_table.tsv', sep='\t')
DESCR = dict(DESCR, scpoli='scPoli (scArches 0.6.1) full pipeline, raw counts of the fold HVGs, query surgery on test donor RNA',
             protocloud='ProtoCloud (commit 90c2678) full pipeline, raw counts of the fold HVGs, official defaults')
t['block'] = t.model.map(lambda m: 'external_full_pipeline' if m in ('scpoli', 'protocloud') else 'same_PCA50_representation')
t['information'] = t.model.map(lambda m: 'TRANSDUCTIVE (unsupervised adaptation on test-donor RNA)' if m == 'scpoli'
                               else 'inductive; source cells+labels+donor IDs; test cell RNA only')
t['description'] = t.model.map(DESCR)
t['n_prototypes'] = t.model.map(lambda m: '6/class (official default)' if m == 'protocloud' else
                                ('C' if m.startswith('P1') else ('3C' if m in ('A00', 'A10', 'A01', 'A11', 'TL_donor', 'RA_donor', 'GMM_K') else 'n/a')))
cols = ['block', 'condition', 'model', 'description', 'information', 'n_prototypes', 'n_eval_donors', 'primary_auprc', 'ci_lo', 'ci_hi',
        'macro_f1', 'brier', 'nll'] + [c for c in t.columns if c.endswith('_auprc') and c != 'primary_auprc'] + \
       [c for c in ('target_recall', 'target_false_pos') if c in t] + ['median_cpu_seconds_per_fit']
t = t[cols].sort_values(['block', 'condition', 'model'])
write_tsv(t, REPORT / 'BASELINES.tsv')
print(t[['block', 'condition', 'model', 'n_eval_donors', 'primary_auprc', 'ci_lo', 'ci_hi', 'macro_f1']].to_string())
