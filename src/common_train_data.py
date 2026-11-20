"""Only the training cells, shared controls, and validation condition inputs."""
import json
import numpy as np
from scipy import sparse

from artifact_io import digest
from common_data import PROTOCOL, fixed_control


class TrainingData:
    def __init__(self, path):
        self.path = path
        self.receipt = json.loads((path/'receipt.json').read_text())
        if self.receipt['contract_sha256'] != digest(PROTOCOL):
            raise ValueError('Dataset and protocol do not match')
        inputs = ('genes.json','train_conditions.json','val_conditions.json','controls.json',
                  'controls_train.npz','treated_train.npz','train_index.npz','eval_controls.npy',
                  'outcomes_val.npz')
        for file in inputs:
            if digest(path/file) != self.receipt['files'][file]:
                raise ValueError(f'Common dataset file mismatch: {file}')
        self.genes = self.json('genes')['output']
        self.gene_index = {g: i for i, g in enumerate(self.genes)}
        self.train = self.json('train_conditions')
        self.val = self.json('val_conditions')
        assert not set(self.train) & set(self.val)
        self.control = sparse.load_npz(path/'controls_train.npz')
        self.treated = sparse.load_npz(path/'treated_train.npz')
        self.eval_control = np.load(path/'eval_controls.npy', allow_pickle=False)
        self.pool_ids = self.json('controls')['train_pool_ids']
        with np.load(path/'train_index.npz', allow_pickle=False) as ix:
            self.train_names, self.train_ids = ix['condition'], ix['cell_ids']
            self.paired = ix['fixed_control_index']
        self.train_rows = {c: np.flatnonzero(self.train_names == c) for c in self.train}
        self.train_targets = {c: self.indices(item) for c,item in self.train.items()}

    def json(self, name):
        return json.loads((self.path/f'{name}.json').read_text())

    def indices(self, item):
        return [self.gene_index[g] for g in item['targets']]

    def schedule(self, mode, seed, epoch):
        n = len(self.train_ids)
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2515]))
        pairs = self.paired
        if mode == 'D1':
            pairs = np.array([fixed_control(self.pool_ids, cell, self.receipt['split_seed'], epoch)
                              for cell in self.train_ids])
        if mode != 'D2':
            return rng.permutation(n), pairs
        conds = sorted(self.train_rows)
        rng.shuffle(conds)
        q, rem = divmod(n, len(conds))
        parts = []
        for i, cond in enumerate(conds):
            count = q + (i < rem)
            rows = self.train_rows[cond]
            draws = [rng.permutation(rows) for _ in range((count+len(rows)-1)//len(rows))]
            parts.append(np.concatenate(draws)[:count])
        rows = np.concatenate(parts)
        rng.shuffle(rows)
        assert len(rows) == n
        return rows, pairs

    def batch(self, rows, pairs, panel=None):
        selected = self.treated[rows]
        control = self.control[pairs[rows]]
        targets = [self.train_targets[c] for c in self.train_names[rows]]
        if panel is not None:
            selected, control = selected[:,panel], control[:,panel]
        return control.toarray(), selected.toarray(), targets
