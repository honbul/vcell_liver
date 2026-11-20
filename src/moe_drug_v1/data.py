"""Load the frozen condition table and expose the split tensors. No split or panel decision lives here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SNU423 = "CVCL_0366"


@dataclass
class Conditions:
    """Valid (line, drug) pairs of one split, flattened, with the loss weights the plan fixes."""
    line: torch.Tensor      # [N] long
    drug: torch.Tensor      # [N] long
    y: torch.Tensor         # [N, G] float32
    weight: torch.Tensor    # [N] uniform over lines, then uniform over that line's drugs; sums to 1

    def __len__(self) -> int:
        return self.line.numel()


class Dataset:
    def __init__(self, data_dir: str | Path, manifest_dir: str | Path, device: str = "cpu",
                 apply_coverage_filter: bool = True):
        D, M = Path(data_dir), Path(manifest_dir)
        Y = np.load(D / "Y.npy")                 # [L, Dg, G]  NaN where invalid
        valid = np.load(D / "valid.npy")
        basal = np.load(D / "basal.npy")
        fp = np.load(D / "fp.npy").astype(np.float32)
        plate = np.load(D / "plate.npy")
        splits = pd.read_csv(M / "splits.tsv", sep="\t")
        cov = pd.read_csv(M / "coverage.tsv", sep="\t")
        lines = pd.read_csv(M / "lines.tsv", sep="\t")

        keep = cov.keep.to_numpy() if apply_coverage_filter else np.ones(len(cov), bool)
        self.line_ids = cov.cell_line_id.to_numpy()[keep]
        self.line_names = cov.cell_name.to_numpy()[keep]
        self.kept_line_index = np.where(keep)[0]
        Y, valid, basal, plate = Y[keep], valid[keep], basal[keep], plate[keep]

        self.Y_np, self.valid_np, self.plate_np = Y, valid, plate
        self.split_of = splits.split.to_numpy()
        self.groups = splits.group_smiles.to_numpy()
        self.n_lines, self.n_drugs, self.n_genes = Y.shape
        self.n_bits = fp.shape[1]
        self.device = device
        self.basal = torch.as_tensor(basal, device=device)
        self.fp = torch.as_tensor(fp, device=device)
        self.snu423 = int(np.where(self.line_ids == SNU423)[0][0]) if (self.line_ids == SNU423).any() else None
        self.dropped_lines = cov.loc[~keep, ["cell_line_id", "cell_name", "train", "val", "test"]]

    def conditions(self, split: str) -> Conditions:
        cols = np.where(self.split_of == split)[0]
        m = self.valid_np[:, cols]
        li, dj = np.where(m)
        drug = cols[dj]
        y = self.Y_np[li, drug]
        per_line = m.sum(1).astype(np.float64)
        n_lines_present = int((per_line > 0).sum())
        wt = 1.0 / (n_lines_present * per_line[li])
        return Conditions(
            line=torch.as_tensor(li, dtype=torch.long, device=self.device),
            drug=torch.as_tensor(drug, dtype=torch.long, device=self.device),
            y=torch.as_tensor(y, dtype=torch.float32, device=self.device),
            weight=torch.as_tensor(wt, dtype=torch.float32, device=self.device))

    def inputs(self, c: Conditions):
        return self.basal[c.line], self.fp[c.drug]

    def drug_index(self, split: str) -> np.ndarray:
        return np.where(self.split_of == split)[0]
