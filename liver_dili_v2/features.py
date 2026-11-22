"""Structure-only DILI representations and verified frozen embeddings."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from liver_dili.chemistry import features, identity
from liver_dili.splits import digest

ALERTS = {
    "michael_acceptor": "[C,c]=[C,c][C,S](=O)[#6,#7,#8]",
    "quinone": "O=C1C=CC(=O)C=C1",
    "epoxide": "[O;r3]1[C;r3][C;r3]1",
    "aromatic_amine": "[NX3;H1,H2][c]",
    "nitro": "[$([N+](=O)[O-]),$([N](=O)=O)]",
    "furan": "o1cccc1",
    "thiophene": "s1cccc1",
    "hydrazine": "[N;X3][N;X3]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "alkyl_halide": "[C;X4][Cl,Br,I]",
    "metal": "[Li,Na,K,Mg,Ca,Al,Fe,Co,Ni,Cu,Zn,As,Se,Ag,Cd,Pt,Hg,Pb,Bi]",
}
PATTERNS = [Chem.MolFromSmarts(s) for s in ALERTS.values()]
MORGAN_COUNT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
FM_NAMES = ("chemberta2", "molformer", "unimol", "clamp")


@lru_cache(maxsize=1)
def catalogs():
    config = FilterCatalog.FilterCatalogParams
    result = []
    for name in ("PAINS", "BRENK"):
        options = config()
        options.AddCatalog(getattr(config.FilterCatalogs, name))
        result.append(FilterCatalog.FilterCatalog(options))
    return result


@lru_cache(maxsize=24000)
def structure(smiles):
    canonical, *_ = identity(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    _, original = features(canonical)
    mw, logp, hbd, hba = (fn(molecule) for fn in (
        Descriptors.MolWt, Crippen.MolLogP, Lipinski.NumHDonors, Lipinski.NumHAcceptors,
    ))
    extended = [
        rdMolDescriptors.CalcFractionCSP3(molecule), rdMolDescriptors.CalcNumHeteroatoms(molecule),
        rdMolDescriptors.CalcNumAmideBonds(molecule), rdMolDescriptors.CalcLabuteASA(molecule),
        Chem.GetFormalCharge(molecule), mw > 500, logp > 5, hbd > 5, hba > 10,
        max(0., logp - 3.), logp - mw / 100., molecule.GetNumHeavyAtoms(),
        *(len(c.GetMatches(molecule)) for c in catalogs()),
        *(len(molecule.GetSubstructMatches(p)) for p in PATTERNS),
    ]
    counts = MORGAN_COUNT.GetCountFingerprintAsNumPy(molecule)
    return np.r_[original, extended].astype(np.float32), counts.astype(np.float32)


def verified_bank(path: Path):
    meta = json.loads((path / "receipt.json").read_text())
    archive = path / "vectors.npz"
    output = meta["output"]
    if archive.stat().st_size != output["bytes"] or digest(archive) != output["sha256"]:
        raise ValueError(f"Embedding SHA-256 or file extent mismatch: {archive}")
    with np.load(archive, allow_pickle=False) as data:
        keys = data["id"].astype(str)
        vectors = data["vectors"]
    if not pd.Index(keys).is_unique or len(keys) != len(vectors) or not np.isfinite(vectors).all():
        raise ValueError(f"Embedding keys or values invalid: {archive}")
    return pd.DataFrame(vectors, index=keys), output["sha256"]


class Features:
    def __init__(self, fm_root, chemberta):
        self.banks = {}
        self.hashes = {}
        for name in FM_NAMES:
            self.banks[name], self.hashes[name] = verified_bank(fm_root / name)
        self.chemberta = pd.read_parquet(chemberta).drop(columns="chemberta_truncated")
        self.hashes["chemberta_fda"] = digest(chemberta)

    def arrays(self, rows):
        base, counts = map(np.stack, zip(*(structure(s) for s in rows.smiles)))
        blocks = {"base": base, "counts": counts}
        for name, bank in self.banks.items():
            fm = bank.reindex(rows.inchikey).to_numpy(dtype=np.float32)
            if name == "chemberta2":
                original = self.chemberta.reindex(rows.inchikey).to_numpy(dtype=np.float32)
                if original.shape == fm.shape:
                    missing = ~np.isfinite(fm).all(axis=1)
                    fm[missing] = original[missing]
            blocks[name] = fm
        blocks["v1"] = self.chemberta.reindex(rows.inchikey).to_numpy(dtype=np.float32)
        return blocks


def design(blocks, representation):
    base = blocks["base"]
    if representation == "structure":
        return base
    if representation == "v1":
        return np.c_[base[:, :2060], blocks["v1"]]
    names = FM_NAMES if representation == "all_fm" else (representation,)
    extra = [np.c_[blocks[name], np.isfinite(blocks[name]).all(axis=1)] for name in names]
    return np.concatenate([base, *extra], axis=1)


def subset(blocks, indices):
    return {name: block[indices] for name, block in blocks.items()}


def tanimoto(first, second, *, mode):
    if mode == "minmax":
        from scipy.spatial.distance import cdist

        distance = cdist(first, second, metric="cityblock")
        mass = first.sum(axis=1)[:, None] + second.sum(axis=1)
        return np.divide(mass-distance, mass+distance, out=np.zeros_like(distance), where=mass+distance > 0)
    if mode == "bits":
        first, second = first[:, :2048], second[:, :2048]
    dot = first @ second.T
    union = np.square(first).sum(axis=1)[:, None] + np.square(second).sum(axis=1) - dot
    return np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
