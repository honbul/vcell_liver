"""Build identical CLAMP molecular inputs for registered and unregistered structures."""
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

from vc_fm_bench.artifacts import digest
from vc_fm_bench.clamp import embeddings


CHECKPOINT = {
    "checkpoint.pt": "fc41395199dde8b7afce18a2214537c66b46602b636f27eefb22463a1e78eb2a",
    "hp.json": "d5430190b2848a53138865bc1335581a9730a72a4debc7953fd47dfc513d02f7",
}
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


@lru_cache(maxsize=8000)
def molecule(smiles):
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None or not parsed.GetNumHeavyAtoms():
        raise ValueError("Invalid SMILES or empty molecular parent")
    parent = rdMolStandardize.ChargeParent(parsed)
    key = Chem.MolToInchiKey(parent)
    if len(key) != 27:
        raise ValueError("Unable to identify the molecular parent")
    return Chem.MolToSmiles(parent), key, GENERATOR.GetFingerprintAsNumPy(parent), Crippen.MolLogP(parent)


def checkpoint_verified(path):
    for name, checksum in CHECKPOINT.items():
        if digest(path / name) != checksum:
            raise ValueError(f"Unexpected CLAMP checkpoint artifact: {name}")


def extract(path, smiles, *, device="cpu", batch_size=1):
    checkpoint_verified(path)
    return embeddings(path, smiles, device=device, batch_size=batch_size)


def arrays(smiles, clamp=None):
    bits = np.stack([molecule(s)[2] for s in smiles]).astype(np.float32)
    logp = np.array([molecule(s)[3] for s in smiles], dtype=np.float32)[:, None]
    result = {"bits": bits, "clogp": logp}
    if clamp is not None:
        result["clamp"] = np.asarray(clamp, dtype=np.float32)
        result["combined"] = np.concatenate((bits, clamp), axis=1)
    return result


def nearest(bits, references):
    overlap = bits @ references.T
    union = bits.sum(axis=1)[:, None] + references.sum(axis=1) - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0).max(axis=1)
