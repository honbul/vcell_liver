"""Stereo-resolved molecular identity, grouping and structure-only features."""

from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
PROPERTIES = (Descriptors.MolWt, Crippen.MolLogP, Descriptors.TPSA, Lipinski.NumHDonors,
              Lipinski.NumHAcceptors, Lipinski.NumRotatableBonds, Lipinski.RingCount,
              Lipinski.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.MolMR,
              Lipinski.NumAromaticRings, Lipinski.NumAliphaticRings)


@lru_cache(maxsize=10000)
def identity(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES")
    parent = rdMolStandardize.FragmentParent(molecule)
    if parent.GetNumHeavyAtoms() == 0:
        raise ValueError("Empty parent molecule")
    Chem.GetSymmSSSR(parent)
    isomeric = Chem.MolToSmiles(parent, isomericSmiles=True)
    key = Chem.MolToInchiKey(parent)
    if not key:
        raise ValueError("Cannot generate parent InChIKey")
    connectivity = key.split("-")[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return isomeric, key, connectivity, scaffold or "acyclic:" + connectivity


def features(smiles):
    canonical, *_ = identity(smiles)
    molecule = Chem.MolFromSmiles(canonical)
    values = [descriptor(molecule) for descriptor in PROPERTIES]
    fp = MORGAN.GetFingerprint(molecule)
    return fp, np.r_[MORGAN.GetFingerprintAsNumPy(molecule), values].astype(np.float32)
