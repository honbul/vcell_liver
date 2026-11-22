"""Prepare deterministic bounded Uni-Mol conformers on a CPU allocation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_record, new_directory, verify_record, write_json
from .embed import molecular_queries, read_queries


MAX_HEAVY_ATOMS = 256
TIMEOUT_SECONDS = 25


def generate(query):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem

    key, smiles = query
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            return key, (), (), f"Exceeds Uni-Mol's {MAX_HEAVY_ATOMS} heavy-atom input limit"
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.timeout = TIMEOUT_SECONDS
        try:
            explicit = Chem.AddHs(molecule)
            if AllChem.EmbedMolecule(explicit, params) != 0:
                return key, (), (), "Bounded RDKit ETKDGv3 failed to generate a 3D conformer"
            if AllChem.UFFHasAllMoleculeParams(explicit):
                AllChem.UFFOptimizeMolecule(explicit, maxIters=200)
            molecule = Chem.RemoveHs(explicit)
            xyz = molecule.GetConformer().GetPositions()
        except (ValueError, RuntimeError) as error:
            return key, (), (), f"RDKit conformer error: {type(error).__name__}"
    if not np.isfinite(xyz).all():
        return key, (), (), "RDKit generated nonfinite coordinates"
    return key, tuple(atom.GetSymbol() for atom in molecule.GetAtoms()), xyz, None


def checked_conformers(directory, input_file, queries):
    receipt = json.loads((directory / "receipt.json").read_text())
    verify_record(input_file, receipt["input"])
    filename = directory / "coordinates.npz"
    verify_record(filename, receipt["coordinates"])
    with np.load(filename, allow_pickle=False) as data:
        keys = data["ids"].tolist()
        offsets = data["offsets"]
        extent = data["xyz"]
        symbols = data["symbols"]
    if (len(keys) != len(set(keys)) or not set(keys) <= {key for key, _ in queries}
            or offsets.shape != (len(keys) + 1,) or offsets[0] != 0 or offsets[-1] != len(extent)
            or len(symbols) != len(extent) or (np.diff(offsets) < 1).any()
            or extent.shape[1:] != (3,) or not np.isfinite(extent).all()):
        raise ValueError("Invalid registered conformer axes")
    atoms = [symbols[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]
    coords = [extent[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    return keys, atoms, coords, receipt["failures"]


def prepare(a):
    start = perf_counter()
    from rdkit.rdBase import rdkitVersion

    original = read_queries(a.input, "smiles")
    queries, failures = molecular_queries(original)
    keys, atoms, xyz, boundaries = [], [], [], [0]
    with ProcessPoolExecutor(max_workers=a.workers) as workers:
        for i, (key, symbols, points, error) in enumerate(workers.map(generate, queries, chunksize=2), 1):
            if error:
                failures[key] = error
            else:
                keys.append(key)
                atoms.extend(symbols)
                xyz.append(points)
                boundaries.append(len(atoms))
            if i % 100 == 0:
                print(f"Coordinates prepared for {i}/{len(queries)} inputs", flush=True)
    if not keys or set(keys) & set(failures) or len(keys) + len(failures) != len(original):
        raise ValueError("Incomplete conformer roster")
    with new_directory(a.output) as staging:
        coordinates = staging / "coordinates.npz"
        np.savez(coordinates, ids=np.asarray(keys, dtype=str), symbols=np.asarray(atoms, dtype=str),
                 xyz=np.concatenate(xyz), offsets=np.asarray(boundaries, dtype=np.int64))
        write_json(staging / "receipt.json", {
            "input": file_record(a.input), "coordinates": file_record(coordinates),
            "rdkit": rdkitVersion, "method": "ETKDGv3 seed 42, 200 UFF iterations when parameters are available",
            "timeout_seconds_per_molecule": TIMEOUT_SECONDS, "max_heavy_atoms": MAX_HEAVY_ATOMS,
            "workers": a.workers, "wall_seconds": perf_counter() - start,
            "n_requested": len(original), "n_conformers": len(keys), "failures": failures,
        })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    prepare(a)


if __name__ == "__main__":
    main()
