"""Molecule-only inference with the public ICML 2023 CLAMP checkpoint."""
import json

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from torch import nn


def input_fingerprint(smiles, size):
    molecule = Chem.MolFromSmiles(smiles)
    Chem.FastFindRings(molecule)
    with rdBase.BlockLogs():
        morgan = AllChem.GetMorganFingerprint(
            molecule, 2, useChirality=True, useBondTypes=True, useFeatures=True, useCounts=True,
        )
        paths = AllChem.UnfoldedRDKFingerprintCountBased(molecule, maxPath=6)
    # The pretrained "morganc+rdkc" contract folds into the full common
    # 8192-bin axis. Preserve the legacy author code's overwrite order for
    # intra-family collisions, add the two families, then apply log1p.
    fingerprint = np.zeros(size, dtype=np.float32)
    for sparse in (morgan, paths):
        family = np.zeros(size, dtype=np.float32)
        for key, count in sparse.GetNonzeroElements().items():
            family[key % size] = count
        fingerprint += family
    return np.log1p(fingerprint)


def embeddings(checkpoint, values, device, batch_size):
    metadata = json.loads((checkpoint / "hp.json").read_text())
    if metadata["compound_mode"] != "morganc+rdkc" or (
        metadata["model"], metadata["nonlinearity"]
    ) != ("ScaledMLPLayerNorm", "ReLU"):
        raise ValueError("CLAMP checkpoint is incompatible with folded molecule inference")
    size = metadata["compound_features_size"]
    dimensions = [size, *metadata["compound_layer_sizes"], metadata["embedding_size"]]
    # The official checkpoint also stores optimizer state and a NumPy
    # validation scalar. Keep PyTorch deserialization in weights-only mode.
    allowed = [np.core.multiarray.scalar, np.dtype, np.dtypes.Float32DType,
               np.dtypes.Float64DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        checkpoint_data = torch.load(checkpoint / "checkpoint.pt", mmap=True,
                                     map_location="cpu", weights_only=True)
    compounds = {key.removeprefix("compound_encoder."): value
                 for key, value in checkpoint_data["model_state_dict"].items()
                 if key.startswith("compound_encoder.")}
    stem, *hidden, output = list(zip(dimensions, dimensions[1:]))
    layers = [
        ("linear_input", stem),
        *((f"linear_hidden_l.{i}", shape) for i, shape in enumerate(hidden)),
        ("linear_output", output),
    ]
    model = []
    for name, (in_features, out_features) in layers:
        projection = nn.Linear(in_features, out_features)
        projection.load_state_dict({k: compounds.pop(f"{name}.{k}") for k in ("weight", "bias")}, strict=True)
        model.append(projection)
        if name != "linear_output":
            model.extend((nn.LayerNorm(out_features, elementwise_affine=False), nn.ReLU()))
    if compounds:
        raise ValueError("CLAMP molecule checkpoint has unsupported layer parameters")
    network = nn.Sequential(*model).to(device).eval()
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(values), batch_size):
            batch = np.stack([input_fingerprint(value, size) for value in values[offset:offset + batch_size]])
            chunks.append(network(torch.from_numpy(batch).to(device)).cpu().numpy())
    return np.concatenate(chunks)
