"""Store only predicted mean expression, in the frozen gene and condition order."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from artifact_io import digest


def export(path, genes, methods):
    content = {'gene_names':np.asarray(genes,dtype=str)}
    for method, predictions in methods.items():
        order = sorted(predictions)
        content[method+'_condition'] = np.asarray(order,dtype=str)
        values = (np.stack([predictions[c] for c in order]) if order
                  else np.empty((0,len(genes)),dtype=np.float64))
        if values.shape != (len(order),len(genes)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid predicted expression vectors: {method}')
        content[method+'_expression_mean'] = values
    with TemporaryDirectory(prefix='predictions-',dir=path.parent) as tmp:
        candidate = Path(tmp)/'predictions.npz'
        np.savez(candidate,**content)
        candidate.replace(path)
    return digest(path)
