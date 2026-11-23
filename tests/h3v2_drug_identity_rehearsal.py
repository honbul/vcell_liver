from collections import defaultdict
from hashlib import sha256

import numpy as np
from h3v2.data import ROOT, SPLITS

violations = {}
roles = ("outer", "inner-0", "inner-1", "inner-2")
parts = 0
for split in SPLITS:
    for role in roles:
        path = ROOT / "features" / split / role / "atlas.npz"
        with np.load(path, allow_pickle=False) as atlas:
            fitted = set(np.unique(atlas["fit_drugs"]))
            keys = ("bits", "props", "targets", "lincs_mask", "mu")
            same = defaultdict(list)
            for i, blocks in enumerate(zip(*(atlas[key] for key in keys))):
                token = sha256(b"".join(np.asarray(value).tobytes() for value in blocks)).hexdigest()
                same[token].append(i)
            collisions = [members for members in same.values()
                          if len(members) > 1 and fitted.intersection(members)]
            if collisions:
                violations[f"{split}/{role}"] = len(collisions)
        parts += 1
print(f"{parts} atlases; collisions involving any fitting parent: {violations}", flush=True)
if violations:
    raise ValueError("Some exact training parents cannot be distinguished by the common chemistry and fit anchor")
