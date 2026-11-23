"""Materialize a common fold-local atlas for tree, neural and hybrid training."""
import argparse
from hashlib import sha256 as digest_bytes
from time import perf_counter
import pickle

import numpy as np

from vc_build.io import publish_json, sha256
from nextgen_h2.run import read_json
from .data import ASSETS, ROOT, SPLITS, nested, read_data, table_hash
from .features import SourceFeatures, active_scores, build_atlas, checked_embedding, load_registered, positions, normalize_fit
from .graphs import make_graphs


def save(fit, query, data, dest):
    begin = perf_counter()
    dest.mkdir(parents=True, exist_ok=False)
    atlas, schema, cells, drugs, dimensions = build_atlas(data, fit, query)
    fa, fb = positions(fit, cells, drugs)
    qa, qb = positions(query, cells, drugs)
    fit_drugs = np.unique(fb)
    graph = make_graphs(schema["molecules"].to_list())
    arrays = atlas | graph | {"fit_cells":fa,"fit_drugs":fb,"query_cells":qa,"query_drugs":qb,
                              "cell_ids":cells.to_numpy(dtype=str),"drug_ids":drugs.to_numpy(dtype=str)}
    embedding = {}
    for name in ("clamp","unimol"):
        raw, have, digest = checked_embedding(name, drugs)
        if np.count_nonzero(have[fit_drugs]) < 20:
            raise ValueError("Insufficient validated pretrained chemistry support")
        vector, parameters = normalize_fit(raw,fit_drugs[have[fit_drugs]])
        arrays[name] = vector
        arrays[name+"_mask"] = have
        schema[name+"_scale"] = parameters
        embedding[name] = {"available":int(have.sum()),"width":vector.shape[1],"sha256":digest}
    file = dest / "atlas.npz"
    with file.open("xb") as stream:
        np.savez(stream, **arrays)
    path = dest / "schema.pkl"
    with path.open("xb") as stream:
        pickle.dump(schema,stream,protocol=5)
    publish_json(dest/"receipt.json", {
        "atlas_sha256":sha256(file),"schema_sha256":sha256(path),"dimensions": dimensions,
        "fit_rows":len(fit),"held_rows":len(query),"fit_id_sha256":table_hash(fit),
        "held_id_sha256":table_hash(query),"pretrained":embedding,"seconds":perf_counter()-begin,
        "held_endpoint_values_used":False,
    })
    print(dest, dimensions, flush=True)


def prepare(split):
    source = SourceFeatures.read()
    fit, query = read_data(split,"fit"), read_data(split,"query")
    root = ROOT/"features"/split
    for fold in range(3):
        train, val = nested(fit, split, source.registry, fold)
        save(train,val,source,root/f"inner-{fold}")
    save(fit,query,source,root/"outer")


def read(root, fit, query):
    record=read_json(root/"receipt.json")
    path=root/"atlas.npz"
    if (sha256(path),table_hash(fit),table_hash(query)) != (
        record["atlas_sha256"],record["fit_id_sha256"],record["held_id_sha256"]
    ):
        raise ValueError("Training feature parity seal drifted")
    with np.load(path,allow_pickle=False) as reader:
        atlas=dict(reader)
    ids=atlas["drug_ids"]
    fit_drugs=np.unique(atlas["fit_drugs"])
    measured=atlas["lincs_mask"]
    source=read_json(ASSETS/"data_receipt.json")["files"]
    digests=[]
    for dataset, key in (("lincs_modz24.parquet","lincs"),("paths_modz24.parquet","perturb_paths")):
        raw=load_registered(dataset).reindex(ids).to_numpy(np.float32)
        if not np.array_equal(np.isfinite(raw).all(1),measured):
            raise ValueError("LINCS reattachment cannot change natural measurement support")
        atlas[key],_=active_scores(raw,measured,fit_drugs)
        digests.append(source[dataset])
    exact=record["atlas_sha256"]+"|"+"|".join(digests)+"|lincs_full_unclipped"
    return atlas, record | {"model_input_sha256":digest_bytes(exact.encode()).hexdigest(),
                           "lincs_full_unclipped":True}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("split",choices=SPLITS)
    prepare(p.parse_args().split)
