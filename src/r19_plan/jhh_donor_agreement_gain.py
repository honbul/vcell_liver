"""Test gain modulation by the consistency of measured same-drug donor responses."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zlib import crc32

import numpy as np
import pandas as pd

from src.r11_chemical_probe.common import read_json, sha256, write_json
from src.r19_plan import jhh_donor_variance as lineage
from src.r19_plan.hcc_lincs_24h_transfer import effect, group_means
from src.r19_plan.jhh5_donor_attribution import shrink_consensus
from src.r19_plan.jhh5_off_position import CALIBRATED, COUNT, SHRINK
from src.r19_plan.t4_position import compare
from src.r19_plan.t4_workflow import scalar


CANDIDATE = "agreement_gain"
ENERGY = "source_norm_gain"
SHUFFLE = "shuffled_agreement_gain"
ARMS = (CALIBRATED, CANDIDATE, ENERGY, SHUFFLE)
PAIRS = ((CANDIDATE, CALIBRATED), (CANDIDATE, ENERGY), (CANDIDATE, SHUFFLE), (ENERGY, CALIBRATED))
RIDGE = .05


def source_statistics(panels):
    averaged = panels.mean(axis=1)
    log_rms = np.log(np.maximum(np.sqrt(np.square(averaged).mean(axis=1)), 1e-8))
    norms = np.linalg.norm(panels, axis=2, keepdims=True)
    unit = np.divide(panels, norms, out=np.zeros_like(panels), where=norms>0)
    i,j = np.triu_indices(COUNT,k=1)
    agreement = np.sum(unit[:,i]*unit[:,j], axis=2).mean(axis=1)
    return np.column_stack((log_rms, agreement))


def resampled_agreement(features, parents, identifier):
    support, members = np.unique(parents, return_index=True)
    if len(support)<2:
        raise ValueError("Agreement shuffle needs at least two different parents")
    unique = features[members]
    rows = np.lexsort((support, unique[:,0]))
    mapped = unique[:,1].copy()
    rng = np.random.default_rng(crc32(f"190711|{identifier}".encode()))
    for block in np.array_split(rows,max(1,(len(rows)+4)//8)):
        offset = int(rng.integers(1,len(block)))
        mapped[block] = unique[np.roll(block,offset),1]
    answer = features.copy()
    answer[:,1] = pd.Series(mapped,index=support).loc[parents].to_numpy()
    return answer


def model_gain(tuning, testing, x, y, *, agreement):
    mean, scale = tuning.mean(axis=0), tuning.std(axis=0)
    scale = np.where(scale>1e-8,scale,1.)
    def design(t):
        n = (t-mean)/scale
        return np.column_stack((np.ones(len(n)), n))
    train, query = design(tuning), design(testing)
    power = np.einsum("ij,ij->i",x,x,dtype=np.float64)
    cross = np.einsum("ij,ij->i",x,y,dtype=np.float64)
    if power.sum()<=0:
        raise ValueError("No training energy to fit the gain")
    gram = train.T@(train*power[:,None])
    gram += RIDGE*power.sum()*np.diag([0.,1.,1.][:train.shape[1]])
    rhs = train.T@cross
    beta = np.linalg.solve(gram,rhs)
    if agreement and beta[-1] < 0:
        beta[:-1] = np.linalg.solve(gram[:-1,:-1],rhs[:-1])
        beta[-1] = 0.
    return np.clip(query@beta,0,2),beta


def predict(args, old, ref, genes, cols, prior, stamp):
    role = args.role
    inventory, panels = lineage.donor_profiles(args,old,ref,genes,cols,prior,role)
    means = panels.mean(axis=1)
    shrink = shrink_consensus(panels,[COUNT]*len(panels),means)
    feature = source_statistics(panels)
    root = args.run/"predictions"/role
    root.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root.parent,prefix=".gain-") as work:
        stage=Path(work)
        entries, audit = {}, []
        for key in lineage.keys(args,ref,role):
            plan = lineage.tables(args,old,ref,"predict",key)
            fit, query = plan["fit"],plan["query"]
            if (set(query.parent)&set(fit.parent) or set(query.scaffold)&set(fit.scaffold)
                    or set(query.rna_plate)&set(fit.rna_plate) or "level3_row_index" in query
                    or not query.cell_id.eq(args.cell).all() or not fit.cell_id.eq("JHH5").all()):
                raise ValueError("Gain estimation must remain outside the held parent, scaffold and RNA plate")
            labels, y = group_means(effect(fit,plan["fit_controls"],ref,genes,cols,role),fit,["parent"])
            i,j = (inventory.get_indexer(f) for f in (labels.parent,query.parent))
            if (i<0).any() or (j<0).any():
                raise ValueError("A target lacks its locked donor statistics")
            original = lineage.historical(args,prior,key,(len(query),len(genes)))
            base = scalar(labels,shrink[i],y)
            np.testing.assert_array_equal((base*shrink[j]).astype("float32"),original[CALIBRATED])
            np.testing.assert_array_equal(shrink[j].astype("float32"),original[SHRINK])
            shifted_fit = resampled_agreement(feature[i],labels.parent,f"{key}|fit")
            shifted_query = resampled_agreement(feature[j],query.parent,f"{key}|query")
            arms = {CALIBRATED:original[CALIBRATED]}
            for arm,train,test in ((CANDIDATE,feature[i],feature[j]),(ENERGY,feature[i,:1],feature[j,:1]),
                                   (SHUFFLE,shifted_fit,shifted_query)):
                gain,beta = model_gain(train,test,shrink[i],y,agreement=arm!=ENERGY)
                arms[arm]=(gain[:,None]*shrink[j]).astype("float32")
                audit.append(dict(slice=key,arm=arm,fit_parents=len(i),query_wells=len(j),original_gain=base,
                    intercept=beta[0],norm_slope=beta[1],agreement_slope=beta[2] if len(beta)==3 else None,
                    query_gain_min=gain.min(),query_gain_median=np.median(gain),query_gain_max=gain.max()))
            entry=stage/(key+".npz")
            np.savez_compressed(entry,**arms)
            entries[key]=sha256(entry)
        write_json(stage/"receipt.json",dict(stamp,role=role,target_expression_read=False,files=entries,fits=audit))
        stage.rename(root)


def finish(args,old,ref,genes,cols,prior,stamp):
    path=args.run/"freeze.json"
    scoring=args.phase=="score"
    if scoring and (sha256(path)!=args.digest or (args.run/"score").exists()):
        raise ValueError("The gain comparison requires a separate frozen forecast")
    if not scoring and path.exists():
        raise FileExistsError(path)
    manifest,rows,queries,estimates,audit=lineage.verified_forecasts(args,old,ref,genes,stamp,arms=ARMS,seedwise=())
    if not scoring:
        write_json(path,manifest)
        return
    if read_json(path)!=manifest:
        raise ValueError("Forecast artifacts differ from the independent freeze")
    truth=[]
    for key,(role,query) in queries.items():
        plan=lineage.tables(args,old,ref,"score",key)
        target=plan["query"]
        if not target.drop(columns="level3_row_index").equals(query):
            raise ValueError("A gain forecast misses its original score well")
        ctl=plan["score_controls"] if args.cell=="JHH5" else plan["controls"]
        truth.append(effect(target,ctl,ref,genes,cols,"ABC"[("ABC".index(role)+2)%3]))
    y=np.concatenate(truth)
    arrays={a:np.concatenate([v[a] for v in estimates.values()]) for a in ARMS}
    absolute=args.previous/"score/paired.tsv"
    original=read_json(args.previous_score_pin)
    if (sha256(args.previous/"score/receipt.json")!=sha256(args.previous_score_pin)
            or sha256(absolute)!=original["files"]["paired.tsv"]):
        raise ValueError("The comparison must reproduce the original score support")
    registered=pd.read_csv(absolute,sep="\t")
    base=registered.loc[registered.scope.eq("all")&registered.arm.eq(CALIBRATED)&registered.reference.eq("scalar")]
    values=compare(rows,y,{CALIBRATED:arrays[CALIBRATED],"B0":np.zeros_like(y)},((CALIBRATED,"B0"),),"all")
    np.testing.assert_allclose(base.arm_value,[row["arm_value"] for row in values],atol=1e-12,rtol=0)
    full=compare(rows,y,arrays,PAIRS,"all")
    folds=[]
    for k in range(3):
        keep=rows.fold.eq(k)
        folds.extend(compare(rows.loc[keep],y[keep],{arm:v[keep] for arm,v in arrays.items()},PAIRS,k))
    with TemporaryDirectory(dir=args.run,prefix=".gain-score-") as work:
        stage=Path(work)
        for name,table in (("paired",full),("folds",folds),("gains",audit)):
            pd.DataFrame(table).to_csv(stage/(name+".tsv"),sep="\t",index=False,na_rep="NA")
        write_json(stage/"receipt.json",dict(manifest,freeze_sha256=args.digest,original_point_replayed=True,
            files={f.name:sha256(f) for f in stage.iterdir()}))
        stage.rename(args.run/"score")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("predict","freeze","score"))
    parser.add_argument("--cell",choices=tuple(lineage.SCOPE),required=True)
    parser.add_argument("--role",choices=tuple("ABC"))
    parser.add_argument("--digest")
    for name in ("run","previous","previous-pin","previous-score-pin","access-pin","previous-protocol","native-protocol","protocol"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    *other,stamp=lineage.references(args)
    stamp.update(code_sha256=sha256(__file__),shared_access_code_sha256=sha256(lineage.__file__))
    (predict if args.phase=="predict" else finish)(args,*other,stamp)
