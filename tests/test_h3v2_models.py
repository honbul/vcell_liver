import numpy as np
import torch

from h3v2.graphs import make_graphs
from h3v2.grid import initial
from h3v2.network import CrossAttention,Film,on_device
from h3v2.objectives import batches,objectives


def fixture():
    rng=np.random.default_rng(42)
    shape={"rna":(15,19),"activity":(15,21),"proxy":(15,7),"paths":(15,44),"bits":(4,31),
           "props":(4,4),"targets":(4,12),"lincs":(4,13),"perturb_paths":(4,44),
           "target_match":(15,4,5),"mu":(4,),"clamp":(4,10),"unimol":(4,12)}
    arrays={k:rng.normal(size=v).astype(np.float32) for k,v in shape.items()}
    obs=np.array([True,False,False,True])
    arrays |= {"lincs_mask":obs,"clamp_mask":obs,"unimol_mask":obs}
    for key in ("lincs","perturb_paths","clamp","unimol"):
        arrays[key][~obs]=np.nan
    return arrays|make_graphs(["CCC","C1CC1","CCCCO","CN(Cl)C"])


def test_every_neural_family_learns_with_masked_modalities():
    arr=fixture()
    arr["rna"]=arr["rna"].astype(np.float64)
    at=on_device(arr,"cpu")
    assert at["rna"].dtype==torch.float32
    i,j=torch.arange(15).repeat_interleave(4),torch.arange(4).repeat(15)
    cases=[initial("film")[0],initial("attention")[0],*initial("gnn"),initial("hybrid")[-1]]
    for cfg in cases:
        settings=cfg|{"width":32,"heads":4,"depth":2}
        cls=CrossAttention if settings["arch"]=="attention" else Film
        model=cls(at,settings)
        model.eval()
        base=torch.rand(len(i)) if settings["hybrid"] else None
        p,alpha,delta=model(at,i,j,base)
        changed={k:v.clone() for k,v in at.items()}
        changed["lincs"][~at["lincs_mask"]]=999999.
        changed["perturb_paths"][~at["lincs_mask"]]=-999999.
        q=model(changed,i,j,base)[0]
        assert torch.isfinite(p).all()
        torch.testing.assert_close(p,q)
        loss=objectives(p,torch.rand(len(i)),j,i,alpha,delta)
        loss.sum().backward()
        assert any(x.grad is not None and x.grad.norm()>0 for x in model.parameters())


def test_regular_blocks_use_all_observed_rows_exactly_once():
    rng=np.random.default_rng(13)
    a,b,s=(rng.integers(n,size=2001) for n in (45,70,3))
    seen=np.sort(np.concatenate(list(batches(a,b,s,rng,45,70,12,23))))
    np.testing.assert_array_equal(seen,np.arange(len(a)))


if __name__=="__main__":
    test_every_neural_family_learns_with_masked_modalities()
    test_regular_blocks_use_all_observed_rows_exactly_once()
    print("H3-v2 neural feature masks and observed-exposure tests passed")
