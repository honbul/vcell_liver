"""Matched-size training-only sampling; validation and evaluation IDs remain fixed."""
from research_core import *

def balanced_without_replacement(obs,indices,size,columns,seed):
    assert size<=len(indices)
    rng=np.random.default_rng(seed);groups={}
    for i in indices:
        key=tuple(str(obs.iloc[i][column]) for column in columns);groups.setdefault(key,[]).append(int(i))
    keys=sorted(groups);pools={k:rng.permutation(groups[k]) for k in keys};take={k:0 for k in keys}
    remaining=size
    while remaining:
        progressed=False
        for key in keys:
            if take[key]<len(pools[key]):take[key]+=1;remaining-=1;progressed=True
            if not remaining:break
        assert progressed
    result=np.sort(np.concatenate([pools[k][:take[k]] for k in keys]));assert len(result)==size and len(np.unique(result))==size
    return result

def sampling_roles(obs,c,fold,variant):
    roles=donor_roles(obs,c,fold);roles['outer_training']=np.concatenate([roles['inner_training'],roles['inner_validation']])
    if variant=='D0_natural':return roles
    columns={'D1_donor_balanced':['patient_id'],'D2_donor_class_balanced':['patient_id','cell_type_coarse']}[variant]
    uncapped=dict(c,train_cell_cap=None);candidates=donor_roles(obs,uncapped,fold)
    roles['inner_training']=balanced_without_replacement(obs,candidates['inner_training'],len(roles['inner_training']),columns,c['seed']+fold+10000)
    roles['outer_training']=balanced_without_replacement(obs,np.concatenate([candidates['inner_training'],candidates['inner_validation']]),len(roles['outer_training']),columns,c['seed']+fold+20000)
    assert not np.intersect1d(roles['inner_training'],roles['inner_validation']).size
    assert not np.intersect1d(roles['outer_training'],roles['evaluation']).size
    return roles
