"""Small phase-2 helpers. Legacy CLI, caches and evaluation behavior stay intact."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from common import DATA,Resources,sha,write_json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,classification_report,confusion_matrix,f1_score,log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPORT=ROOT/'reports/research_phase2'
ASSETS=DATA/'research_phase2'

def git_commit():return subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()

def donor_roles(obs,c,fold):
    frame=pd.read_csv(ROOT/c['split_manifest'],sep='\t');frame=frame[frame.fold==fold]
    assert len(frame)==8 and not frame.patient_id.isin(c['legacy_donors']).any()
    roles=obs.patient_id.map(frame.set_index('patient_id').role)
    assert roles.notna().all()
    result={name:np.flatnonzero(roles.to_numpy()==name) for name in ['inner_training','inner_validation','evaluation']}
    cap=c.get('train_cell_cap')
    if cap:
        outer=np.concatenate([result['inner_training'],result['inner_validation']])
        if len(outer)>cap:
            chosen=np.random.default_rng(c['seed']+fold).choice(outer,cap,replace=False)
            for role in ['inner_training','inner_validation']:result[role]=np.intersect1d(result[role],chosen)
    return result

def metric_summary(y,pred,prob,labels):
    y=np.asarray(y,dtype=str);pred=np.asarray(pred,dtype=str)
    known=np.isin(y,labels);answered=pred!='Unknown';k=known & answered
    result=dict(macro_f1=float(f1_score(y[known],pred[known],labels=labels,average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y[known],pred[known])),n_cells=len(y),n_known=int(known.sum()),
        evaluation_coverage=float(known.mean()),prediction_coverage=float(answered.mean()),n_unknown_truth=int((~known).sum()),n_abstained=int((~answered).sum()),
        confidence_calibration='uncalibrated; no confidence threshold; ECE/Brier reported',labels=labels)
    result['answered_only_macro_f1']=float(f1_score(y[k],pred[k],labels=labels,average='macro',zero_division=0)) if k.any() else None
    if prob is not None:
        assert prob.shape==(len(y),len(labels)) and np.isfinite(prob).all()
        truth=np.zeros_like(prob);lookup={v:i for i,v in enumerate(labels)}
        for i,t in enumerate(y):
            if t in lookup:truth[i,lookup[t]]=1
        result['brier_multiclass']=float(np.mean(np.sum((prob[known]-truth[known])**2,axis=1)))
        confidence=prob.max(1);correct=pred==y;ece=0.
        for left,right in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
            ix=known&(confidence>left)&(confidence<=right)
            if ix.any():ece+=ix.sum()/known.sum()*abs(float(correct[ix].mean()-confidence[ix].mean()))
        result['ece_10_bins']=ece
    return result

def fit_head(features_inner,features_final,y,roles,c,seed):
    tr,va,te=[roles[k] for k in ['inner_training','inner_validation','evaluation']]
    scaler=StandardScaler().fit(features_inner[tr]);x=scaler.transform(features_inner)
    best=None;grid=[];start=time.monotonic()
    for C in c['head_C']:
        for weight in c['head_class_weight']:
            head=LogisticRegression(C=C,class_weight=weight,max_iter=2000,random_state=seed).fit(x[tr],y[tr])
            score=f1_score(y[va],head.predict(x[va]),labels=c['labels'],average='macro',zero_division=0)
            grid.append(dict(C=C,class_weight=weight,validation_macro_f1=float(score)))
            if best is None or score>best[0]:best=(score,C,weight)
    outer=roles.get('outer_training',np.concatenate([tr,va]));scaler=StandardScaler().fit(features_final[outer]);x=scaler.transform(features_final)
    head=LogisticRegression(C=best[1],class_weight=best[2],max_iter=2000,random_state=seed).fit(x[outer],y[outer])
    training_time=time.monotonic()-start;start=time.monotonic()
    pred=head.predict(x[te]);native_prob=head.predict_proba(x[te]);prob=np.zeros((len(te),len(c['labels'])))
    for j,label in enumerate(head.classes_):prob[:,c['labels'].index(label)]=native_prob[:,j]
    return pred,prob,dict(scaler=scaler,classifier=head),dict(validation_grid=grid,selected_C=best[1],selected_class_weight=best[2],validation_macro_f1=best[0]),training_time,time.monotonic()-start

def save_evaluation(a,test,pred,prob,c,run_id,metadata,head=None):
    out=REPORT/'runs'/run_id
    assert not (out/'run.json').exists(),'Run IDs are immutable; use a new ID'
    out.mkdir(parents=True,exist_ok=True)
    raw=ASSETS/'runs'/run_id;raw.mkdir(parents=True,exist_ok=True)
    labels=c['labels'];obs=a.obs.iloc[test];y=obs.cell_type_coarse.astype(str).to_numpy()
    metrics=metric_summary(y,pred,prob,labels)
    p=obs[['cell_id','patient_id','cell_type_coarse']].copy();p=p.rename(columns={'cell_type_coarse':'truth'});p['prediction']=pred;p['confidence']=prob.max(1) if prob is not None else np.nan
    p.to_csv(raw/'predictions.tsv',sep='\t',index=False)
    if prob is not None:np.savez_compressed(raw/'probabilities.npz',cell_ids=obs.index.to_numpy(dtype=str),probabilities=prob,labels=np.array(labels))
    pd.DataFrame(classification_report(y,pred,labels=labels,output_dict=True,zero_division=0)).T.to_csv(out/'per_class.tsv',sep='\t')
    pd.DataFrame(confusion_matrix(y,pred,labels=labels+(['Unknown'] if 'Unknown' in pred else [])),index=labels+(['Unknown'] if 'Unknown' in pred else []),columns=labels+(['Unknown'] if 'Unknown' in pred else [])).to_csv(out/'confusion_matrix.tsv',sep='\t')
    patients=[]
    for donor in sorted(obs.patient_id.unique()):
        ix=obs.patient_id.to_numpy()==donor
        supported=sorted(set(y[ix])|set(pred[ix]))
        patients.append(dict(patient_id=donor,n_cells=int(ix.sum()),macro_f1_common_ontology=float(f1_score(y[ix],pred[ix],labels=labels,average='macro',zero_division=0)),
             macro_f1_observed_or_predicted_classes=float(f1_score(y[ix],pred[ix],labels=supported,average='macro',zero_division=0)),class_support=json.dumps(pd.Series(y[ix]).value_counts().to_dict())))
    pd.DataFrame(patients).to_csv(out/'by_donor.tsv',sep='\t',index=False)
    if head is not None:
        import joblib
        joblib.dump(head,raw/'head.joblib')
    write_json(out/'metrics.json',metrics)
    run=dict(run_id=run_id,status='executed',task='annotation',regime=c['evaluation_regime'],dataset_version=c['dataset_id'],split_id=c['split_id'],
        project_commit=git_commit(),seed=c['seed'],data_variant=c['data_variant'],model_variant=c['model_variant'],
        label_mapping='original six broad lineages, unchanged',pretraining_overlap='unknown',
        split_manifest=str(ROOT/c['split_manifest']),split_manifest_sha256=sha(ROOT/c['split_manifest']),
        allowed_target_information='own observed counts only; no fitting to outer evaluation donors',
        prediction_path=str(raw/'predictions.tsv'),prediction_sha256=sha(raw/'predictions.tsv'),metrics=metrics)
    run.update(metadata)
    write_json(out/'run.json',run)
    print(json.dumps(dict(run_id=run_id,model_id=run['model_id'],**metrics)),flush=True)
    return run
