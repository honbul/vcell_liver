"""One fixed external batch: known-label metrics, Unknown accounting and paired donor intervals."""
import anndata as ad
from sklearn.metrics import confusion_matrix
from research_core import *

out=REPORT/'external_final';assert not (out/'evaluation.json').exists();freeze=json.loads((out/'freeze.json').read_text());assert freeze['status']=='frozen_before_predictions';assert sha(freeze['truth_path'])==freeze['truth_sha256']
a=ad.read_h5ad(freeze['truth_path']);labels=['B','Endothelial','Fibroblast','Hepatocyte','Myeloid','T/NK'];five=[v for v in labels if v!='Hepatocyte'];truth=a.obs.cell_type_coarse.astype(str).to_numpy();known=np.isin(truth,labels);donors=sorted(a.obs.patient_id.unique());counts=[];rows=[];patient_rows=[];sources=[]
for name,model in freeze['models'].items():
    root=Path(freeze['prediction_root'])/name;meta=json.loads((root/'inference.json').read_text());assert meta['status']=='executed' and meta['bundle_sha256']==model['bundle_sha256'] and meta['input_data_sha256']==freeze['query_sha256']
    assert sha(root/'predictions.tsv')==meta['prediction_sha256'] and sha(root/'probabilities.npz')==meta['probability_sha256'];pred=pd.read_csv(root/'predictions.tsv',sep='\t');prob=np.load(root/'probabilities.npz');assert np.array_equal(pred.cell_id,a.obs_names.astype(str)) and np.array_equal(prob['cell_ids'],a.obs_names.astype(str)) and list(prob['labels'])==labels
    predictions=pred.prediction.astype(str).to_numpy();metrics=metric_summary(truth,predictions,prob['probabilities'],labels);metrics['macro_f1_fixed_five']=float(f1_score(truth[known],predictions[known],labels=five,average='macro',zero_division=0));rows.append(dict(candidate=name,n_training_cells=model['n_training_cells'],**{k:v for k,v in metrics.items() if not isinstance(v,(list,dict))}))
    matrices=[]
    for donor in donors:
        ix=a.obs.patient_id.to_numpy()==donor;eligible=ix&known;cm=confusion_matrix(truth[eligible],predictions[eligible],labels=labels);matrices.append(cm)
        patient_rows.append(dict(candidate=name,patient_id=donor,n_cells=int(ix.sum()),n_known=int(eligible.sum()),n_unknown_truth=int((ix&~known).sum()),macro_f1_fixed_six=float(f1_score(truth[eligible],predictions[eligible],labels=labels,average='macro',zero_division=0)) if eligible.any() else None,macro_f1_fixed_five=float(f1_score(truth[eligible],predictions[eligible],labels=five,average='macro',zero_division=0)) if eligible.any() else None))
    counts.append(np.stack(matrices));sources.append(dict(candidate=name,inference_metadata_sha256=sha(root/'inference.json'),prediction_sha256=meta['prediction_sha256'],probability_sha256=meta['probability_sha256']))
    pd.DataFrame(confusion_matrix(truth,predictions,labels=labels+['Unknown']),index=labels+['Unknown'],columns=labels+['Unknown']).to_csv(out/f'{name}_confusion_including_unknown_truth.tsv',sep='\t')
    pd.DataFrame(classification_report(truth[known],predictions[known],labels=labels,output_dict=True,zero_division=0)).T.to_csv(out/f'{name}_per_class_known_truth.tsv',sep='\t')
counts=np.stack(counts);names=list(freeze['models']);rng=np.random.default_rng(42);draws=rng.integers(0,len(donors),size=(5000,len(donors)));weights=np.stack([(draws==i).sum(1) for i in range(len(donors))],axis=1)
def macro(cm,mask=None):
    tp=np.diagonal(cm,axis1=-2,axis2=-1);den=cm.sum(-1)+cm.sum(-2);score=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
    return score.mean(-1) if mask is None else score[...,mask].mean(-1)
boot=macro(np.einsum('bd,mdij->bmij',weights,counts));point=macro(counts.sum(1));pairs=[]
def compare(title,terms):
    value=sum(coef*point[names.index(name)] for name,coef in terms.items());samples=sum(coef*boot[:,names.index(name)] for name,coef in terms.items());pairs.append(dict(comparison=title,fixed_six_macro_f1_difference=float(value),lower95=float(np.quantile(samples,.025)),upper95=float(np.quantile(samples,.975)),n_donors=len(donors)))
for name in ['scgpt','geneformer','geneformer_clcancer']:compare(name+' vs PCA',{name:1,'pca':-1})
compare('CLcancer vs base Geneformer',{'geneformer_clcancer':1,'geneformer':-1});compare('4096 Data gain',{'DM10':1,'DM00':-1});compare('4096 Model gain',{'DM01':1,'DM00':-1});compare('4096 Combined gain',{'DM11':1,'DM00':-1});compare('4096 Interaction',{'DM11':1,'DM10':-1,'DM01':-1,'DM00':1})
pd.DataFrame(rows).to_csv(out/'metrics.tsv',sep='\t',index=False);pd.DataFrame(patient_rows).to_csv(out/'by_donor.tsv',sep='\t',index=False);pd.DataFrame(pairs).to_csv(out/'paired_donor_bootstrap.tsv',sep='\t',index=False)
pd.DataFrame([dict(candidate=name,fixed_six_macro_f1=float(point[i]),lower95=float(np.quantile(boot[:,i],.025)),upper95=float(np.quantile(boot[:,i],.975))) for i,name in enumerate(names)]).to_csv(out/'absolute_donor_bootstrap.tsv',sep='\t',index=False)
write_json(out/'evaluation.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),freeze_sha256=sha(out/'freeze.json'),n_cells=a.n_obs,n_donors=len(donors),n_known=int(known.sum()),n_unknown_truth=int((~known).sum()),label_support=a.obs.cell_type_coarse.value_counts().to_dict(),candidates=names,sources=sources,
    uncertainty='5000 paired donor bootstrap samples; repeated biopsies remain grouped; models held fixed, no retraining uncertainty',primary_pipeline='geneformer_frozen_linear_12000',
    limitations=['No gold hepatocyte annotation: fixed-six F1 maximum5/6; fixed-five secondary does not validate hepatocyte transfer','Unknown malignant/unclassified truth excluded from lineage accuracy and counted explicitly; predictions on these cells are not validated malignant lineage truth','Different provided gene panels and treatment histories create external shift','Pretraining overlap and public patient identity overlap unknown','After this batch no retuning or cohort switching; future use for model changes consumes holdout independence']))
print('EXTERNAL_FINAL_COMPLETE',pd.DataFrame(rows)[['candidate','macro_f1','macro_f1_fixed_five']].to_string(index=False),flush=True)
