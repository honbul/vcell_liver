"""Exercise query-only annotation on 64 real development cells, without external access."""
import argparse,subprocess
import anndata as ad
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--probe-id',default='annotation_query_interface_v2');args=p.parse_args()
assert Path(args.probe_id).name==args.probe_id
root=ASSETS/'probes'/args.probe_id;root.mkdir(parents=True,exist_ok=True);a=ad.read_h5ad(ASSETS/'data/development_full.h5ad')[:64].copy();a.obs=a.obs[['cell_id','patient_id','n_counts']].copy();path=root/'counts_only.h5ad';assert not path.exists();a.write_h5ad(path)
python=str(ROOT/'.venvs/vcell-runtime310/bin/python');bootstrap=str(ROOT/'envs/portable310.py');scripts=ROOT/'scripts/research';records=[]
for model in ['pca','scgpt','geneformer','geneformer_clcancer']:
    feature=[]
    if model!='pca':
        subprocess.run([python,bootstrap,'--env','scgpt' if model=='scgpt' else 'geneformer',str(scripts/'extract_query_features.py'),'--input',str(path),'--model',model,'--cache-name',args.probe_id],check=True)
        feature=['--features',str(ASSETS/'query_features'/f'{model}_{args.probe_id}.npz')]
    output=root/model;version=2 if model=='pca' else 1;bundle=ASSETS/'runs'/f'F00_{model}_development12000_seed42_v{version}/annotation_bundle.joblib'
    subprocess.run([python,bootstrap,'--env','research',str(scripts/'predict_annotation.py'),'--input',str(path),'--bundle',str(bundle),'--output',str(output)]+feature,check=True)
    pred=pd.read_csv(output/'predictions.tsv',sep='\t');prob=np.load(output/'probabilities.npz');assert np.array_equal(pred.cell_id,a.obs_names.astype(str)) and np.allclose(prob['probabilities'].sum(1),1) and set(pred.prediction)<=set(prob['labels'])
    record=dict(model_id=model,n_cells=len(pred),input_columns=a.obs.columns.tolist(),inference_metadata=json.loads((output/'inference.json').read_text()))
    if model=='pca':
        import joblib
        b=joblib.load(bundle);prep=b['preprocessor'];x=a.layers['counts'].astype(np.float32);x=x.multiply((10000/np.asarray(x.sum(1)).ravel())[:,None]).tocsr();x.data=np.log1p(x.data)
        # Independent oracle uses original training var_names, bypassing query identifier matching.
        positions=a.var_names.get_indexer(prep['hvg_var_names']);assert (positions>=0).all();expected=prep['pca'].transform(prep['scaler'].transform(x[:,positions].toarray()));expected=b['classifier'].predict_proba(b['scaler'].transform(expected));difference=float(np.abs(expected-prob['probabilities']).max())
        assert record['inference_metadata']['coverage']['n_present_hvg']==2000 and difference<=1e-7
        record['direct_training_preprocessor_probability_max_abs_difference']=difference
    if feature:
        query=np.load(feature[-1]);dev=np.load(ASSETS/'features'/f'{model}_GSE149614_development_full.npz');x=query['X'];y=dev['X'][:64];cosine=np.sum(x*y,axis=1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1));record['representation_roundtrip_cosine_min']=float(cosine.min());record['representation_roundtrip_cosine_mean']=float(cosine.mean());record['note']='New query batching/hardware; descriptive numerical comparison, never selected using annotation scores'
    records.append(record)
write_json(REPORT/'annotation_interface_probe'/f'{args.probe_id}.json',dict(status='executed',project_commit=git_commit(),source='First64 actual development cells, stripped of lineage labels before query serialization',input_data_sha256=sha(path),external_data_accessed=False,records=records))
print('ANNOTATION_QUERY_INTERFACE_PASSED',flush=True)
