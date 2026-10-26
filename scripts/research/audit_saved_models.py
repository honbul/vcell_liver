"""Read-only numerical parameter supplements for older immutable scVI/LiverCT runs."""
import argparse
import anndata as ad
import joblib,torch
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--model',choices=['scvi','liverct'],required=True);args=p.parse_args();torch.set_num_threads(8)
a=ad.read_h5ad(ASSETS/'data/development_full.h5ad');records=[]
if args.model=='scvi':
    import scvi
    for fold in range(4):
        rid=f'A04_full_fold{fold}_seed42_v2';root=ASSETS/'runs'/rid;info=json.loads((root/'outer_model_training.json').read_text());b=a[info['training_cell_ids'][:64],info['genes']].copy();b.X=b.layers['counts'].copy()
        model=scvi.model.SCVI.load(str(root/'outer_model'),adata=b,accelerator='cpu');head=joblib.load(root/'head.joblib')['classifier'];parameters=sum(p.numel() for p in model.module.parameters() if p.requires_grad);assert isinstance(parameters,int) and parameters>0
        records.append(dict(run_id=rid,source_run_sha256=sha(REPORT/'runs'/rid/'run.json'),model_checkpoint_sha256=sha(root/'outer_model/model.pt'),total_model_parameters=sum(p.numel() for p in model.module.parameters()),trainable_backbone_parameters=parameters,trainable_head_parameters=int(head.coef_.size+head.intercept_.size),trainable_parameters=parameters+int(head.coef_.size+head.intercept_.size),method='Loaded actual saved scVI module on64 original training cells with exact saved genes; all module parameters trained from scratch',upstream_version=scvi.__version__))
else:
    import scipy.sparse as sp
    sys.path.insert(0,str(ROOT/'third_party/LiverCT'));from test_function import feature_alignment
    import scarches as sca
    for fold in range(4):
        rid=f'S00_liverct_full_fold{fold}_seed42_v4';run=json.loads((REPORT/'runs'/rid/'run.json').read_text());root=ASSETS/'runs'/rid
        split=pd.read_csv(ROOT/'manifests/research_phase2/development_donor4_v2.tsv',sep='\t');donors=split[(split.fold==fold)&(split.role=='evaluation')].patient_id;sample=a[a.obs.patient_id.isin(donors)][:64].copy();symbols=sample.var.gene_symbol.astype(str).to_numpy();unique=pd.Index(pd.unique(symbols));indices=unique.get_indexer(symbols);mapping=sp.csr_matrix((np.ones(len(indices)),(np.arange(len(indices)),indices)),shape=(len(indices),len(unique)))
        q=ad.AnnData(sp.csr_matrix(sample.layers['counts'])@mapping,obs=pd.DataFrame(index=sample.obs_names),var=pd.DataFrame(index=unique));reference=root/'reference_copy/scarches_model_new';genes=pd.read_csv(reference/'var_names.csv',header=None).iloc[:,0];q=feature_alignment(q,genes).copy();q.obs['batch']='new_batch';q.obs['level1']='Unknown';q.layers['counts']=q.X.copy()
        # Reconstruct the exact query-loading freeze contract without training or reference writes.
        model=sca.models.SCANVI.load_query_data(q,str(reference),freeze_dropout=True,use_gpu=False);trainable={name:p.numel() for name,p in model.module.named_parameters() if p.requires_grad}
        saved=ASSETS/'runs'/(run.get('adaptation_source_run') or rid)/'my_surgery_model/model_params.pt';checkpoint=torch.load(saved,map_location='cpu');state=checkpoint.get('model_state_dict',checkpoint);actual=model.module.state_dict();assert set(state)==set(actual) and all(state[k].shape==actual[k].shape for k in state)
        records.append(dict(run_id=rid,paired_control_run_id=rid+'_scanvi_control',source_run_sha256=sha(REPORT/'runs'/rid/'run.json'),model_checkpoint_sha256=sha(saved),total_model_parameters=sum(p.numel() for p in model.module.parameters()),requires_grad_parameters_under_reconstructed_query_contract=sum(trainable.values()),requires_grad_tensors=trainable,native_svm_trainable_parameters_during_query_mapping=0,shared_adaptation_cost=True,
            method='CPU reconstruction of original load_query_data freeze contract on64 actual unlabeled evaluation cells; saved adapted checkpoint keys/shapes verified; no model fit or new scientific scoring',limitation='Gradient hooks can mask columns of trainable tensors. requires_grad count is an upper bound on updated scalar entries, not an exact hook-adjusted count; native reference SVM remains fixed'))
out=REPORT/'model_parameter_audit';out.mkdir(exist_ok=True);write_json(out/f'{args.model}.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),immutable_original_runs_changed=False,records=records));print('SAVED_MODEL_PARAMETER_AUDIT',args.model,flush=True)
