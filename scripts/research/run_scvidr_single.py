"""Conditional mouse-liver TCDD cell-type holdout; native scVIDR, same-VAE arithmetic and expression controls."""
import argparse,sys,time,traceback
import anndata as ad
import scipy.sparse as sp
import scanpy as sc
import scvi,torch
from sklearn.metrics import r2_score
from research_core import *

p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--epochs',type=int,default=100);args=p.parse_args()
root=ROOT/'third_party/scVIDR';sys.path.insert(0,str(root/'vidr'));from vidr import VIDR
out=REPORT/'runs'/args.run_id;assert not (out/'run.json').exists();out.mkdir(parents=True,exist_ok=True);raw=ASSETS/'runs'/args.run_id;raw.mkdir(parents=True,exist_ok=True)
launch=git_commit();source_hash=sha(Path(__file__));torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);scvi.settings.seed=42;r=Resources();started=time.monotonic()
source=ASSETS/'official/scVIDR/nault2021_singleDose.h5ad';target='Hepatocytes - portal'
try:
    assert scvi.__version__=='0.13.0', 'Use the child environment with native scvi0.13 API; scvi0.14 makes VIDR abstract'
    a=ad.read_h5ad(source);assert a.shape==(30927,22213) and a.var_names.is_unique
    allowed=[line.strip() for line in (root/'metadata/liver_celltypes').read_text().splitlines() if line.strip()];a=a[a.obs.celltype.isin(allowed)].copy();assert set(a.obs.Dose.astype(int))=={0,30}
    a.X=sp.csr_matrix(a.X,dtype=np.float32);assert np.all(a.X.data>=0) and np.all(a.X.data==np.round(a.X.data))
    a.obs['dose']=a.obs.Dose.astype(int).astype(str);a.obs['original_cell_id']=a.obs_names.astype(str)
    sc.pp.normalize_total(a,target_sum=1e4);sc.pp.log1p(a)
    test_mask=((a.obs.celltype==target)&(a.obs.dose=='30')).to_numpy();assert test_mask.sum()>20
    train=a[~test_mask].copy();assert not ((train.obs.celltype==target)&(train.obs.dose=='30')).any()
    # The released raw file has no HVGs; select on the allowed training population only.
    sc.pp.highly_variable_genes(train,n_top_genes=5000,flavor='seurat');genes=train.var_names[train.var.highly_variable];assert len(genes)==5000
    train=train[:,genes].copy();observed=a[test_mask,genes].copy();control=train[(train.obs.celltype==target)&(train.obs.dose=='0')].copy();assert control.n_obs>20
    train.obs['celltype']=train.obs.celltype.astype('category');train.obs['dose']=train.obs.dose.astype('category');scvi.data.setup_anndata(train,batch_key='dose',labels_key='celltype')
    train.obs[['original_cell_id','celltype','dose']].to_csv(raw/'training_manifest.tsv',sep='\t',index=False);observed.obs[['original_cell_id','celltype','dose']].to_csv(raw/'heldout_manifest.tsv',sep='\t',index=False)
    write_json(out/'data_contract.json',dict(status='prepared',source_sha256=sha(source),species='mouse',cell_context='liver single nuclei; six author-selected nonimmune cell types',compound='2,3,7,8-tetrachlorodibenzo-p-dioxin (TCDD)',dose_units='microgram_per_kg',control_dose=0,treated_dose=30,
        exposure='Oral gavage every4days for28days, from Nault study Methods; not an acute single administration',source_study='GSE148339 (Nault single-dose source); author supplied nault2021_singleDose file pinned separately',
        replicate='Publication reports biological replicates, but per-mouse identity is not retained as an explicit field in the supplied h5ad; batch0/1 equals treatment, not replicate',
        source_links=['https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE148339','https://pmc.ncbi.nlm.nih.gov/articles/PMC7674514/','https://github.com/BhattacharyaLab/scVIDR'],
        holdout='All treated Hepatocytes - portal excluded from VAE training and HVG fit; control cells of this target context are supplied',n_training_cells=train.n_obs,n_heldout_treated_cells=observed.n_obs,n_target_control_cells=control.n_obs,n_genes=5000,training_manifest_sha256=sha(raw/'training_manifest.tsv'),heldout_manifest_sha256=sha(raw/'heldout_manifest.tsv'),
        information='Same TCDD exposure observed in other cell types; this is context transfer, not an unseen drug or animal-replicate holdout'))
    model=VIDR(train,linear_decoder=False);initial=next(model.module.parameters()).detach().cpu().clone();preprocessing=time.monotonic()-started;start=time.monotonic()
    model.train(max_epochs=args.epochs,batch_size=128,early_stopping=True,early_stopping_patience=25);training=time.monotonic()-start
    change=float((next(model.module.parameters()).detach().cpu()-initial).abs().max());assert change>0
    model.save(str(raw/'model'),overwrite=False);start=time.monotonic();predictions={};input_control_ids=None
    for regression,name in [(False,'native_same_VAE_arithmetic'),(True,'scVIDR_regression')]:
        np.random.seed(42);torch.manual_seed(42);result=model.predict(ctrl_key='0',treat_key='30',cell_type_to_predict=target,regression=regression);pred=result[0];assert np.isfinite(pred.X).all()
        ids=pred.obs.original_cell_id.astype(str).tolist()
        if input_control_ids is None:input_control_ids=ids
        else:assert ids==input_control_ids,'Native delta estimators must use the same sampled target controls'
        predictions[name]=np.asarray(pred.X).mean(0)
    lookup={str(cell):i for i,cell in enumerate(train.obs_names)};sampled=np.array([lookup[cell] for cell in input_control_ids]);sampled_control=np.asarray(train.X[sampled].mean(0)).ravel();full_control=np.asarray(control.X.mean(0)).ravel();truth=np.asarray(observed.X.mean(0)).ravel()
    predictions['no_change_all_target_controls']=full_control;predictions['no_change_same_native_control_sample']=sampled_control
    deltas=[]
    for celltype in allowed:
        if celltype==target:continue
        ctrl=train[(train.obs.celltype==celltype)&(train.obs.dose=='0')];treated=train[(train.obs.celltype==celltype)&(train.obs.dose=='30')]
        if ctrl.n_obs and treated.n_obs:deltas.append(np.asarray(treated.X.mean(0)).ravel()-np.asarray(ctrl.X.mean(0)).ravel())
    assert len(deltas)>=3;predictions['training_celltype_mean_response']=full_control+np.mean(deltas,axis=0)
    # Native prediction balances/resamples controls. Preserve duplicate-row counts rather than treating them as new biological units.
    rows=[];true_delta=truth-full_control
    def corr(x,y):return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None
    for name,pred in predictions.items():
        error=pred-truth;delta=pred-full_control;rows.append(dict(model_id=name,delta_rmse=float(np.sqrt(np.mean(error**2))),delta_mae=float(np.mean(np.abs(error))),pearson_delta=corr(true_delta,delta),pearson_expression=corr(truth,pred),expression_r2=float(r2_score(truth,pred))))
    np.savez_compressed(raw/'predictions_mean.npz',gene_names=genes.to_numpy(dtype=str),truth=truth,control=full_control,**predictions)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics.tsv',sep='\t',index=False);inference=time.monotonic()-start
    history=getattr(model,'history_',{});history_lengths={k:len(v) for k,v in history.items() if hasattr(v,'__len__')} if isinstance(history,dict) else {}
    res=r.finish();write_json(out/'run.json',dict(run_id=args.run_id,status='executed',task='chemical_response',regime='modified_official_celltype_holdout',model_id='scVIDR_native_and_matched_controls',project_commit=launch,source_sha256=source_hash,upstream_commit='bcb6a96c85cddc4eef052be739c122efb737fe0e',dataset_version='official_nault2021_singleDose',split_id='portal_treated_holdout_controls_supplied',seed=42,
        dose_units='microgram_per_kg',dose=30,time_days=28,training_celltypes=allowed,evaluation_celltype=target,n_training_cells=train.n_obs,n_evaluation_cells=observed.n_obs,n_native_prediction_rows=len(input_control_ids),n_unique_native_input_controls=len(set(input_control_ids)),n_genes=5000,max_epochs=args.epochs,history_lengths=history_lengths,
        trainable_parameters=sum(p.numel() for p in model.module.parameters() if p.requires_grad),actual_parameter_max_abs_update=change,data_sha256=sha(source),prediction_path=str(raw/'predictions_mean.npz'),prediction_sha256=sha(raw/'predictions_mean.npz'),wall_preprocessing_seconds=preprocessing,wall_training_seconds=training,wall_inference_seconds=inference,wall_total_seconds=res.pop('wall_seconds'),
        environment=dict(scvi_tools=scvi.__version__,torch=torch.__version__),modifications=['Train-only5000 HVGs instead of any full-data feature selection','Native100epoch/batch128/patience25 settings retained','Native scvi0.13 API isolated in child environment; torch1.13.1 instead of original1.8','One prespecified portal context rather than all manuscript cell types','Compare means; do not use cell-resampling intervals as biological reproducibility'],
        failure_or_limitations=['Mouse toxicant response, not adult HCC drug efficacy','Same compound/exposure transferred across cell types; not unseen-drug or unseen-dose generalization','Per-mouse assignments unavailable; no biological-replicate confidence interval claimed','Native control balancing may duplicate nuclei; sampled-row count is not independent sample size'],**res));print('SCVIDR_SINGLE_COMPLETE',frame.to_string(index=False),flush=True)
except Exception as exc:
    write_json(out/'failure.json',dict(status='failed',run_id=args.run_id,source_sha256=source_hash,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),**r.finish()));raise
