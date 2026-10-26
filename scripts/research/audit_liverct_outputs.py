"""Verify all native/coarse labels and distinguish ontology coverage from uncertainty rejection."""
from research_core import *

tables=[];sources=[]
for fold in range(4):
    rid=f'S00_liverct_full_fold{fold}_seed42_v4';mapping=json.loads((REPORT/'runs'/rid/'mapping.json').read_text());native=pd.read_csv(ASSETS/'runs'/rid/'native_predictions.tsv',sep='\t',index_col=0)
    for method,labels,run_id in [('native_voting_lv1',native.pred_lv1.astype(str).to_numpy(),rid),('same_adapted_SCANVI',np.load(ASSETS/'runs'/rid/'scanvi_native_labels.npy').astype(str),rid+'_scanvi_control')]:
        run=json.loads((REPORT/'runs'/run_id/'run.json').read_text());coarse=pd.read_csv(run['prediction_path'],sep='\t');assert sha(run['prediction_path'])==run['prediction_sha256'] and np.array_equal(coarse.cell_id,native.index.astype(str))
        assert set(labels)<=set(mapping),'Do not silently treat an unmapped native label as confidence abstention'
        mapped=pd.Series(labels).map(mapping).to_numpy();assert np.array_equal(mapped,coarse.prediction.astype(str));counts=pd.Series(labels).value_counts()
        for label,count in counts.items():tables.append(dict(method=method,fold=fold,raw_label=label,mapped_label=mapping[label],n_cells=int(count)))
        unknown_labels=sorted({label for label in labels if mapping[label]=='Unknown'});assert set(unknown_labels)<={'Cholangiocyte'}
        sources.append(dict(run_id=run_id,prediction_sha256=run['prediction_sha256'],n_cells=len(coarse),n_outside_six_lineages=int((mapped=='Unknown').sum()),raw_unknown_causes=unknown_labels))
frame=pd.DataFrame(tables);out=REPORT/'liverct_output_audit';out.mkdir(exist_ok=True);frame.to_csv(out/'by_fold_raw_labels.tsv',sep='\t',index=False);frame.groupby(['method','raw_label','mapped_label']).n_cells.sum().to_csv(out/'raw_label_support.tsv',sep='\t')
assert all(frame.groupby('method').n_cells.sum()==63036)
write_json(out/'provenance.json',dict(status='executed',project_commit=git_commit(),source_sha256=sha(Path(__file__)),sources=sources,scored_native_column='pred_lv1, from exact saved soft VotingClassifier broad-lineage stage',native_head='RandomForestClassifier + MLPClassifier + OneVsRestClassifier + XGBClassifier, unweighted soft voting; exact types/hash in model_parameter_audit/liverct.json',confidence_threshold_applied_to_scored_column=False,coverage_interpretation='Unknown predictions in both scored pipelines arise from Cholangiocyte outside the fixed six-lineage ontology; not a confidence rejection or demonstrated OOD detector',auxiliary_outputs='Native suggest_label_lv1 uses a threshold and disease/transition scores use auxiliary models; those outputs were not used in six-lineage scoring'))
print('LIVERCT_OUTPUT_MAPPING_AUDITED',flush=True)
