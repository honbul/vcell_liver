"""Train-fitted UMAP projections for labels and donors; visualization only."""
import argparse
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import umap
from common import DATA,ROOT,load_features,write_json

p=argparse.ArgumentParser();p.add_argument('--subset',default='pilot');args=p.parse_args()
a=ad.read_h5ad(DATA/'subsets'/f'{args.subset}.h5ad');tr=a.obs.split.to_numpy()=='train'
out=ROOT/'reports/figures';out.mkdir(parents=True,exist_ok=True)
for model in ('pca','scgpt','geneformer'):
    rid=f'{model}_{args.subset}';x=load_features(a,rid)
    reducer=umap.UMAP(random_state=42,transform_seed=42,n_jobs=1)
    z=np.empty((len(x),2),dtype=np.float32);z[tr]=reducer.fit_transform(x[tr]);z[~tr]=reducer.transform(x[~tr])
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,col,title in zip(axes,['cell_type_coarse','patient_id'],['Cell type','Patient']):
        for i,label in enumerate(sorted(a.obs[col].unique())):
            ix=a.obs[col].to_numpy()==label
            ax.scatter(z[ix,0],z[ix,1],s=4,alpha=.65,label=label,color=plt.get_cmap('tab10')(i))
        ax.set_title(f'{rid}: {title}');ax.legend(markerscale=3,fontsize=7);ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2')
    fig.suptitle('UMAP fitted on train donors; visualization only, not evidence of superiority')
    fig.tight_layout();fig.savefig(out/f'{rid}_umap.png',dpi=160);plt.close(fig)
    write_json(out/f'{rid}_umap.json',dict(train_only_fit=True,seed=42,cell_ids_source=f'manifests/splits/{args.subset}.tsv',note='Visualization only; no label-driven tuning'))
    print(rid,'plotted',flush=True)
