import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from common import DATA, ROOT, Resources, config, evaluate, provenance, save_features, write_json

c=config();r=Resources();a=ad.read_h5ad(DATA/'subsets'/f"{c['subset']}.h5ad")
train=a.obs.split.to_numpy()=='train'
X=a.layers['counts'].astype(np.float32)
library=np.asarray(X.sum(1)).ravel()
X=X.multiply((10000/np.maximum(library,1))[:,None]).tocsr();X.data=np.log1p(X.data)
# Train-only variance HVGs on log-normalized counts, explicit alternative to Seurat HVG.
mu=np.asarray(X[train].mean(0)).ravel();v=np.asarray(X[train].power(2).mean(0)).ravel()-mu**2
hvg=np.argsort(-v,kind='stable')[:min(2000,len(v))]
small=X[:,hvg].toarray()
sc=StandardScaler().fit(small[train]);small=sc.transform(small)
pca=PCA(n_components=min(50,train.sum()-1,small.shape[1]),random_state=42).fit(small[train])
features=pca.transform(small)
save_features(a,features,c['run_id'])
metrics=evaluate(features,a,c['run_id'])
write_json(ROOT/'results'/c['run_id']/'run.json',provenance(c,a,'PCA50+multinomial_logistic',r,metrics=metrics,hvg_method='top train log-normalized variance',n_hvg=len(hvg),normalization_target=10000,dtype='float32',batch_size=None,max_sequence_length=None,resource_measurement_scope='input preprocessing, PCA and validation/head evaluation; excludes interpreter imports and final provenance hashing'))
import pandas as pd
pd.DataFrame({'gene_symbol':a.var.gene_symbol.iloc[hvg].to_numpy()}).to_csv(ROOT/'results'/c['run_id']/'hvg.tsv',sep='\t',index=False)
