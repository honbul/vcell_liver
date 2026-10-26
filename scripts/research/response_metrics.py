"""Measured intervention-mean metrics; zero-change correlations remain undefined."""
import numpy as np
from sklearn.metrics import average_precision_score

def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None

def score_response(targets,truth,control,pred,n_cells,gene_names=None):
    rows=[]
    for i,target in enumerate(targets):
        delta=truth[i]-control[i];estimated=pred[i]-control[i];top=np.argsort(-np.abs(delta),kind='stable')[:20]
        relevant=np.zeros(len(delta),dtype=int);relevant[top]=1
        magnitude=np.abs(estimated);nonconstant=bool(np.ptp(magnitude)>1e-10)
        predicted_top=np.argsort(-magnitude,kind='stable')[:20]
        exclude=np.ones(len(delta),dtype=bool) if gene_names is None else ~np.isin(np.asarray(gene_names,dtype=str),[g for g in str(target).split('+') if g!='ctrl'])
        error=truth[i]-pred[i]
        rows.append(dict(target=str(target),n_cells=int(n_cells[i]),mse=float(np.mean(error**2)),delta_rmse=float(np.sqrt(np.mean(error**2))),delta_mae=float(np.mean(np.abs(error))),
            top20_response_retrieval_average_precision=float(average_precision_score(relevant,magnitude)),top20_response_retrieval_precision_at20=float(relevant[predicted_top].mean()) if nonconstant else None,
            ranking_scores_nonconstant=nonconstant,n_direct_targets_in_output=int((~exclude).sum()) if gene_names is not None else None,delta_rmse_excluding_direct_target=float(np.sqrt(np.mean(error[exclude]**2))) if gene_names is not None and exclude.any() else None,
            mae=float(np.mean(np.abs(truth[i]-pred[i]))),
            pearson_expression=correlation(truth[i],pred[i]),pearson_delta=correlation(delta,estimated),observed_mean_delta_l2=float(np.linalg.norm(delta)),
            mse_observed_top20_delta=float(np.mean((delta[top]-estimated[top])**2)),sign_accuracy_observed_top20_delta=float(np.mean(np.sign(delta[top])==np.sign(estimated[top])))))
    return rows
