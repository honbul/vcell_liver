"""Measured intervention-mean metrics; zero-change correlations remain undefined."""
import numpy as np

def correlation(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-10 and np.std(y)>1e-10 else None

def score_response(targets,truth,control,pred,n_cells):
    rows=[]
    for i,target in enumerate(targets):
        delta=truth[i]-control[i];estimated=pred[i]-control[i];top=np.argsort(-np.abs(delta),kind='stable')[:20]
        rows.append(dict(target=str(target),n_cells=int(n_cells[i]),mse=float(np.mean((truth[i]-pred[i])**2)),mae=float(np.mean(np.abs(truth[i]-pred[i]))),
            pearson_expression=correlation(truth[i],pred[i]),pearson_delta=correlation(delta,estimated),observed_mean_delta_l2=float(np.linalg.norm(delta)),
            mse_observed_top20_delta=float(np.mean((delta[top]-estimated[top])**2)),sign_accuracy_observed_top20_delta=float(np.mean(np.sign(delta[top])==np.sign(estimated[top])))))
    return rows
