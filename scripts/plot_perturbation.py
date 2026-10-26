"""Plot only measured embedding shifts, separating token-absent cells."""
import argparse,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from common import ROOT

def plot_shifts(table,subset):
    present=table.token_present.astype(str).str.lower()=='true'
    groups=list(table.loc[present].groupby(['gene','condition'],sort=False))
    fig,ax=plt.subplots(figsize=(10,5))
    values=[g.cosine_shift.to_numpy() for _,g in groups]
    labels=[]
    for (gene,condition),g in groups:
        kind='no-op' if condition=='no_op' else 'independent deletion' if condition=='independent_delete' else 'matched to '+str(g.matched_target.iloc[0])
        labels.append(f'{gene}\n{kind}\nn={len(g)}')
    boxes=ax.boxplot(values,tick_labels=labels,patch_artist=True,showfliers=True)
    colors={'no_op':'#999999','independent_delete':'#377eb8','expression_matched_random_delete':'#ff9900'}
    for box,((gene,condition),g) in zip(boxes['boxes'],groups):box.set_facecolor(colors[condition])
    try:
        path=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True).strip()
        font=FontProperties(fname=path)
        ax.set_title('실측 섭동 검증 전 탐색',fontproperties=font,fontsize=15)
        fig.text(.5,.01,'token_present=True만 표시 · embedding 변화 크기이며 발현량/약효 예측이 아님',ha='center',fontproperties=font,fontsize=10)
    except (OSError,subprocess.SubprocessError):
        ax.set_title('Exploratory embedding shifts: not validated against measured perturbations')
    ax.set_ylabel('CLS cosine distance (1 - cosine)')
    ax.tick_params(axis='x',labelsize=7)
    fig.tight_layout(rect=[0,.05,1,1])
    out=ROOT/'reports/figures';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'virtual_perturbation_{subset}.png',dpi=170);plt.close(fig)
    print('Saved exploratory embedding-shift figure',subset,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--subset',default='pilot');a=p.parse_args()
    plot_shifts(pd.read_csv(ROOT/'reports'/f'virtual_perturbation_embedding_shift_{a.subset}.tsv',sep='\t'),a.subset)
