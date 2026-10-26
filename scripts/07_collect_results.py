"""Build the summary exclusively from real run JSON and resource measurements."""
import argparse,json
from pathlib import Path
import pandas as pd
from common import ROOT

p=argparse.ArgumentParser();p.add_argument('--runs',default='results');p.add_argument('--out',default='reports/final_summary.md');args=p.parse_args()
rows=[]
for path in sorted(Path(args.runs).glob('*/run.json')):
    r=json.loads(path.read_text())
    if r.get('task')!='annotation' or 'metrics' not in r:continue
    m=r['metrics']
    stage_path=ROOT/'reports/stages'/f"{r['run_id']}.json"
    stage=json.loads(stage_path.read_text()) if stage_path.exists() else {}
    rows.append(dict(run_id=r['run_id'],status=r['status'],n_cells=r['n_cells'],macro_f1=m['macro_f1'],balanced_accuracy=m['balanced_accuracy'],n_test=m['n_evaluated'],n_unknown=m['n_test_unknown'],
       peak_allocated_gib=r.get('peak_cuda_allocated_gib'),peak_reserved_gib=r.get('peak_cuda_reserved_gib'),peak_process_gpu_gib=r.get('peak_process_gpu_gib'),
       peak_cpu_rss_gib=r.get('peak_cpu_rss_gib'),wall_seconds=r.get('wall_seconds'),cli_wall_seconds=stage.get('wall_seconds'),shuffle_macro_f1=m['shuffled_train_labels_test_macro_f1']))
df=pd.DataFrame(rows);df.to_csv(ROOT/'reports/model_comparison_measured.tsv',sep='\t',index=False)
def table(frame):
    cols=frame.columns.tolist();lines=['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
text='''# 서버 실제 실행 결과\n\n이 보고서는 results/*/run.json의 실제 실행 결과로 생성했다. 문헌 재현 benchmark가 아니라 GSE149614의 신규 HCC 적용이다. frozen cell embedding + supervised logistic head이며 전체 시스템을 zero-shot classifier로 부르지 않는다.\n\n## 실제 성능과 자원\n\n'''
if len(df):text+=table(df)+'\n\n'
for subset in ('pilot','comparison'):
    path=ROOT/'reports'/f'kg_ablation_{subset}.tsv'
    if path.exists():
        kg=pd.read_csv(path,sep='\t')
        text+=f'## Reactome 추가 실험 ({subset})\n\n'+table(kg)+'\n\n'
        base=kg[kg.run_id==f'scgpt_{subset}'].iloc[0]
        combined=kg[kg.run_id==f'kg_{subset}_foundation_pathway'].iloc[0]
        random=kg[kg.run_id==f'kg_{subset}_foundation_random'].iloc[0]
        text+=f"scGPT 대비 경로 결합 macro-F1 차이 {combined.macro_f1-base.macro_f1:+.4f}; 크기/유전자 참여 분포를 보존한 random 대조 대비 {combined.macro_f1-random.macro_f1:+.4f}. Test prediction 변경 수: {int(combined.n_changed_test_predictions_vs_foundation)}.\n\n"
text+='''## 해석과 한계\n\nMacro-F1와 balanced accuracy는 고정된 두 test 환자에서 측정했다. 환자별/클래스별 지표와 confusion matrix는 각 results/<run_id>에 있다. 환자별 macro-F1도 전체 train class 목록을 기준으로 하므로 해당 환자에 없는 class는 0을 포함한다. test 환자가 두 명뿐이어서 일반화와 신뢰구간 해석이 제한적이며 세포 bootstrap CI를 제시하지 않는다. 64세포 smoke는 동작 확인용이다.\n\nPCA HVG(로그 정규화 분산 상위2000), scaler, PCA, classifier는 train만 학습했다. C/weight와 경로 수는 validation으로 골랐다. 모든 모델은 동일 split/cell IDs/class schema다. 모델별 공식 입력 범위가 다르므로 아키텍처만의 비교가 아니다. 원 cohort와 사전학습 데이터의 중복은 unknown이며 pretraining-unseen 성능을 주장하지 않는다. shuffled-train-label 결과는 label feature 혼입에 대한 간단한 sanity check이며 별도 유효성 증명은 아니다.\n\nReactome membership은 외부 prior, gene expression은 환자 관측이다. 경로 mean-zscore는 활성화/인과관계의 측정값이 아니다. 실제 입력에 concatenate한 모델을 random 대조와 비교했으며 이득을 보장하지 않는다.\n\n가상 삭제는 실측 섭동 검증 전 탐색. 독립 gene token 삭제 전후 CLS cosine shift이며 post-perturbation expression/logFC/약효/생존율이 아니다. 신뢰할 목표 상태가 없어 변화 크기만 보고한다. 결과는 virtual_perturbation_*와 한계 문서 참조.\n\nGEARS, scFoundation 설치/학습, 미세조정은 이번 필수 범위에서 실행하지 않았다. GEARS는 실제 genetic intervention 데이터가 필요한 조건부 후속이다. 공식 논문 예제의 정량적 성과나 생물학적 검증을 재현했다고 주장하지 않는다.\n\n## 재실행\n\n```bash\ncd /mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver\nsource envs/activate.sh scgpt\npython scripts/02_baseline_annotation.py --config configs/pca_pilot.yaml\npython scripts/03_scgpt_annotation.py --config configs/scgpt_pilot.yaml\nsource envs/activate.sh geneformer\npython scripts/04_geneformer_annotation.py --config configs/geneformer_pilot.yaml\nsource envs/activate.sh scgpt\npython scripts/05_kg_features_ablation.py --config configs/kg_pilot.yaml\nsource envs/activate.sh geneformer\npython scripts/06_geneformer_virtual_perturb.py --config configs/isp_pilot.yaml\nsource envs/activate.sh scgpt\npython scripts/08_execute_notebooks.py\npython scripts/07_collect_results.py --runs results --out reports/final_summary.md\n```\n\n## 다음 한 가지 실험\n\n두 test 환자만의 변동을 확인하기 위해 현재 고정 protocol을 유지한 leave-one-patient-out 평가를 별도 실험으로 수행한다. 현재 test 결과로 모델 설정을 바꾸지 않는다.\n'''
text+='\n자원 측정 범위: model run의 wall_seconds는 checkpoint 로딩/입력 처리/embedding 구간이며 head CPU 실행은 제외한다. PCA는 전처리·분류기까지 포함한다. CLI 전체 시간은 cli_wall_seconds와 reports/stages에 별도로 있다. KG 세 head의 자원 값은 공통 feature 준비와 세 head 전체 실행 구간을 공유한다. 다른 GPU 작업과 공유한 서버에서의 관측이며 속도 우열 benchmark가 아니다. CUDA allocated와 process GPU memory를 구분한다.\n'
out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text)
print(text)
