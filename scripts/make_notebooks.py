"""Generate Korean tutorials. Execution outputs are populated only by nbclient."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',execution_count=None,metadata={},outputs=[],source=s.splitlines(True))
setup='''from pathlib import Path
import os, sys, json, subprocess
ROOT = Path.cwd()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
assert (ROOT / 'configs/paths.yaml').exists()
os.chdir(ROOT)
os.environ.update(OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import yaml, pandas as pd, numpy as np
PATHS = yaml.safe_load((ROOT / 'configs/paths.yaml').read_text())
DATA = Path(PATHS['data_root'])
def run_script(name, config=None, model='scgpt', args=()):
    cmd = [str(ROOT / f'.venvs/vcell-{model}/bin/python'), '-u', str(ROOT / 'scripts' / name)]
    if config: cmd += ['--config', str(ROOT / 'configs' / config)]
    cmd += list(args)
    print(' '.join(cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(p.stdout[-16000:])
    if p.returncode:
        print(p.stderr[-16000:])
        p.check_returncode()
    return p
print('code:', ROOT, '\\ndata:', DATA)
'''
notebooks=[
('00_environment_and_model_contracts',[
md('# 00 환경과 모델 입력 계약\n\n무엇을 하는가: 서버 경로, GPU, 공식 checkpoint를 확인한다. 설치 성공과 HCC 적용 실험은 별도 상태다. 이 실습은 이미 내려받은 작은 입력을 사용하며 다운로드를 시작하지 않는다.'),
code(setup),code("run_script('00_inspect_environment.py', args=['--out', 'reports/environment_notebook.json'])\nprint(json.loads((ROOT / 'reports/environment_notebook.json').read_text())['torch_cuda'])"),
code("print(json.loads((DATA / 'checkpoints/scgpt-whole-human/args.json').read_text()))\nprint(json.loads((ROOT / 'third_party/Geneformer/Geneformer-V2-104M/config.json').read_text()))\nprint(pd.read_csv(ROOT / 'reports/patient_label_counts.tsv', sep='\\t').to_string(index=False))"),
md('생물학적 해석: scGPT는 공식 vocab의 발현 rank bins, Geneformer V2는 전체 raw count·Ensembl·n_counts로 median-normalized rank encoding을 만든다. PCA는 train 환자에서만 학습한다. frozen embedding + supervised head이며 분류 시스템 전체를 zero-shot이라 부르지 않는다.\n\n실패 해결: 경로는 configs/paths.yaml과 envs/activate.sh 확인. CUDA가 없으면 scGPT는 CPU로 동작하고 Geneformer도 느린 CPU fallback이 있다. 다른 GPU 작업을 종료하지 않는다. GEARS의 실측 개입 예측은 별도 task다.')]),
('01_hcc_data_and_patient_split',[
md('# 01 HCC raw count와 환자 분할\n\n무엇을 하는가: 준비된 64세포 subset의 count·annotation·환자 분할을 직접 확인한다. 원본 전체를 다시 다운로드하지 않는다.'),code(setup),
code("import anndata as ad\na = ad.read_h5ad(DATA / 'subsets/smoke.h5ad')\nprint(a)\nprint(a.obs[['patient_id','sample_id','tissue_site','cell_type_original','split','n_counts','n_genes','pct_counts_mt']].head().to_string())\nprint(pd.crosstab(a.obs.patient_id, a.obs.split))"),
code("from scipy import sparse\nassert sparse.issparse(a.layers['counts'])\nassert (a.layers['counts'].data >= 0).all()\nassert np.equal(a.layers['counts'].data, np.floor(a.layers['counts'].data)).all()\nassert np.array_equal(np.asarray(a.layers['counts'].sum(1)).ravel(), a.obs.n_counts)\nfor p, rows in a.obs.groupby('patient_id', observed=True): assert rows.split.nunique() == 1\nprint('count/환자 누수 검사 통과')\nprint(a.var.id_mapping_status.value_counts())\nprint(json.loads((ROOT / 'reports/data_audit.json').read_text()))"),
md('생물학적 해석: 원본 celltype의 T/NK, B, Myeloid, Endothelial, Fibroblast, Hepatocyte를 유지한다. Tumor 조직 출처를 malignant 세포 정답으로 바꾸지 않는다. Normal은 인접 조직이다.\n\n실패 해결: 파일이 없으면 scripts/download_hcc.sh 후 scripts/01_prepare_hcc.py --config configs/hcc_gse149614.yaml을 실행한다. 이 작업은 전체 데이터를 다루며 작은 실습보다 오래 걸린다. sparse count를 전체 dense로 만들지 않는다. 모호한 symbol은 임의 Ensembl ID로 채우지 않는다.')]),
('02_baseline_and_scgpt',[
md('# 02 PCA 기준선과 scGPT 실제 추론\n\n무엇을 하는가: 동일 64세포에서 train-only PCA와 공식 whole-human frozen embedding을 실행하고 환자 holdout 결과를 읽는다. 64세포는 동작 확인용이며 성능 주장은 2000/10000세포 결과로 판단한다.'),code(setup),
code("print((DATA / 'subsets/smoke.h5ad').stat().st_size)\nrun_script('02_baseline_annotation.py', 'pca_smoke.yaml')\nrun_script('03_scgpt_annotation.py', 'scgpt_smoke.yaml')"),
code("for rid in ['pca_smoke', 'scgpt_smoke']:\n    r = json.loads((ROOT / 'results' / rid / 'run.json').read_text())\n    print(rid, 'macro-F1=', r['metrics']['macro_f1'], 'balanced accuracy=', r['metrics']['balanced_accuracy'], 'peak allocated GiB=', r.get('peak_cuda_allocated_gib'), 'wall seconds=', r['wall_seconds'])\n    print(pd.read_csv(ROOT / 'results' / rid / 'by_patient.tsv', sep='\\t').to_string(index=False))"),
md('생물학적 해석: 모델이 PCA를 이길 필요는 없다. 지도 분류 head는 train label로 학습하고 C/weight는 validation에서 고른다. test 환자는 모델 선택에 쓰지 않는다. scGPT는 내부 collator가 raw count를 binning하므로 사전에 binning하지 않는다.\n\n실패 해결: OOM이면 config batch_size=1. sequence 변경은 별도 run ID로 기록한다. flash-attention은 필수가 아니며 이 환경은 공식 non-flash 지원 경로다. cache/출력은 데이터 루트에 저장된다.')]),
('03_geneformer_comparison',[
md('# 03 Geneformer-V2-104M 비교\n\n무엇을 하는가: scGPT와 같은 세포·환자 split에서 V2 공식 TranscriptomeTokenizer와 frozen CLS embedding을 실행한다. 316M root checkpoint나 V1 tokenizer를 섞지 않는다.'),code(setup),
code("assert (ROOT / 'third_party/Geneformer/Geneformer-V2-104M/model.safetensors').exists()\nrun_script('04_geneformer_annotation.py', 'geneformer_smoke.yaml', model='geneformer')"),
code("audit = json.loads((DATA / 'outputs/geneformer_smoke/tokenization_audit.json').read_text())\nprint(json.dumps(audit, indent=2))\nr = json.loads((ROOT / 'results/geneformer_smoke/run.json').read_text())\nprint('actual parameters=', r['actual_parameter_count'])\nprint('macro-F1=',r['metrics']['macro_f1'], 'peak allocated GiB=',r.get('peak_cuda_allocated_gib'))\nprint(pd.read_csv(ROOT / 'results/geneformer_smoke/per_class.tsv', sep='\\t').to_string(index=False))"),
md('생물학적 해석: raw all-gene count에서 mapping 가능한 유전자를 공식 V2 rank encoding으로 변환한다. n_counts는 gene mapping 이전 library size를 유지한다. 4096 길이에 CLS/EOS가 포함된다. 공식 기본 emb_layer=-1과 같은 hidden_states[11] CLS를 사용한다. 입력 범위가 다른 실용 pipeline 비교다. 사전학습 corpus 중복은 unknown이다.\n\n실패 해결: scGPT의 HVG/binned 데이터를 재사용하지 않는다. Geneformer 전용 환경은 envs/activate.sh geneformer. OOM이면 batch1부터 실행하고 sequence 변경을 별도 기록한다.')]),
('04_knowledge_graph_features',[
md('# 04 KG 경로 특징 추가 효과\n\n무엇을 하는가: 외부 Reactome membership으로 계산한 proxy가 실제 분류 예측에 영향을 주는지 비교한다. 앞 노트북의 PCA/scGPT smoke 결과를 입력으로 사용한다.'),code(setup),
code("assert (DATA / 'annotation/Ensembl2Reactome_All_Levels.human.tsv').exists()\nrun_script('05_kg_features_ablation.py', 'kg_smoke.yaml')"),
code("print(pd.read_csv(ROOT / 'reports/kg_ablation_smoke.tsv', sep='\\t').to_string(index=False))\nselection = json.loads((ROOT / 'results/kg_smoke/selection.json').read_text())\nprint('selected pathways=', selection['selected_n'])\nprint('random sizes/degrees preserved=',selection['size_preserved'],selection['gene_membership_degree_preserved'])"),
md('생물학적 해석: 경로 membership은 세포에서의 활성화·인과관계가 아니다. mean train-zscore proxy를 예측 입력으로 concatenate한다. pathway-only, embedding+pathway, embedding+무작위 대조를 동일 split/head로 비교한다. 무작위 대조는 경로 크기와 유전자 참여 횟수를 보존한다. 단순 결과 설명을 붙인 것과 예측 입력으로 사용한 것을 구분한다.\n\n실패 해결: feature 파일이 없으면 02를 먼저 실행한다. 경로 수와 규제 강도는 validation만 사용한다. 성능 향상이 없으면 실제 결과 그대로 기록한다.')]),
('05_virtual_perturbation_interpretation',[
md('# 05 가상 유전자 삭제의 해석\n\n**실측 섭동 검증 전 탐색.** 무엇을 하는가: train hepatocyte 최대8세포에서 ALB/HNF4A를 각각 독립 삭제하고 no-op 및 발현량 유사 random gene 대조와 비교한다. 치료 효과나 발현 예측 실습이 아니다.'),code(setup),
code("run_script('06_geneformer_virtual_perturb.py', 'isp_smoke.yaml', model='geneformer')"),
code("table = pd.read_csv(ROOT / 'reports/virtual_perturbation_embedding_shift_smoke.tsv', sep='\\t')\nprint(table.groupby(['gene','condition','token_present']).cosine_shift.agg(['count','mean','median','max']).to_string())\nassert (table.loc[table.condition == 'no_op','cosine_shift'] < 1e-5).all()\nprint('no-op 확인 통과; token_present=False는 실제 삭제된 세포가 아님')"),
md('생물학적 해석: 출력은 rank token 삭제 전후 CLS embedding의 1−cosine이다. 큰 변화가 강한 생물학적 영향·약효·생존율을 뜻하지 않는다. post-perturbation expression이나 logFC로 바꿔 해석하지 않는다. 신뢰할 목표 상태가 없어 변화의 크기만 본다. 분류 confidence를 정상화로 읽지 않는다.\n\n실패 해결: 실제 발현 및 vocabulary를 확인해야 한다. 여러 genes를 함께 넣는 공동 삭제 대신 한 gene씩 처리한다. token이 없는 세포는 별도 표시한다. 실제 KO/CRISPR 반응 예측에는 대조군·실측 개입 데이터와 별도 검증이 필요하다.')])]
(ROOT/'notebooks').mkdir(exist_ok=True)
for name,cells in notebooks:
    # Tutorial runs use distinct IDs and preserve the initial measured comparisons.
    for cell in cells:
        source=''.join(cell['source'])
        for model in ('pca','scgpt','geneformer','kg','isp'):
            source=source.replace(f'{model}_smoke',f'tutorial_{model}_smoke')
        cell['source']=source.splitlines(True)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='vcell-scgpt',language='python',name='vcell-scgpt'),language_info=dict(name='python')),cells=cells)
    for i,c in enumerate(nb['cells']):c['id']=f'cell-{i:03d}'
    (ROOT/'notebooks'/f'{name}.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Generated',len(notebooks),'unexecuted notebooks')
