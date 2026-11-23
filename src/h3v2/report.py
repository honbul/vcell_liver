"""Verify the completed H3-v2 experiment and assemble versionable aggregate artifacts."""
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
from subprocess import run

import numpy as np
import pandas as pd

from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256

from .data import ROOT,SPLITS,STUDY,V1
from .grid import FAMILIES,SEEDS
from .quality import AXES
from .score import preflight

OUTPUT=ROOT/"reports-final"
SCORED=ROOT/"score"
DISPLAY=("spearman","pearson","interaction_spearman","interaction_pearson","ndcg_at_5","precision_at_5")
LABELS=("약물 내 ρ","약물 내 r","상호작용 ρ","상호작용 r","NDCG@5","Precision@5")


def md_table(headers,rows):
    return ["| "+" | ".join(map(str,headers))+" |","| "+" | ".join("---" for _ in headers)+" |",
            *["| "+" | ".join(map(str,row))+" |" for row in rows],""]


def put(path,lines):
    path.write_text("\n".join(lines).rstrip()+"\n")


def read_score():
    score=read_json(SCORED/"score_receipt.json")
    claims=read_json(SCORED/"consumed.json")
    if claims["score_passes"]!=1 or score["score_passes"]!=1 or score["claim_sha256"]!=sha256(SCORED/"consumed.json"):
        raise ValueError("Repeated or mismatched outer release")
    if score["forecasts"]!=claims["forecasts"]:
        raise ValueError("Score evaluated an unregistered fit")
    if any(sha256(SCORED/k)!=h for k,h in score["tables"].items()):
        raise ValueError("Score table changed after its single pass")
    return score


def evidence(lock,score):
    frames,hashes,_=preflight()
    if hashes!=score["forecasts"] or set(frames)!=set((*SPLITS,"external")):
        raise ValueError("Independent seal verification differs from the scored set")
    source=read_json(ROOT/"splits/manifest.json")
    sha=sha256(ROOT/"splits/manifest.json")
    step0=read_json(ROOT/"step0_verification.json")
    if (step0["audited_atlas_count"],step0["full_partition_sha256"],step0["query_outcomes_for_outer_consumed"]) != (60,sha,False):
        raise ValueError("The parity and group audit must precede the final report")
    if sha!=score["source_manifest_sha256"]:
        raise ValueError("Source fold registry drifted")
    when=[sum(len(frames[f"{track}-{fold}"]) for fold in range(3)) for track in ("drug","cell")]
    if when!=[670830,670830] or sum(len(frames[name]) for name in SPLITS[6:])!=45428:
        raise ValueError("Coverage dropped or duplicated outer passes")
    if len(score["tissue_roster"])!=28 or score["parts"]["LICOB"]!=38*76 or score["parts"]["GDSC1"]!=39545:
        raise ValueError("Missing registered tissue, donor or GDSC1 support")
    registered=set()
    for part in ("r1","r2"):
        lockfile=ROOT/"locks"/f"{part}-selection.json"
        if not lockfile.exists():
            raise ValueError(f"Missing full-round registration: {part}")
        seal=read_json(lockfile)
        if not seal["no_outer_scores_loaded"]:
            raise ValueError("An inner choice used outer responses")
        for split,families in seal["points"].items():
            for family,values in families.items():
                ready={row["id"] for row in values if row["stage"]==part}
                if len(ready)<2:
                    raise ValueError(f"Missing meaningful two-round exploration: {split}/{family}/{part}")
                registered |= {(split,family,part,k) for k in ready}
    return {
        "score_passes":1,"outcomes_opened_by_independent_verification":False,
        "worktree_base":"e90c8e2cf684b6c332e34a678239f8770cd485a7",
        "outer_fold_count":15,"external_datasets":3,"family_count":5,"seeds":list(SEEDS),
        "checkpoint_count":sum(len(x["checkpoints"]) for x in hashes.values()),
        "minimum_full_round_hyperparameter_brackets":len(registered),
        "forecast_supports":{name:len(frame) for name,frame in frames.items()},
        "score_sha256":sha256(SCORED/"score_receipt.json"),
        "partition_manifest_sha256":sha,"step0_sha256":sha256(ROOT/"step0_verification.json"),"frozen_sha256":sha256(ROOT/"locks/frozen.json"),
    }


def previous_metrics(points):
    report=read_json(STUDY/"score/score_receipt.json")
    old=STUDY/"score/H2_METRICS.tsv"
    if sha256(old)!=report["results"][old.name]:
        raise ValueError("Published H2 comparison table differs")
    raw=pd.read_csv(old,sep="\t")
    raw=raw[raw.scope.eq("DRP") & raw.screen.eq("macro")]
    values=[]
    for support,label,reference in (("all","all","H2"),("lincs724","champion_observed","H2"),
                                    ("all","all","C1"),("structure462","champion_unavailable","C1")):
        method="champion" if reference=="H2" else "C1"
        a=raw[raw.support.eq(label)&raw.arm.eq(method)]
        if a.empty:
            continue
        current=points[points.scope.eq("drug")&points.screen.eq("macro")&points.support.eq(support)&
                        points.model.eq(reference)&points.seed.eq("fixed")]
        for metric in DISPLAY:
            p=current.loc[current.metric.eq(metric),"value"].iloc[0]
            past=a[metric].dropna().iloc[0]
            if not np.isclose(p,past,atol=2e-10,rtol=0):
                raise ValueError(f"H2 same-row replay mismatch on {support}/{reference}/{metric}")
            values.append({"support":support,"model":reference,"metric":metric,"historical":past,"replay":p})
    return pd.DataFrame(values)


def prior_held_points(points):
    root=V1/"outer-score"
    receipt=read_json(root/"score_receipt.json")
    file=root/"H3_METRICS.tsv"
    if sha256(file)!=receipt["output_sha256"][file.name]:
        raise ValueError("H3-v1 control metrics changed")
    previous=pd.read_csv(file,sep="\t")
    previous=previous[previous.screen.eq("macro")]
    mapping={"ndcg_at_5":"ndcg_at_5","precision_at_k":"precision_at_5","within_drug_spearman":"spearman"}
    comparisons=[]
    for source,reference,seed in (("structured","structured_v1","fixed"),
                                  ("drug_ridge","drug_ridge_v1","fixed"),("G2","H3v1","mean")):
        for axis,new in mapping.items():
            value=previous[previous.model.eq(source)&previous.seed.eq(seed)&previous.metric.eq(axis)].value.iloc[0]
            actual=points[points.scope.eq("held")&points.model.eq(reference)&points.screen.eq("macro")&
                           points.seed.eq(seed)&points.metric.eq(new)].value.iloc[0]
            if not np.isclose(actual,value,atol=2e-10,rtol=0):
                raise ValueError(f"Historical held-screen formula drifted for {reference}/{new}")
            comparisons.append({"model":reference,"metric":new,"historical":value,"replay":actual})
    return pd.DataFrame(comparisons)


def inner_results(lock):
    paths=sorted((ROOT/"locks").glob("*-selection.json"),
                 key=lambda x:(0 if x.name.startswith("r") else 1,len(x.name),x.name))
    completed=read_json(paths[-1])
    if completed["next"] is not None or sha256(paths[-1])!=lock["selection_ledger_sha256"]:
        raise ValueError("The frozen set lacks a closed frontier ledger")
    grid,curves=[],[]
    for split,study in completed["points"].items():
        for family,options in study.items():
            picked=lock["splits"][split]["families"][family]["id"]
            for row in options:
                config=row["config"]
                key={"split":split,"family":family,"round":row["stage"],"config_id":row["id"]}
                grid.append(key|row["point"]|{"utility":row["utility"],"budget":row["selected_budget"],
                                            "frozen":row["id"]==picked,"configuration":str(config)})
                for i in range(3):
                    root=ROOT/"fits"/row["stage"]/split/f"inner-{i}"/family/row["id"]
                    meta=read_json(root/"receipt.json")
                    path=root/"curves.parquet"
                    if sha256(root/"receipt.json")!=row["receipt_sha256"][i] or sha256(path)!=meta["curve_sha256"]:
                        raise ValueError("Inner curve evidence changed")
                    trial=pd.read_parquet(path)
                    chosen=row["id"]==picked
                    if not chosen and len(trial)>12:
                        trainonly=np.linspace(0,len(trial)-1,num=11,dtype=int)
                        trial=trial.iloc[np.unique(np.r_[trainonly,len(trial)-1])]
                    curves.append(trial.assign(**key,inner=i,frozen=chosen,fit_wall_seconds=meta["training"]["wall_seconds"],
                                               best_iteration=meta["training"]["best_iteration"]))
    return pd.DataFrame(grid),pd.concat(curves,ignore_index=True),completed


def jobs():
    logs=ROOT/"logs"
    initial=[
        ("data",34799,"cpu","-",""),("kernel-bank",34817,"A100","0-14%3",""),
        ("atlas-first",34833,"cpu","0",""),("atlas",34892,"cpu","1-14%4",""),
        ("tree-smoke",34920,"cpu","-",""),("model-qa",34929,"cpu","-",""),
    ]
    for name,id_ in pd.read_csv(logs/"round1-jobs.tsv",sep="\t",header=None).itertuples(index=False,name=None):
        initial.append((name,id_,"A100" if "gpu" in name else "cpu","limit3" if "gpu" in name else "-", ""))
    frame=pd.DataFrame(initial,columns=["stage","job_id","resource","array","dependency"])
    if (logs/"trajectory.tsv").exists():
        remaining=pd.read_csv(logs/"trajectory.tsv",sep="\t",names=frame.columns,dtype={"dependency":str})
        frame=pd.concat([frame,remaining],ignore_index=True)
    return frame


def report_docs(root,points,ci,curves,grid,evidence,score,replays,lock):
    axis=points[(points.seed.isin(["fixed","mean"])) & points.screen.eq("macro")]
    statuses=score["verdicts"]
    primary=axis[axis.scope.eq("drug")&axis.support.eq("all")]
    checkpoint=lock["external"]
    def score_table(frame):
        pivot=frame.pivot(index="model",columns="metric",values="value")
        std=frame.pivot(index="model",columns="metric",values="seed_sd")
        return md_table(["모델",*LABELS],[(name,*[
            "NA" if m not in point or not np.isfinite(point[m]) else (
                f"{point[m]:.4f} ±{std.at[name,m]:.4f}" if pd.notna(std.at[name,m]) else f"{point[m]:.4f}") for m in DISPLAY])
            for name,point in pivot.iterrows()])
    def compare(model,scope="drug",support="all"):
        selected=ci[(ci.candidate.eq(model)) & ci.baseline.eq("H2") & ci.scope.eq(scope) &
                    ci.support.eq(support) & ci.screen.eq("macro") & ci.metric.isin(AXES)]
        return [(x.metric,f"{x.delta:+.5f}",f"[{x.lower:+.5f}, {x.upper:+.5f}]",x.cluster_axis)
                for x in selected.itertuples()]
    raw=[
        "- **가설** 같은 전체 관측으로 현대적 모델 다섯 패밀리와 구조 커널의 세 평가축을 비교했다.",
        "- **달라진 점** 1,186 exact parent, 전체 RNA 및 결측 인식 멀티뷰, 분할 내 포화 탐색을 적용했다.",
        f"- **결과** 전체 비교는 {score['overall']}, 챔피언 정책은 {statuses['Champion']['verdict']}이며 외부 단일 계열은 {checkpoint['champion']}이다.",
        "- **판정** " + " · ".join(f"{k} {v['verdict']}" for k,v in statuses.items())+".",
        "- **다음 결정** 합동 보호 기준을 통과한 단일 모델만 승격 대상으로 분류했다.",
        "", "# H3-v2 모델 패밀리 비교", "",
        f"3개 seed의 개별 모델 성적 평균과 표본 SD를 표에 표시했다. score_passes = {score['score_passes']}. "
        "Champion은 한 outer 분할 안에서 한 패밀리다. 외부에는 전체 패널에 적합한 하나의 고정 설정을 썼다. "
        "Champion-ensemble은 seed 예측 평균의 기술 비교다.", "",
        "## 신규 골격 · 1,186 parents", "",*score_table(primary),
        "### Champion − H2 · 짝지은 구간","",
        *md_table(["지표","Δ","95% CI","클러스터"],compare("Champion")),
        "### LINCS 관측 724", "",*score_table(axis[axis.scope.eq("drug")&axis.support.eq("lincs724")]),
        "### 구조 지원 462", "",*score_table(axis[axis.scope.eq("drug")&axis.support.eq("structure462")]),
        "## 계통 보류와 세포·스크린 동시 보류", "",
        *score_table(axis[axis.scope.eq("cell")]),*score_table(axis[axis.scope.eq("held")]),
    ]
    for cohort in ("GDSC1","LIMORE","LICOB"):
        raw.extend([f"## {cohort}", "",*score_table(axis[axis.scope.eq(cohort)])])
    best_scope=axis[axis.model.isin(FAMILIES)&axis.scope.eq("drug")&axis.metric.isin(AXES)]
    raw.extend([
        "## 암종과 모델링 범위", "",
        "등록된 28개 암종의 지표, 실측 지지 및 빈 지지를 별도 표에 기록했다. "
        "LIMORE와 LICOB은 기노출 기술 확인이다. GDSC1은 신규 골격 공개 패널이며 일부 세포는 소스와 공유된다.", "",
        *md_table(["LINCS 지원","지표","최고 단일 패밀리","점수"],[
            (scope,key,group.loc[group.value.idxmax(),"model"],f"{group.value.max():.5f}")
            for (scope,key),group in best_scope.groupby(["support","metric"])]),
        "등록한 3축 결합 판정을 각 패밀리에 독립 적용했다. CEILING-CONFIRMED라는 판정은 이번 입력·분할과 실제 탐색 범위에 한정한다. "
        "원 스크린 단위의 차이는 예측·실측을 백분위 순위로 변환해 가산 부분과 상호작용 오류로 분해했다. "
        "[오류 분해](H3V2_ERROR_COMPONENTS.tsv)", "",
        "그리드 시험 범위와 지원별 우위 차이는 "
        "범위 내 모델 한계를 설명하는 근거다. [28개 암종 지표](H3V2_PANCANCER_BY_TISSUE.tsv) · "
        "[전체 짝구간](H3V2_PAIRED_95CI.tsv) · [그리드와 학습곡선](ROUND1_TO_2_FAMILY_TUNING.md)", "",
        "[전체 지표](H3V2_METRICS.tsv) · [채점 영수증](score_receipt.json) · [검증](verification.json) · "
        "[모델 카드](model_card.md) · [Slurm 원장](SLURM_JOBS.tsv)",
    ])
    put(root/"RESULTS.md",raw)
    sweep=grid.groupby(["family","round"]).agg(configurations=("config_id","count"),
               inner_best=("utility","max"),typical_best_budget=("budget","median")).reset_index()
    wall=curves.drop_duplicates(["split","family","round","config_id","inner"]).groupby(
        ["family","round"]).fit_wall_seconds.sum()/3600
    put(root/"ROUND1_TO_2_FAMILY_TUNING.md",[
        "# 계열별 Inner CV 학습 및 포화 탐색", "",
        "각 행은 외부 보류 지원과 독립인 3개 inner 분할의 고정 설정 비교다. 블록 미니배치는 관측만 방문했다. "
        "다음 표는 후보의 최대 inner 점수와 wall-clock 학습 시간을 참조용으로 집계한다.", "",
        *md_table(["패밀리","라운드","분할별 설정 수 합","내측 최상 점수","최적 budget 중앙값","총 fit 시간(h)"],[
          (x.family,x.round,x.configurations,f"{x.inner_best:.4f}",x.typical_best_budget,f"{wall[x.family,x.round]:.2f}")
          for x in sweep.itertuples()]),
        "LightGBM은 L2/Huber/LambdaRank를, FiLM은 반복 조건부 residual 블록을, Transformer는 결측 마스크가 있는 교차 "
        "어텐션을 사용했다. GNN은 GINE/GATv2/AttentiveFP와 등록된 CLAMP·Uni-Mol 서브암을 포함했다. "
        "하이브리드는 H2 OOF를 기본값으로 고정하고 두 단계 트리 또는 공동 손실 신경망으로 보정했다.", "",
        "학습곡선은 같은 iteration의 train 및 validation을 병기한다. Validation 모델은 best iteration으로 재적합했다. "
        "포화점 주위의 설정, 실제 검증점 및 라운드별 동결 여부는 설정 TSV에 있다.",
        "[학습 곡선](H3V2_FAMILY_LEARNING_CURVES.tsv) · [학습 곡선 그림](H3V2_FAMILY_LEARNING_CURVES.png) · [포화 탐색](H3V2_GRID_SATURATION.tsv) · [최종 동결](final_lock.json)",
    ])
    put(root/"model_card.md",[
       "# H3-v2 연구용 감수성 모델", "",
       f"외부 단일 모델은 {checkpoint['champion']}, 설정 {checkpoint['operating_point']['id']}이다. "
       "모델 checkpoint와 fit-only 특징 정의는 G1 final 및 features에 보관했다.", "",
       f"등록 seed는 {', '.join(map(str,SEEDS))}다. 생체 밖 공개 스크린의 감수성 순위와 상호작용 점수를 평가했다. "
       "환자 치료 선택·용량·현장 효용 평가는 범위 밖이다. 데스크톱 배포 모델과 임상 성능을 보고하는 문서는 별도다.", "",
       "[데이터와 지원](DATA_SUPPORT_AND_EXPOSURE.md) · [결과](RESULTS.md) · [동결](final_lock.json)",
    ])


def main():
    score=read_score()
    lock=read_json(ROOT/"locks/frozen.json")
    audit=evidence(lock,score)
    OUTPUT.mkdir(parents=True,exist_ok=False)
    for name in [*score["tables"],"score_receipt.json"]:
        shutil.copy2(SCORED/name,OUTPUT/name)
    shutil.copy2(ROOT/"locks/frozen.json",OUTPUT/"final_lock.json")
    metrics=pd.read_csv(OUTPUT/"H3V2_METRICS.tsv",sep="\t")
    ic=pd.read_csv(OUTPUT/"H3V2_PAIRED_95CI.tsv",sep="\t")
    progress,curves,completion=inner_results(lock)
    progress.to_csv(OUTPUT/"H3V2_GRID_SATURATION.tsv",sep="\t",index=False)
    curves.to_csv(OUTPUT/"H3V2_FAMILY_LEARNING_CURVES.tsv",sep="\t",index=False)
    jobs().to_csv(OUTPUT/"SLURM_JOBS.tsv",sep="\t",index=False)
    previous=previous_metrics(metrics)
    previous.to_csv(OUTPUT/"H3V2_H2_REPLAY.tsv",sep="\t",index=False)
    held=prior_held_points(metrics)
    held.to_csv(OUTPUT/"H3V2_H3V1_REPLAY.tsv",sep="\t",index=False)
    report_docs(OUTPUT,metrics,ic,curves,progress,audit,score,previous,lock)
    plotter=Path(__file__).resolve().parents[2]/"experiments/nextgen_h3_v2/plot_curves.py"
    environment=dict(os.environ,MPLCONFIGDIR=str(ROOT/"tmp/matplotlib"))
    run(["/usr/bin/python3",str(plotter),str(OUTPUT)],check=True,env=environment)
    publish_json(OUTPUT/"verification.json",audit|{
        "registered_replay_metrics":len(previous),"family_learning_curve_rows":len(curves),
        "historical_held_metric_replays":len(held),"saturation_grid_rows":len(progress),"first_pass_verification_utc":datetime.now(timezone.utc).isoformat(),
        "deliverables":{p.name:sha256(p) for p in OUTPUT.iterdir()},
    })
    print("Verified final H3-v2 artifact assembly", flush=True)


if __name__=="__main__":
    main()
