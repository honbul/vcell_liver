"""Verify the completed H3-v2 experiment and assemble versionable aggregate artifacts."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
from subprocess import run
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from nextgen_h2.run import read_json
from vc_build.io import publish_json,sha256

from .atlas import model_input_digest
from .data import ROOT,SPLITS,STUDY,V1
from .grid import FAMILIES,SEEDS
from .quality import AXES
from .score import preflight
from .selection import PRE_UNSEALING_FREEZE

OUTPUT=ROOT/"reports-final"
SCORED=ROOT/"score"
DISPLAY=("spearman","pearson","interaction_spearman","interaction_pearson","ndcg_at_5","precision_at_5")
LABELS=("약물 내 ρ","약물 내 r","상호작용 ρ","상호작용 r","NDCG@5","Precision@5")
SLURM_FIELDS = re.compile(r"(?:^| )(JobId|ArrayJobId|ArrayTaskId|JobState|StartTime|EndTime|NodeList)=([^\s]+)")
SLURM_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T")
TERMINAL_STATES = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
REPLAY_LIBRARIES = {
    "cuda": "deps/lightgbm-cuda-4.6.0/runtime/lightgbm/lib/lib_lightgbm.so",
    "opencl": "deps/lightgbm/lib/lib_lightgbm.so",
}


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
    comparisons = [(support, archive_support, model) for support, archive_support in (
        ("all", "all"), ("lincs724", "champion_observed"), ("structure462", "champion_unavailable")
    ) for model in ("H2", "C1")]
    for support,label,reference in comparisons:
        method="champion" if reference=="H2" else "C1"
        a=raw[raw.support.eq(label)&raw.arm.eq(method)]
        if a.empty:
            raise ValueError(f"Published H2 comparison is missing {support}/{reference}")
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


def sampled_learning_curve(trial, training):
    groups = trial.groupby("branch", sort=False, dropna=False) if "branch" in trial else [("neural", trial)]
    budgets = training.get("branch_iterations", [training["best_iteration"]])
    samples = []
    for position, (branch, frame) in enumerate(groups):
        if len(frame) > 12:
            best = budgets[position] if branch != "selected" else training["best_iteration"]
            measured = np.flatnonzero(frame.iteration.to_numpy() == best)
            regular = np.linspace(0, len(frame) - 1, num=11, dtype=int)
            frame = frame.iloc[np.unique(np.r_[regular, measured])]
        samples.append(frame)
    return pd.concat(samples).sort_index()


def selection_ledger(lock):
    paths=sorted((ROOT/"locks").glob("*-selection.json"),
                 key=lambda x:(0 if x.name.startswith("r") else 1,len(x.name),x.name))
    path = paths[-1]
    completed=read_json(path)
    if completed["next"] is not None or sha256(path)!=lock["selection_ledger_sha256"]:
        raise ValueError("The frozen set lacks a closed frontier ledger")
    return path, completed


def inner_results(lock, completed):
    step0 = read_json(ROOT / "step0_verification.json")["folds"]
    grid,curves=[],[]
    input_evidence = {}
    for split,study in completed["points"].items():
        for registered in step0[split]["feature_receipts"]:
            role = registered["role"]
            if role == "outer":
                continue
            stamp = read_json(ROOT / "features" / split / role / "receipt.json")
            if (stamp | {"role": role}) != registered:
                raise ValueError(f"The atlas differs from the feature audit for {split}/{role}")
            input_evidence[f"{split}/{role}"] = {
                "atlas_sha256": stamp["atlas_sha256"], "fit_rows": stamp["fit_rows"],
                "model_input_sha256": model_input_digest(stamp["atlas_sha256"]), "verified_trials": 0,
            }
        for family,options in study.items():
            picked=lock["splits"][split]["families"][family]["id"]
            for row in options:
                config=row["config"]
                key={"split":split,"family":family,"round":row["stage"],"config_id":row["id"]}
                grid.append(key | row["point"] | {"utility":row["utility"],"budget":row["selected_budget"],
                            "branch_budgets":row["selected_branch_budgets"],"frozen":row["id"]==picked,
                            "max_budget_pressure_iteration":row["max_budget_pressure_iteration"],
                            "configuration":json.dumps(config, sort_keys=True)} | config)
                for i in range(3):
                    root=ROOT/"fits"/row["stage"]/split/f"inner-{i}"/family/row["id"]
                    meta=read_json(root/"receipt.json")
                    path=root/"curves.parquet"
                    if sha256(root/"receipt.json")!=row["receipt_sha256"][i] or sha256(path)!=meta["curve_sha256"]:
                        raise ValueError("Inner curve evidence changed")
                    inputs = input_evidence[f"{split}/inner-{i}"]
                    if (meta["atlas_sha256"], meta["model_input_sha256"], meta["training"]["train_rows"]) != (
                        inputs["atlas_sha256"], inputs["model_input_sha256"], inputs["fit_rows"]
                    ) or not meta["lincs_full_unclipped"] or meta["query_has_outer_labels"]:
                        raise ValueError(f"Verified inputs differ from the measured trial for {root}")
                    inputs["verified_trials"] += 1
                    trial=pd.read_parquet(path)
                    chosen=row["id"]==picked
                    if not chosen and len(trial)>12:
                        trial=sampled_learning_curve(trial, meta["training"])
                    device=meta["training"].get("device","cpu" if config["backend"]=="gbdt" else "cuda")
                    curves.append(trial.assign(**key,inner=i,frozen=chosen,backend=config["backend"],device=device,
                                               fit_wall_seconds=meta["training"]["wall_seconds"],
                                               best_iteration=meta["training"]["best_iteration"]))
    return pd.DataFrame(grid),pd.concat(curves,ignore_index=True),input_evidence


def job_observations(path, job_ids):
    """Summarize observed allocation and state by Slurm job or array parent."""
    allowed = {str(job_id) for job_id in job_ids}
    observed = defaultdict(dict)
    nodes = defaultdict(set)
    starts, ends = {}, {}
    if not path.is_file():
        return {}
    with path.open() as source:
        for line in source:
            if not line.endswith("\n"):
                continue
            fields = dict(SLURM_FIELDS.findall(line))
            parent = fields.get("ArrayJobId", fields.get("JobId"))
            task = fields.get("ArrayTaskId", "0")
            if parent not in allowed or not task.isdecimal() or "JobState" not in fields:
                continue
            state = fields["JobState"]
            observed[parent][task] = state
            node = fields.get("NodeList", "(null)")
            if node != "(null)":
                nodes[parent].add(node)
            if state != "PENDING" and node != "(null)" and SLURM_TIME.match(started := fields.get("StartTime", "")):
                starts[parent] = min(starts.get(parent, started), started)
            if state in TERMINAL_STATES and SLURM_TIME.match(ended := fields.get("EndTime", "")):
                ends[parent] = max(ends.get(parent, ended), ended)
    return {
        int(parent): {
            "observed_elements": len(tasks),
            "observed_last_states": " | ".join(f"{name}={count}" for name, count in sorted(Counter(tasks.values()).items())),
            "observed_nodes": ",".join(sorted(nodes[parent])),
            "first_observed_start_local": starts.get(parent, ""),
            "last_observed_terminal_end_local": ends.get(parent, ""),
        }
        for parent, tasks in observed.items()
    }


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
    adjustments = logs / "runtime-adjustments.tsv"
    if adjustments.is_file():
        history = pd.read_csv(adjustments, sep="\t", header=None, dtype=str, keep_default_na=False,
                              names=["time", "job_id", "field", "previous", "updated"])
        job_keys = frame.job_id.astype(str)
        changes = history.loc[history.field.eq("ArrayTaskThrottle")].copy()
        changes["record"] = changes.time + " " + changes.previous + "→" + changes.updated
        frame["array_throttle_changes"] = job_keys.map(changes.groupby("job_id").record.agg(" | ".join))
        placement_fields = ["CPUsPerTask", "ExcNodeList", "ReqNodeList", "C4PartialCancel", "MigrationCores",
                            "RestoredPendingIndices"]
        placement = history.loc[history.field.isin(placement_fields)].copy()
        placement["record"] = placement.time + " " + placement.field + " " + placement.previous + "→" + placement.updated
        frame["cpu_resource_changes"] = job_keys.map(placement.groupby("job_id").record.agg(" | ".join))
    observed = job_observations(logs / "slurm-observations.txt", frame.job_id)
    if observed:
        detail = pd.DataFrame.from_dict(observed, orient="index")
        detail["observed_elements"] = detail.observed_elements.astype("Int64")
        frame = frame.join(detail, on="job_id")
    return frame


def device_replays():
    plan = ROOT / "locks/cuda-g3-checks.json"
    if not plan.is_file():
        return [], {}
    cases = read_json(plan)["cases"]
    rows, digests = [], {}
    for device, library in REPLAY_LIBRARIES.items():
        root = ROOT / f"{device}-validation"
        if not root.is_dir():
            continue
        library_stamp = sha256(ROOT / library)
        for case in cases:
            stage, split, inner, family, identity = (case[field] for field in
                ("stage", "split", "inner", "family", "config_id"))
            suffix = Path(stage) / split / f"inner-{inner}" / family / identity
            comparison = root / suffix / "comparison.json"
            if not comparison.is_file():
                continue
            baseline = ROOT / "fits" / suffix / "receipt.json"
            accelerated = comparison.with_name("receipt.json")
            point, cpu, gpu = map(read_json, (comparison, baseline, accelerated))
            if (any(point[field] != value for field, value in case.items()) or point["query_has_outer_labels"]
                    or sha256(baseline) != case["cpu_receipt_sha256"]
                    or point[f"{device}_receipt_sha256"] != sha256(accelerated)
                    or point[f"{device}_library_sha256"] != library_stamp):
                raise ValueError(f"The LightGBM device replay changed for {device}/{suffix}")
            for key in ("config", "seed", "model_input_sha256", "query_has_outer_labels"):
                if cpu[key] != gpu[key]:
                    raise ValueError(f"The LightGBM device replay changed {key} for {device}/{suffix}")
            times = cpu["training"]["wall_seconds"], gpu["training"]["wall_seconds"]
            deltas = {axis: gpu["score"][axis] - score for axis, score in cpu["score"].items()}
            if (cpu["query_has_outer_labels"] or cpu["training"]["device"] != "cpu" or gpu["training"]["device"] != device
                    or (point["cpu_seconds"], point[f"{device}_seconds"]) != times
                    or point["speedup"] != times[0] / times[1] or point["metric_deltas"] != deltas
                    or point["utility_delta"] != gpu["utility"] - cpu["utility"]):
                raise ValueError(f"The measures do not reproduce for {device}/{suffix}")
            digests[str(comparison.relative_to(ROOT))] = sha256(comparison)
            rows.append((device, stage, split, inner, identity, *(f"{t / 60:.1f}" for t in times),
                         f"{point['speedup']:.2f}×", f"{point['utility_delta']:+.5f}",
                         *(f"{deltas[axis]:+.5f}" for axis in AXES)))
    return rows, digests


def error_diagnosis(root):
    errors = pd.read_csv(root / "H3V2_ERROR_COMPONENTS.tsv", sep="\t")
    errors["model"] = errors.arm.str.split("__").str[0]
    components = ["rank_mse", "additive_mse", "interaction_mse"]
    errors[components] = errors[components].multiply(errors.rows, axis=0)
    totals = errors.groupby(["support", "model"])[["rows", *components]].sum()
    totals[components] = totals[components].divide(totals.rows, axis=0)
    rows = []
    for support, block in totals.groupby(level="support", sort=False):
        h2 = block.loc[(support, "H2"), components]
        for (_, model), values in block.iterrows():
            delta = values[components] - h2
            rows.append((support, model, *(f"{value:+.5f}" for value in delta)))
    return rows


def tissue_deltas(root, roster):
    table = pd.read_csv(root / "H3V2_PANCANCER_BY_TISSUE.tsv", sep="\t")
    selected = table[table.screen.eq("macro") & table.support.eq("all") & table.metric.isin(AXES)]
    refs = selected[selected.model.eq("H2") & selected.seed.eq("fixed")].set_index(["tissue", "metric"])
    champ = selected[selected.model.eq("Champion") & selected.seed.eq("mean")].set_index(["tissue", "metric"])
    paired = refs[["value", "unit_count"]].join(champ[["value"]], how="inner", lsuffix="_h2", rsuffix="_champ")
    paired["delta"] = paired.value_champ - paired.value_h2
    rows = []
    for tissue in roster:
        effects = []
        support = []
        for metric in AXES:
            key = (tissue, metric)
            valid = key in paired.index and np.isfinite(paired.loc[key, "delta"])
            effects.append(f"{paired.loc[key, 'delta']:+.4f}" if valid else "미추정")
            support.append(int(paired.loc[key, "unit_count"]) if valid else 0)
        rows.append((tissue, *effects, support[0], support[-1]))
    return rows


def architecture_specs(grid):
    fitted = defaultdict(list)
    for config in grid.configuration.map(json.loads):
        branch = config["objective"] if config["backend"] == "gbdt" else config["arch"]
        fitted[config["family"], branch].append(config)

    def span(configs, key):
        low, high = min(c[key] for c in configs), max(c[key] for c in configs)
        return f"{low:g}" if low == high else f"{low:g}–{high:g}"

    rows = []
    for family in FAMILIES:
        for (registered, branch), configs in sorted(fitted.items()):
            if registered != family:
                continue
            tree = configs[0]["backend"] == "gbdt"
            width = span(configs, "leaves" if tree else "width")
            limit = span(configs, "trees" if tree else "epochs")
            structure = "—"
            if branch == "attention":
                structure = f"헤드 {span(configs, 'heads')}"
            elif branch == "gnn":
                structure = f"메시지 층 {span(configs, 'message_depth')}"
            rows.append((family, branch, width, span(configs, "depth"), limit, span(configs, "learning_rate"),
                         "—" if tree else span(configs, "dropout"),
                         span(configs, "lambda_l2" if tree else "weight_decay"), structure))
    return [
        "## 모델 구조와 실측 탐색 범위", "",
        "- **GBDT** LightGBM 4.6.0의 결정론적 CPU 학습이다. 동일 atlas의 표형 입력으로 약물 평균을 뺀 "
        "잔차를 L2 또는 Huber로 적합한다. LambdaRank는 약물 평균을 초기 점수로 사용하고 세포×스크린 그룹의 NDCG@5를 최적화한다. "
        "최대 bin은 63이며 내부 objective가 130회 개선되지 않으면 중단한다.",
        "- **FiLM-ResNet** 다중뷰 약물 인코더가 각 residual 블록의 γ·β를 생성한다. 세포 벡터에 직접 타깃·경로 매칭 벡터를 더하고, "
        "LayerNorm과 폭 2배의 GELU 피드포워드를 거친다. LINCS 비관측이면 섭동 임베딩을 마스킹한다.",
        "- **Cross-Attention** 세포 전역 RNA 토큰, 경로 44개, 경로·TF 모듈 12개를 키·값으로 사용한다. "
        "질의는 직접 매칭, 화학 지문, 물성, 타깃, LINCS의 다섯 토큰이다. "
        "약물 토큰 간 어텐션 이후 세포 토큰에 교차 어텐션을 수행하며, 비관측 LINCS는 패딩 마스크로 제외한다. "
        "교차 어텐션은 등록 설정의 폭과 헤드 수를 사용하며 직접 매칭 토큰에서 점수를 읽는다.",
        "- **분자 GNN** RDKit 원자·결합을 입력받아 GINE, GATv2 또는 AttentiveFP를 3–5층 적용하고 "
        "마스크가 있는 어텐션 풀링을 한다. 그래프 표현을 ECFP4 표현에 더해 공통 FiLM-ResNet으로 전달한다. "
        "CLAMP와 Uni-Mol은 각각 가용성 마스크를 둔 별도 서브암이다.",
        "- **하이브리드** 훈련 행의 H2 OOF 감수성을 고정한다. 트리는 L2 잔차 뒤에 NDCG@5 LambdaRank를 순차 적용한다. "
        "신경망은 잔차 FiLM 또는 교차 어텐션으로 SmoothL1, 약물 내 Pearson, 세포 내 ListNet을 공동 최적화한다.", "",
        "신경망은 세포×약물 블록에서 관측 행을 매 epoch 한 번 방문한다. AdamW, 최대 8 epoch warmup, "
        "cosine 감쇠와 내부 검증 조기 종료를 사용한다. 2차 탐색에는 기울기 크기로 손실의 균형을 조정한 설정과 "
        "약물 평균·세포 효과·상호작용의 명시적 분해를 포함한다.", "",
        "아래 범위는 전체 내부 분할에서 실제로 적합한 등록 설정을 집계한 값이다. 트리 계열의 용량은 잎 수이며 "
        "신경망의 용량은 은닉 폭이다. 예산은 등록된 최대 트리 수 또는 epoch다. "
        "이중 목적 트리는 두 분기에 예산을 나눠 사용한다.", "",
        *md_table(["패밀리", "목적 또는 구조", "잎 수 또는 폭", "깊이", "예산 상한", "학습률", "Dropout",
                   "L2 또는 weight decay", "어텐션·그래프"], rows),
    ]


def report_docs(root,points,ci,curves,grid,score,lock,*,replays=(),open_neighbors=None):
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
    labels = dict(zip(DISPLAY, LABELS))
    summary_effects = ", ".join(f"{labels[axis]} {delta}" for axis,delta,_,_ in compare("Champion"))
    verdict = statuses["Champion"]["verdict"]
    promoted = [name for name in (*FAMILIES, "Champion") if statuses[name]["verdict"] == "GO-MODEL"]
    next_decision = ("승격 기준 충족 모델은 " + ", ".join(promoted) + "이다." if promoted else
                     "현재 패널과 동결된 탐색 범위에서 승격 기준을 충족한 모델이 없었다.")
    gates = ("minimum_improvement_and_ci", "other_two_axes_protected", "two_screens_improved",
             "gdsc1_three_axes_protected")
    gate_rows = []
    for model in (*FAMILIES, "Champion"):
        for metric in AXES:
            decision = statuses[model]["by_axis"][metric]
            gate_rows.append((model, labels[metric], *("충족" if decision[key] else "미충족" for key in gates)))
    raw=[
        "- **가설** 같은 전체 관측으로 현대적 모델 다섯 패밀리와 구조 커널의 세 평가축을 비교했다.",
        "- **달라진 점** 1,186 exact parent, 전체 RNA 및 결측 인식 멀티뷰, 분할 내 포화 탐색을 적용했다.",
        f"- **결과** 신규 골격 전체에서 Champion의 H2 대비 변화는 {summary_effects}다. 전체 판정은 {score['overall']}, Champion은 {verdict}다.",
        "- **판정** " + " · ".join(f"{k} {v['verdict']}" for k,v in statuses.items())+".",
        f"- **다음 결정** {next_decision}",
        "", "# H3-v2 모델 패밀리 비교", "",
        f"3개 seed의 개별 모델 성적 평균과 표본 SD를 표에 표시했다. score_passes = {score['score_passes']}. "
        "Champion은 한 outer 분할 안에서 한 패밀리다. 외부에는 전체 패널에 적합한 하나의 고정 설정을 썼다. "
        "Champion-ensemble은 seed 예측 평균의 기술 비교다.", "",
        "## 신규 골격 · 1,186 parents", "",*score_table(primary),
        "### Champion − H2 · 짝지은 구간","",
        *md_table(["지표","Δ","95% CI","클러스터"],compare("Champion")),
        "### 패밀리별 승격 조건", "",
        "개선 축마다 H2 대비 사전 지정 효과 크기와 95% 하한, 다른 두 축의 비열등, "
        "세 스크린 중 두 개의 개선, GDSC1 세 축의 비열등을 함께 적용했다.", "",
        *md_table(["모델", "개선 축", "효과와 95% 하한", "다른 두 축 비열등", "스크린 재현", "GDSC1 비열등"], gate_rows),
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
        "신규 골격 비교에서 암종별 Champion − H2 점수 차이다. 지지 열은 각 지표 계산에 포함된 고유 약물과 세포 수다. "
        "LIMORE와 LICOB은 기노출 기술 확인이다. GDSC1은 신규 골격 공개 패널이며 일부 세포는 소스와 공유된다.", "",
        *md_table(["암종", "Δ약물 내 ρ", "Δ상호작용 ρ", "ΔNDCG@5", "약물 지지", "세포 지지"],
                  tissue_deltas(root, score["tissue_roster"])),
        *md_table(["LINCS 지원","지표","최고 단일 패밀리","점수"],[
            (scope,key,group.loc[group.value.idxmax(),"model"],f"{group.value.max():.5f}")
            for (scope,key),group in best_scope.groupby(["support","metric"])]),
        "등록한 3축 결합 판정을 각 패밀리에 독립 적용했다. CEILING-CONFIRMED라는 판정은 이번 입력·분할과 실제 탐색 범위에 한정한다. "
        "원 스크린 단위의 차이는 예측·실측을 백분위 순위로 변환해 가산 부분과 상호작용 오류로 분해했다. "
        "[오류 분해](H3V2_ERROR_COMPONENTS.tsv)", "",
        "아래는 같은 스크린·분할·지원에 대한 H2 대비 순위 평균제곱오차 차이다. 각 항목은 관측 행으로 가중했다. "
        "낮을수록 오차가 작으며 신규 모델의 오차 분해에는 사전 지정 첫 번째 seed를 사용했다.", "",
        *md_table(["LINCS 지원", "모델", "전체 ΔMSE", "가산 ΔMSE", "상호작용 ΔMSE"], error_diagnosis(root)),
        "그리드 시험 범위와 지원별 우위 차이는 "
        "범위 내 모델 한계를 설명하는 근거다. [28개 암종 지표](H3V2_PANCANCER_BY_TISSUE.tsv) · "
        "[전체 짝구간](H3V2_PAIRED_95CI.tsv) · [그리드와 학습곡선](ROUND1_TO_2_FAMILY_TUNING.md)", "",
        "[전체 지표](H3V2_METRICS.tsv) · [채점 영수증](score_receipt.json) · [검증](verification.json) · "
        "[모델 카드](model_card.md) · [Slurm 원장](SLURM_JOBS.tsv)",
    ])
    sweep=grid.groupby(["family","round"]).agg(configurations=("config_id","count"),
               inner_best=("utility","max"),typical_best_budget=("budget","median")).reset_index()
    unique_fits=curves.drop_duplicates(["split","family","round","config_id","inner"])
    compute_device = unique_fits.device.str.split(":").str[0].replace({"opencl": "cuda"})
    wall=unique_fits.groupby(["family","round",compute_device]).fit_wall_seconds.sum()/3600
    tuning = [
        "# 계열별 Inner CV 학습 및 포화 탐색", "",
        "각 행은 외부 보류 지원과 독립인 3개 inner 분할의 고정 설정 비교다. 블록 미니배치는 관측만 방문했다. "
        "다음 표는 후보의 최대 inner 점수와 wall-clock 학습 시간을 참조용으로 집계한다.", "",
        *md_table(["패밀리","라운드","분할별 설정 수 합","내측 최상 점수","최적 budget 중앙값","GPU fit 시간(h)","CPU fit 시간(h)"],[
          (x.family,x.round,x.configurations,f"{x.inner_best:.4f}",x.typical_best_budget,
           f"{wall.get((x.family,x.round,'cuda'),0):.2f}", f"{wall.get((x.family,x.round,'cpu'),0):.2f}")
          for x in sweep.itertuples()]),
        *architecture_specs(grid),
        "## 학습 및 포화 곡선", "",
        "학습곡선은 같은 iteration의 train 및 validation을 병기한다. 동결 설정은 기록한 전체 곡선을 싣고, "
        "다른 설정은 분기별로 표본을 남긴다. 각 분기의 첫·마지막 관측 및 최적 iteration을 포함한다. "
        "포화 그림은 Drug-Cold fold 0에서 동결 모델과 같은 구조의 용량별 실측 최고점과, 동결 설정에 맞춘 단일 축 비교를 함께 보여준다. "
        "용량별 최고점에는 다른 설정의 차이도 반영된다. "
        "다른 축과 분할의 측정값은 설정 TSV에 있다.",
        "[학습 곡선](H3V2_FAMILY_LEARNING_CURVES.tsv) · [학습 곡선 그림](H3V2_FAMILY_LEARNING_CURVES.png) · [포화 탐색](H3V2_GRID_SATURATION.tsv) · [포화 그림](H3V2_GRID_SATURATION.png) · [최종 동결](final_lock.json)",
    ]
    if open_neighbors is not None:
        total = sum(len(ids) for arms in open_neighbors.values() for ids in arms.values())
        tuning.extend([
            "", "## 외부 라벨 개봉 전 탐색 상한 개정", "",
            "동일한 r1, r2, boundary-1 내부 학습 4,071건으로 75개 분할·패밀리의 운영점을 동결했다. "
            "단일 축 exact-match 확장에서 작은 점수 차이로 후보가 계속 생성되어 라벨 개봉 전에 boundary-1을 상한으로 정했다. "
            "일부 먼저 실행된 boundary-2 결과는 선택과 동결에서 제외했다.", "",
            f"추가로 측정할 수 있었던 후보는 {total:,}개다. 각 분할·패밀리별 수와 정확한 설정 ID를 남겼다. "
            "포화 곡선과 최종 판정은 동결까지 실제 측정한 범위를 설명한다.", "",
            *md_table(["분할", *FAMILIES], [(split, *(len(arms[family]) for family in FAMILIES))
                                            for split, arms in open_neighbors.items()]),
            "", "[동결 당시 선택 원장과 미측정 후보](inner_selection_ledger.json)",
        ])
        raw.extend(["", "## 내부 탐색 상한", "",
                    "외부 라벨을 열기 전에 boundary-1까지의 4,071건으로 동결했다. "
                    f"추가 탐색 후보 {total:,}개는 원장에 남겼으며 결과는 실제 탐색 범위에 한정한다. "
                    "[동결 당시 선택 원장](inner_selection_ledger.json)"])
    put(root/"RESULTS.md",raw)
    if replays:
        tuning.extend([
            "", "## A40 LightGBM 비교", "",
            "동일 내부 분할, 설정 및 seed를 사용한 Huber 후보의 CPU, CUDA, OpenCL float64 실측이다. "
            "검증종합은 약물 내 ρ·상호작용 ρ·NDCG@5의 가중합이고 차이는 GPU에서 CPU를 뺀 값이다. "
            "본 GBM 후보 선정에는 등록된 CPU 학습 영수증을 사용했다.", "",
            *md_table(["구현", "라운드", "분할", "내부 폴드", "설정", "CPU 분", "A40 분", "배속", "Δ검증종합",
                       "Δ약물 내 ρ", "Δ상호작용 ρ", "ΔNDCG@5"], replays),
        ])
    put(root/"ROUND1_TO_2_FAMILY_TUNING.md",tuning)
    put(root/"model_card.md",[
       "# H3-v2 연구용 감수성 모델", "",
       f"외부 단일 모델은 {checkpoint['champion']}, 설정 {checkpoint['operating_point']['id']}이다. "
       "모델 checkpoint와 fit-only 특징 정의는 G1 final 및 features에 보관했다.", "",
       "## 사용 목적과 입력", "",
       "생체 밖 암 모델의 약물 감수성 순위와 상호작용을 연구한다. 세포 기저 RNA, 경로·TF, 정확 parent의 구조와 ChEMBL 타깃, "
       "관측 시 LINCS 변화를 사용한다. 모델 출력은 스크린 감수성 점수다.", "",
       "## 평가 범위", "",
       f"독립 seed는 {', '.join(map(str,SEEDS))}다. 신규 골격, 신규 세포, 스크린 동시 보류, GDSC1·LIMORE·LICOB에서 평가했다. "
       "GDSC1에 소스와 공유하는 세포가 있고 LIMORE와 LICOB은 기노출 코호트다. 지원과 성능 차이는 결과표에 기록했다.", "",
       "환자 치료 선택, 용량 산정과 임상 효용은 평가 범위 밖이다. 결측 LINCS 지원과 생물학적 플랫폼 차이에서 불확실성이 달라진다.", "",
       "[데이터와 지원](DATA_SUPPORT_AND_EXPOSURE.md) · [결과](RESULTS.md) · [동결](final_lock.json)",
    ])


def assemble(output, score, lock, audit):
    for name in [*score["tables"],"score_receipt.json"]:
        shutil.copy2(SCORED/name,output/name)
    shutil.copy2(ROOT/"locks/frozen.json",output/"final_lock.json")
    shutil.copy2(ROOT/"step0_verification.json", output/"step0_verification.json")
    shutil.copy2(ROOT/"splits/manifest.json", output/"split_manifest.json")
    shutil.copy2(ROOT/"external_roster_verification.json", output/"external_roster_verification.json")
    metrics=pd.read_csv(output/"H3V2_METRICS.tsv",sep="\t")
    ic=pd.read_csv(output/"H3V2_PAIRED_95CI.tsv",sep="\t")
    ledger_path, ledger = selection_ledger(lock)
    frontier = None
    if lock.get("freeze_reason") == PRE_UNSEALING_FREEZE:
        shutil.copy2(ledger_path, output/"inner_selection_ledger.json")
        frontier = ledger["open_neighbors"]
    progress,curves,fit_inputs=inner_results(lock,ledger)
    progress.to_csv(output/"H3V2_GRID_SATURATION.tsv",sep="\t",index=False)
    curves.to_csv(output/"H3V2_FAMILY_LEARNING_CURVES.tsv",sep="\t",index=False)
    jobs().to_csv(output/"SLURM_JOBS.tsv",sep="\t",index=False)
    previous=previous_metrics(metrics)
    previous.to_csv(output/"H3V2_H2_REPLAY.tsv",sep="\t",index=False)
    held=prior_held_points(metrics)
    held.to_csv(output/"H3V2_H3V1_REPLAY.tsv",sep="\t",index=False)
    replays,replay_hashes=device_replays()
    report_docs(output,metrics,ic,curves,progress,score,lock,replays=replays,open_neighbors=frontier)
    plotter=Path(__file__).resolve().parents[2]/"experiments/nextgen_h3_v2/plot_curves.py"
    environment=dict(os.environ,MPLCONFIGDIR=str(ROOT/"tmp/matplotlib"))
    run(["/usr/bin/python3",str(plotter),str(output)],check=True,env=environment)
    publish_json(output/"verification.json",audit|{
        "registered_replay_metrics":len(previous),"family_learning_curve_rows":len(curves),
        "historical_held_metric_replays":len(held),"saturation_grid_rows":len(progress),"first_pass_verification_utc":datetime.now(timezone.utc).isoformat(),
        "inner_fit_inputs": fit_inputs,
        "gbdt_device_replay_receipts": replay_hashes,
        "deliverables":{p.name:sha256(p) for p in output.iterdir()},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    target = parser.parse_args().output.resolve()
    if not target.is_relative_to(ROOT.resolve()) or target == ROOT.resolve():
        raise ValueError("The report directory must be inside the registered study root")
    if target.exists():
        raise FileExistsError(target)
    score=read_score()
    lock=read_json(ROOT/"locks/frozen.json")
    audit=evidence(lock,score)
    with TemporaryDirectory(prefix=".reports-", dir=target.parent) as stage:
        output = Path(stage) / target.name
        output.mkdir()
        assemble(output, score, lock, audit)
        output.rename(target)
    print("Verified final H3-v2 artifact assembly", flush=True)


if __name__=="__main__":
    main()
