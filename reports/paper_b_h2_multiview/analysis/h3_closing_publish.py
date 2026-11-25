"""Publish the closing audit after checking both replay and all stage receipts."""
from datetime import datetime, timezone
import platform

from h3_closing_inputs import *
from h3_closing_run import COUNTS, STAGE, gate

REPORT = "H3V2_CLOSING_AUDIT_AND_PAPER_B_BRIDGE.md"
RECEIPT = "tables/h3v2_closing_audit_receipt.json"
OLD_STUDIES = (
    "PAPER_B_POSTHOC_AUDIT_AND_CLAIM_LEDGER.md", "E1_E4_E5_PAIRED_AUDIT.md",
    "tables/e5_trackb_posthoc_audit_receipt.json", "tables/e5_trackb_machine_replay_receipt.json",
    "tables/e1_e4_e5_paired_audit_receipt.json", "tables/e5_provenance_receipt.json",
    "tables/e5_trackb_screen_fold_receipt.json", "tables/e5_trackb_family_receipt.json",
    "tables/e5_trackb_cross_family_receipt.json",
)
ANCHORS = {
    "research/nextgen-h3-v2":"c4428216f64ca9f165acd2f6d01c5caece2aa36c",
    "research/followup-h2-e5-upgrade":"1287c0baa4baba649665c54f2efafa9e7c026eb7",
    "research/followup-h2-e1e3":"9d6636632bef164dc761b81565b0c97d4f77b6d0",
    "research/nextgen-h2-v2":"dafa4d82ff5e568ace6a58b4275620b34a4061c0",
}
SPEC_SHA = "0d7896398cbbeb279024da7f6c9934f1cac6c1017155b3821f783024da5b70aa"
BASES = ("H2-v2","Cchem-int","P3B-Shared","H2-v3-Combined")
ARMS = ("H2-v1",*BASES,"P3B-InScreen","Champion","gbdt","film","attention","gnn","hybrid")
AXES = {"spearman":"약물 내 Spearman ρ","interaction_spearman":"상호작용 Spearman ρ","ndcg_at_5":"NDCG@5"}
SUPPORT = {"all":"all","lincs_obs":"LINCS 관측","lincs_unobs":"LINCS 미관측"}
SCOPES = ("Drug-Cold","GDSC1","LIMORE","LICOB","LICOB18")
PANELS = tuple((s,u) for s in SCOPES for u in SUPPORT)
CONTRACT = ("† Drug-Cold의 H2-v3-Combined와 P3B-InScreen은 알려진 스크린의 τ f_s를 사용한다. "
            "‡ 외부 H2-v3-Combined의 query에는 이 항이 없으며 shared inference를 사용한다. P3B-Shared도 shared inference다.")


def table(header, rows):
    line = lambda x: "| "+" | ".join(str(v).replace("|",r"\|") for v in x)+" |"
    return "\n".join([line(header),line(["---"]*len(header)),*(line(r) for r in rows)])


def name(scope,support):
    pretty = {"lincs_obs":"LINCS724","lincs_unobs":"Unobs462"} if scope=="Drug-Cold" else SUPPORT
    return scope+" · "+pretty.get(support,support)


def ci(row):
    return f"{row.delta:+.5f} [{row.lo95:+.5f}, {row.hi95:+.5f}]"


def sign(lo,hi):
    return np.where(lo>0,1,np.where(hi<0,-1,0))


def make_bridge(d, gates, notes):
    m = d["H3V2_VS_H2V2_E5_METRICS.tsv"]
    c = d["H3V2_VS_H2V2_E5_PAIRED_95CI.tsv"]
    held = d["H3V2_HELD68_VS_H2_PAIRED_95CI.tsv"]
    overlap = d["H3V2_FAMILY980_SPLIT_OVERLAP.tsv"]
    sens = d["H3V2_FAMILY980_SENSITIVITY.tsv"]
    nu = d["H3V2_PARTS_AND_NULL_AUDIT.tsv"]
    limits = notes["native"]["extra"]["boundary_audit"]
    held_support = notes["held"]["extra"]["held68_scope"]
    proof = notes["null"]["extra"]
    clamp = proof["null_clamp"]
    points = m.loc[m.screen.eq("macro")&m.seed.isin(("mean","fixed"))].set_index(
                    ["scope","support","metric","model"],verify_integrity=True)
    champ = c.loc[c.screen.eq("macro")&c.model.eq("Champion")].set_index(
                   ["scope","support","metric","baseline"],verify_integrity=True)
    g = m.loc[m.scope.eq("GDSC1")&m.support.eq("all")&m.screen.eq("macro")&m.model.eq("Champion")
              &m.metric.eq("ndcg_at_5")&m.seed.ne("mean")]
    assert g.seed.tolist()==["11711","11729","11743"]
    gs = points.loc[("GDSC1","all","ndcg_at_5","Champion")]
    assert round(gs.seed_sd,4)==.0407
    gdsc = ", ".join(f"{k} {v:.6f}" for k,v in zip(g.seed,g.value))
    ds = ci(champ.loc[("Drug-Cold","all","ndcg_at_5","H2-v2")])
    es = ci(champ.loc[("GDSC1","all","ndcg_at_5","H2-v3-Combined")])
    blocks = [
        "질문 · H3-v2 동결 예측이 동일 평가 행의 H2-v2, E1, E5와 비교해 어떤 성적을 내는가?",
        "바뀐 것 · 동결 예측만 사용해 직접 짝 비교, 980-family 교차, 성분 복원과 영점 민감도를 계산했다.",
        f"주요 구간 · Champion NDCG@5는 Drug-Cold 전체에서 H2-v2 대비 {ds}, GDSC1에서 E5 Combined 대비 {es}였다.",
        "판정 · **NO-GO within the frozen boundary-1 search budget.**",
        "Paper B · 동일 행 직접 대조와 독립성 민감도를 기존 Track B 주장에 연결한다. 외부 패널은 기존 노출을 명시한다.",
        "# H3-v2 마감 감사 및 Paper B 연결",
        "## 기계 재현",
        table(["원표","동일 키 수","최대 절대오차","C3 Slurm 잡"],
              [(k,v["comparison"]["rows"],f'{v["comparison"]["max_abs_diff"]:.17g}',v["slurm"]["id"])
               for k,v in gates.items()]),
        "전체 키, 분모, 결측 패턴과 수치를 확인했다. 원본 H3-v2 및 E5의 score_passes는 각각 1이다.",
        "## A1 · 동일 행 직접 비교",
        "Drug-Cold의 670,830행·1,186 parents와 외부 데이터의 row ID, source, cell, parent, endpoint, lineage를 1:1 대조했다. "
        "LICOB 성인 공여자 집계는 2,888행, LICOB18은 같은 집계에서 조건을 충족하는 684행이다. "
        "진리값 최대 절대차는 0.0이다. LINCS 지원도 동일했다. 외부 패널은 기노출 사후 분석이다.",
        "Native NDCG는 시료 내 min–max gain을 사용한다. Native interaction ρ는 관측 셀×약물 격자에서 두 주효과를 제거한 뒤 "
        "약물별 순위 상관을 평균한다. 화면 내 폴드와 전체 화면을 균등 평균한다. 짝 CI는 2,000회 공유 Bayesian cluster draw이며 "
        "약물 지표는 transitive scaffold, 검색 지표는 global lineage 단위다. 세 seed의 단위별 점수를 평균해 대비했고, "
        "표본 SD는 개별 seed 점수에서 계산했다. Cchem-int 상호작용에는 E1의 등록 개인화 성분 rc를 사용했다.",
        "### Champion 짝 NDCG@5와 seed SD",
        table(["패널","H2-v2","E1 Cchem-int","E5 P3B-Shared","E5 Combined †/‡","Champion seed SD"],[
            (name(s,u),*(ci(champ.loc[(s,u,"ndcg_at_5",r)]) for r in BASES),
             f'{points.loc[(s,u,"ndcg_at_5","Champion"),"seed_sd"]:.6f}') for s,u in PANELS]),
        CONTRACT,
        f"GDSC1 전체 Champion NDCG@5는 {gdsc}, 평균 {gs.value:.6f}, 표본 SD {gs.seed_sd:.6f}다.",
        "Drug-Cold 전체에서 Champion의 H2-v2 대비 약물 내 ρ 차이는 "
        f'{ci(champ.loc[("Drug-Cold","all","spearman","H2-v2")])}, interaction ρ 차이는 '
        f'{ci(champ.loc[("Drug-Cold","all","interaction_spearman","H2-v2")])}였다. NDCG@5는 LINCS724에서 낮고 '
        "Unobs462에서 높았다. 같은 전체 패널에서 Cchem-int 대비 약물 내 ρ와 interaction ρ는 상승하고 NDCG@5는 하락했다. "
        "P3B-Shared 대비에도 interaction ρ와 NDCG@5의 방향이 달랐다.",
        "GDSC1 NDCG@5에서는 H2-v2와 Cchem-int 대비 개선, P3B-Shared와의 0 포함, E5 Combined 대비 열세가 관측됐다. "
        "HCC 외부의 검색 점수는 LIMORE와 LICOB18에서 네 비교군 대비 높았다. LICOB 전체의 H2-v2 대비 구간은 0을 포함했다. "
        "네 외부 패널의 LINCS 미관측 NDCG@5는 네 직접 비교군 모두보다 낮았으며 각 짝 구간도 0 아래였다. "
        "검색 성적은 약물 내 개인화 성적과 별도로 해석한다.",
        "### 동일 행 성적표",
        "비교군은 동결 단일 예측이며 H3 점수는 세 seed의 평균이다. Champion은 Drug-Cold 및 외부에서 hybrid와 같다. "
        "예측값을 평균한 Champion-ensemble은 TSV에 기술 통계로 기록했다.",
    ]
    for metric,label in AXES.items():
        r = [a+(" †/‡" if a in ("H2-v3-Combined","P3B-InScreen") else "") for a in ARMS]
        blocks.extend([f"#### {label}",table(["패널","parents",*r],[
            (name(s,u),int(points.loc[(s,u,metric,"Champion"),"parents"]),
             *(f'{points.loc[(s,u,metric,a),"value"]:.6f}' if (s,u,metric,a) in points.index else "—" for a in ARMS))
            for s,u in PANELS]),CONTRACT+" 외부 InScreen 동결 arm은 미등록이다."])
    blocks.extend([
        "### Held-Screen/core68 · 전용 평가식",
        f"CTRPv2/GDSC2/PRISM20Q2의 core68 등록 패널에 실제 관측된 45,428행·{held_support['observed_parents']} parents에서 각 훈련 분할의 동결 H3 보관 H2 커널을 기준으로 비교했다. "
        "H2-v2·E1·E5의 동결 예측은 이 core68 분할 계약을 충족하는 대조군이 없다. "
        "Core68은 lineage×parent로 합친 뒤 (rank−1)/(n−1) gain과 예측 동점의 기대 gain을 사용한다. "
        "상호작용은 각 fold의 직교 잔차를 모아 Spearman ρ를 계산한다. 2,000회 공유 lineage multinomial 추출을 사용했다. "
        "상호작용 CI는 원 패널의 동결 잔차를 추출별 재순위화한 사후 확장이다.",
        table(["스크린",*AXES.values(),"Champion NDCG seed SD"],[
            (screen,*(ci(held.loc[held.model.eq("Champion")&held.screen.eq(screen)&held.metric.eq(k)].iloc[0]) for k in AXES),
             f'{held.loc[held.model.eq("Champion")&held.screen.eq(screen)&held.metric.eq("ndcg_at_5"),"delta_seed_sd"].item():.6f}')
            for screen in ("macro","CTRPv2","GDSC2","PRISM20Q2")]),
        "[native 지표 및 seed](tables/H3V2_VS_H2V2_E5_METRICS.tsv) · "
        "[native 짝 구간](tables/H3V2_VS_H2V2_E5_PAIRED_95CI.tsv) · "
        "[core68 전용 짝 구간](tables/H3V2_HELD68_VS_H2_PAIRED_95CI.tsv)",
        "## A2 · 980-family 독립성",
        "E3 로스터의 1,014 cell ID, 991 lineage, 980 family를 정확 ID로 연결했다. Drug-Cold 분할에서 세포 family 중복은 "
        "scaffold 독립 설계에 따른다. Cell-Cold와 Held68의 교차는 강화된 family 수준 독립성의 한계다.",
        table(["split","train/query cells","991 교차","980 교차","교차 행","교차 비율"],[
            (r.split,f"{r.n_train_cells}/{r.n_query_cells}",r.lineage_overlap,r.family_overlap_count,
             f"{r.query_rows_crossing:,}/{r.query_rows:,}",f"{r.query_rows_crossing_pct:.6f}%") for r in overlap.itertuples()]),
    ])
    filt = sens.loc[sens.analysis.eq("exclude_train_family_rows")&sens.quantity.eq("contrast")]
    boot = sens.loc[sens.analysis.eq("family_cluster_bootstrap")]
    drop_rows, boot_rows = [], []
    for scope in ("Cell-Cold","Held68"):
        view = filt.loc[filt.scope.eq(scope)]
        q = view.loc[view.screen.eq("macro")&view.model.eq("Champion")&view.metric.eq("ndcg_at_5")].iloc[0]
        flips = view.loc[view.sign_changed.eq(True)&view.fold.eq("all")]
        drop_rows.append((scope,f"{int(q.rows_before):,} → {int(q.rows_after):,}",
                          f"{q.before_value:+.6f} → {q.after_value:+.6f}",f"{view.change.abs().max():.6f}",len(flips)))
    for scope in ("Drug-Cold","Cell-Cold","GDSC1"):
        v = boot.loc[boot.scope.eq(scope)]
        dist = max((v.lo95_980-v.lo95).abs().max(),(v.hi95_980-v.hi95).abs().max())
        flips = int(np.sum(sign(v.lo95,v.hi95)!=sign(v.lo95_980,v.hi95_980)))
        boot_rows.append((scope,int(v.clusters.iloc[0]),int(v.clusters_980.iloc[0]),f"{dist:.6f}",flips))
    blocks.extend([
        table(["패널","제외 전후 행","Champion−H2 NDCG@5","모든 짝 최대 |변화|","macro·화면 부호 변경"],drop_rows),
        "교차 query 행을 제거하고 같은 동결 예측을 재집계했다. 이 사후 민감도 검사는 원래 분할의 강화 family 독립성을 보증하지 않는다. "
        "Drug-Cold 및 외부 등록 승격 입력은 유지된다.",
        "행 제외 후 Champion의 전체 NDCG@5 대비 방향은 두 패널에서 유지됐다. 집계 수준의 부호 변화는 "
        "Cell-Cold macro GNN의 Pearson −0.000102→+0.000195 및 Held68 GDSC2 GBDT의 Precision@5 "
        "+0.000561→−0.000095였다.",
        table(["NDCG@5 재표본","원 lineage 수","강화 family 수","CI 끝점 최대 이동","CI 0 포함·방향 변화"],boot_rows),
        "GDSC1의 단위 수는 NDCG@5 적격 셀 기준이다. GDSC1 attention−H2-v2의 NDCG@5 차이는 −0.013311로 유지됐고 "
        "95% CI는 [−0.026759, −0.000011]에서 [−0.027098, +0.000760]으로 이동해 0을 포함했다.",
        "[교차 family 및 cell ID](tables/H3V2_FAMILY980_SPLIT_OVERLAP.tsv) · "
        "[행 제외와 강화 클러스터 민감도](tables/H3V2_FAMILY980_SENSITIVITY.tsv)",
        "## A3 · 보조 성분과 영점 평가",
        "parts=True는 "+", ".join(f'{r["split"]}/{r["family"]}' for r in proof["inventory"])+" 네 설정이다. "
        "Drug-Cold Champion 0/3, 외부 Champion 0/1이다. 불균형 7행 예제에서 "
        f"max|y − (baseline + c + residual)| = {proof['toy']['max_abs_reconstruction_error']:.12g}다. "
        "불균형 관측에서는 고정 약물 baseline을 뺀 뒤 다시 공동 추정한 약물 주효과가 남는다. "
        "보조 타깃 함수가 이 추가 약물 성분을 반환하지 않아 합이 성립하지 않는다.",
    ])
    q = nu.loc[nu.analysis.eq("clamp_model")&nu.study.eq("H3-v2")&nu.support.eq("all")
               &nu.screen.eq("macro")&nu.model.eq("drug_mean")].set_index("scope")
    blocks.extend([
        table(["패널","drug_mean 상호작용 ρ","clamp 후","최대 절대 예측 잔차"],[
            (scope,f'{q.loc[scope,"interaction_spearman"]:+.8f}',f'{q.loc[scope,"interaction_spearman_clamped"]:+.8f}',
             f'{q.loc[scope,"max_abs_prediction_residual"]:.4g}') for scope in ("Drug-Cold","GDSC1","LICOB")]),
        "완전 및 실제 DRP 결측 격자에서 상수·약물·세포·가법 합성 표면을 평가했다. "
        "두 방향 중심화 후 미세한 부동소수점 잔차가 순위로 증폭돼 약물 평균에도 상호작용 ρ가 나타난다. "
        f"LICOB 원시 예측 dtype은 {clamp['drug_mean_LICOB_raw_dtype']}이며 원시 약물 내 최대 범위는 "
        f"{clamp['drug_mean_LICOB_max_raw_drug_spread']:.4g}였다. 보존된 공여자 집계 후 범위는 "
        f"{clamp['drug_mean_LICOB_max_donor_drug_spread']:.4g}, 원 약물값과의 최대 차이는 "
        f"{clamp['drug_mean_LICOB_max_donor_aggregation_deviation']:.4g}였다. "
        "float32 공여자 평균에서 생긴 1 ULP 규모의 차이가 기존 실측 마스크의 중심화·순위 계산을 거친다. "
        "아래 진단은 같은 공여자·약물 지지집합과 실측값에 정밀도만 바꾼다.",
        table(["LICOB 공여자 집계 진단","예측 dtype","약물 내 최대 범위","원 상호작용 ρ","clamp 후 ρ"],[
            (z["aggregation"],z["prediction_dtype"],f'{z["max_within_drug_prediction_spread"]:.6g}',
             f'{z["interaction_spearman"]:+.8f}',f'{z["interaction_spearman_clamped"]:+.8f}')
             for z in clamp["drug_mean_LICOB_aggregation_comparison"]]),
        f"1e-9 clamp의 비퇴화 동결 모델 짝 최대 변화는 Spearman {clamp['max_spearman_contrast_change_excluding_analytic_null']:.8g}, "
        f"Pearson {clamp['max_pearson_contrast_change_excluding_analytic_null']:.8g}였다. drug_mean을 포함한 짝의 최대 Spearman 변화는 "
        f"{clamp['max_spearman_contrast_change_including_drug_mean']:.8g}였다. 원본 지표·판정은 보존했다.",
        "[보조 성분 및 null surface](tables/H3V2_PARTS_AND_NULL_AUDIT.tsv)",
        "## A4 · Paper B 해석",
        "NO-GO within the frozen boundary-1 search budget는 사전 고정한 후보, 데이터, 학습 레시피와 승격 규칙의 결과다. "
        f"{limits['frozen_from_boundary_1']}/75 arm이 boundary-1에서 선택됐고, "
        f"{limits['arms_near_iteration_cap_94_percent']}/75는 적어도 한 inner fold에서 설정 iteration 상한의 94% 이상에 도달했다. "
        f"{limits['unprobed_neighbors']}개 이웃은 미탐색이다. 해석은 실제 평가한 탐색 예산과 패널에 한정한다.",
        "다섯 패밀리는 완성된 모델링 레시피다. GBDT의 행 단위 L2/Huber와 시료 단위 LambdaRank, 신경망의 Huber·screen×drug Pearson·"
        "screen×cell ListNet과 동적 손실 비중, 하이브리드의 커널 잔차 보정은 손실 및 assay별 기여도를 다르게 만든다. "
        "Cell-Cold FiLM의 inner 선택은 약물 내 ρ 0.45, 상호작용 ρ 0.30, NDCG@5 0.25의 선형 합이며 축별 보호 하한이 없었다. "
        "이 규칙은 순위 성능 개선과 추천축 손실의 교환을 허용한다.",
        "Track B의 등록 재현, 상위 커널 계보, 화면별 추천 손실, 강화 계통과 E1–E4–E5 대비는 기존 원장에 연결했다. "
        "Paper B는 각 등록 판정과 E3의 독립 검정을 보존하고, known-screen 및 shared 적용과 native 및 core68 평가식을 각 결과에 명시한다.",
        "[Track B 주장 원장](PAPER_B_POSTHOC_AUDIT_AND_CLAIM_LEDGER.md) · "
        "[E1–E4–E5 직접 비교](E1_E4_E5_PAIRED_AUDIT.md) · "
        "[마감 영수증](tables/h3v2_closing_audit_receipt.json) · "
        "[영수증 SHA256](tables/h3v2_closing_audit_receipt.sha256)",
    ])
    return "\n\n".join(blocks)+"\n"


def protected_inputs(notes,gates):
    for group in (*notes.values(),*gates.values()):
        for name,expected in group["input_sha256"].items():
            if name in INPUTS and INPUTS[name]!=expected:
                raise ValueError(f"Conflicting input hashes: {name}")
            INPUTS[name] = expected
    for name,digest in INPUTS.items():
        if Path(name).name in ("score_receipt.json","verification.json","RESULTS.md","consumed.json"):
            assert sha256(Path(name))==digest,name
    for item in OLD_STUDIES:
        pin(HOME/item)
    prior = read(HOME/"tables/e5_trackb_machine_replay_receipt.json")
    assert prior["max_abs_diff"]==0 and prior["status"]=="PASS"


def publish():
    require_slurm()
    gates = gate()
    notes = {p:read(STAGE/f"{p}_done.json") for p in ("native","held","null")}
    protected_inputs(notes,gates)
    tmp = {}
    for note in notes.values():
        for f,h in note["output_sha256"].items():
            path = STAGE/f
            assert sha256(path)==h
            tmp[f] = pd.read_csv(path,sep="\t",float_precision="round_trip")
            assert len(tmp[f])==note["rows"][f]
    table_m = tmp["H3V2_VS_H2V2_E5_METRICS.tsv"]
    table_m["seed"] = table_m["seed"].astype(str)
    tmp["H3V2_FAMILY980_SENSITIVITY.tsv"] = pd.concat([tmp["family980_native.tsv"],tmp["family980_held.tsv"]],ignore_index=True)
    tables = {k:tmp[k] for k in COUNTS}
    for k,n in COUNTS.items():
        assert len(tables[k]) == n if n is not None else len(tables[k])>0
    for k,n in {"H3V2_VS_H2V2_E5_METRICS.tsv":6552,
                "H3V2_VS_H2V2_E5_PAIRED_95CI.tsv":8064,
                "H3V2_HELD68_VS_H2_PAIRED_95CI.tsv":144}.items():
        assert len(tables[k])==n,(k,len(tables[k]),n)
    content = make_bridge(tables,gates,notes)
    for name,df in tables.items():
        if name=="H3V2_FAMILY980_SENSITIVITY.tsv":
            df.to_csv(STAGE/name,sep="\t",index=False,na_rep="NA",float_format="%.17g")
        (HOME/"tables"/name).write_bytes((STAGE/name).read_bytes())
    (HOME/REPORT).write_text(content,encoding="utf-8")
    outputs = {s:sha256(HOME/s) for s in (REPORT,*("tables/"+s for s in tables))}
    scripts = {str(p.relative_to(HOME)):sha256(p) for p in sorted((HOME/"analysis").glob("h3_closing_*"))
               if p.suffix in (".py",".sbatch")}
    log_paths = sorted((HOME/"logs").glob("h3v2-closing-*.log"))
    logs = {str(p.relative_to(HOME)):sha256(p) for p in log_paths if "publish" not in p.name}
    all_inputs = dict(sorted(INPUTS.items()))
    all_inputs["/home/gsun/work/vc_liver/H3V2_CLOSING_AUDIT_SPEC.md"]=SPEC_SHA
    manifest = {"status":"PASS","scope":"A1–A4 H3-v2 closing audit",
                "finished_utc":datetime.now(timezone.utc).isoformat(),"new_outer_score_passes":0,
                "retrained_models":0,"refits":0,"new_grid_trials":0,"existing_audits_rerun":0,
                "git_reference_commits":ANCHORS,"original_score_passes":{"H3-v2":1,"E5":1},
                "machine_replay_threshold":1e-9,
                "machine_replay":{k:{"comparison":v["comparison"],"slurm":v["slurm"]} for k,v in gates.items()},
                "audit_slurm":{k:v["slurm"] for k,v in notes.items()}|{"publication":job_meta()},
                "native_draws":2000,"native_bootstrap_seed":318743,"core68_bootstrap_seed":14639,
                "input_sha256":all_inputs,"analysis_sha256":scripts,"output_sha256":outputs,"log_sha256":logs,
                "runtime":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},
                "table_rows":{k:len(t) for k,t in tables.items()},
                "known_screen_term":"Drug-Cold Combined/InScreen. External Combined uses shared inference.",
                "held68_interaction_bootstrap":"Rerank frozen pooled orthogonal residuals with shared lineage multiplicities.",
                "detached_self_sha256":"tables/h3v2_closing_audit_receipt.sha256",
                "results":{key:note["extra"] for key,note in notes.items()}}
    draft = STAGE / f"closing-receipt-{job_meta()['id']}.json"
    publish_json(draft,manifest)
    draft.replace(HOME/RECEIPT)
    self_hash = sha256(HOME/RECEIPT)
    detached = STAGE / f"closing-receipt-{job_meta()['id']}.sha256"
    detached.write_text(f"{self_hash}  {RECEIPT}\n")
    detached.replace(HOME/(RECEIPT.removesuffix(".json")+".sha256"))
    print("AUDIT RECEIPT",self_hash,flush=True)
    print("TABLES",manifest["table_rows"],flush=True)


if __name__=="__main__":
    publish()
