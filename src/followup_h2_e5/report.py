"""Render the E5 report from sealed aggregate artifacts without reading outcomes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


C1 = "C1"
V2 = "H2-v2"
E4 = "H2-LOSO-Full"
COMBINED = "H2-v3-Combined"
P1 = "P1-Minimax"
P2 = "P2-Decoupled"
SHARED = "P3B-Shared"
IN_SCREEN = "P3B-InScreen"
POOLED = "Sel-Pooled"
LOSO = "Sel-LOSO"
COHORTS = ("CrossScreen", "DRP", "GDSC1", "LIMORE", "LICOB", "LICOB18")
LABELS = {"CrossScreen": "Track 1 · 미접촉 스크린", "DRP": "Track 2 · DRP"}
EXTERNAL = COHORTS[2:]
PILLARS = (P1, "P1-RelMinimax", P2, "P3A-ScreenDebiased", SHARED, IN_SCREEN,
           "P3C-EqualScreenWeight", COMBINED)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def support(metric):
    return "lincs_observed" if metric in ("spearman", "interaction_spearman") else "all"


def effect_text(point):
    if point is None:
        return "—"
    value, lo, hi = point
    return f"{value:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def from_gate(x):
    return tuple(x[k] for k in ("delta", "lo95", "hi95"))


class Tables:
    def __init__(self, score):
        self.score = score
        self.points = {}
        for row in rows(score/"E5_METRICS.tsv"):
            k = tuple(row[x] for x in ("cohort", "support", "metric", "arm", "screen", "fold"))
            if k in self.points:
                raise ValueError(f"Duplicate metric: {k}")
            self.points[k] = float(row["value"])
        self.effects = {}
        for row in rows(score/"E5_PAIRED_95CI.tsv"):
            k = tuple(row[x] for x in ("cohort", "support", "metric", "contrast", "screen", "fold"))
            if k in self.effects:
                raise ValueError(f"Duplicate paired contrast: {k}")
            if int(row["draws"]) != 2000:
                raise ValueError(f"Different paired resampling count: {k}")
            self.effects[k] = tuple(float(row[x]) for x in ("delta", "lo95", "hi95"))

    def point(self, cohort, metric, arm, *, scope=None):
        return self.points[cohort, scope or support(metric), metric, arm, "macro", "all"]

    def effect(self, cohort, metric, arm, ref, *, scope=None, screen="macro", fold="all"):
        if arm == ref:
            return None
        prefix = cohort, scope or support(metric), metric
        v = self.effects.get((*prefix, f"{arm} - {ref}", screen, fold))
        if v is not None:
            return v
        v = self.effects.get((*prefix, f"{ref} - {arm}", screen, fold))
        return (-v[0], -v[2], -v[1]) if v is not None else None

    def required(self, cohort, metric, arm, ref):
        v = self.effect(cohort, metric, arm, ref)
        if v is None:
            raise ValueError(f"Missing {cohort}/{metric}/{arm}/{ref}")
        return effect_text(v)

    def cohort_table(self, cohort):
        arms = (C1, "H2-v1", V2, E4, *PILLARS)
        if cohort != "DRP":
            arms = tuple(x for x in arms if x != IN_SCREEN)
        if cohort == "CrossScreen":
            arms = (*arms[:4], POOLED, LOSO, *arms[4:])
        if cohort == "DRP":
            arms = (*arms, "C2", "C3")
        out = [
            "| Arm | 관측 ρ | 관측 상호작용 ρ | 전체 NDCG@5 | Precision@5 | 고유 Top-5 | Δρ vs v2 | Δρ vs E4 | ΔNDCG vs v2 | ΔNDCG vs E4 |",
            "|---|---:|---:|---:|---:|---:|---|---|---|---|",
        ]
        keys = ("spearman", "interaction_spearman", "ndcg_at_5", "precision_at_5", "unique_top5_sets")
        for arm in arms:
            p = [f"{self.point(cohort,k,arm):.4f}" for k in keys]
            p[-1] = f"{self.point(cohort,keys[-1],arm):.2f}"
            ci = [effect_text(self.effect(cohort,metric,arm,ref))
                  for metric in ("spearman","ndcg_at_5") for ref in (V2,E4)]
            out.append("| " + " | ".join([arm,*p,*ci]) + " |")
        return "\n".join(out)


def anchor_name(rec):
    key = rec["intercept_candidate"]
    x = rec["intercept"]
    if "alpha" in x:
        chem = "/".join(map(str, x["chem"]))
        lincs = "/".join(map(str, x["lincs"]))
        return f"{key} · α={x['alpha']:g}, γ={x['gamma']:g} · {chem} + {lincs}"
    return key


def selection_table(selection):
    out = ["| 역할 | Arm | λ | w | shape/base | τ | 절편 · 선택 방식 | 내부 ρ Pooled / LOSO |",
           "|---|---|---:|---:|---|---:|---|---:|"]
    for role, policies in sorted(selection["policies"].items()):
        for arm, item in sorted(policies.items()):
            out.append(f"| {role} | {arm} | {item['penalty']:g} | {item['weight']:g} | "
                       f"{item['shape']}/{item['base']} | {item['tau']:g} | "
                       f"{anchor_name(item)} · {item['anchor_selection']} | "
                       f"{item['pooled']:.4f} / {item['loso']:.4f} |")
    return "\n".join(out)


def pareto_and_tau(score, selection):
    grid = list(rows(score/"E5_TAU_AND_PARETO_DIAGNOSTIC.tsv"))
    text = ["| 역할 | P1 Pareto 점 | P3 Pareto 점 | 절대 후회 λ | 상대 후회 λ | P3-Best |",
            "|---|---:|---:|---:|---:|---|"]
    for role, items in sorted(selection["policies"].items()):
        group = [r for r in grid if r["kind"] == "internal_pooled_loso" and r["outer"] == role]
        front = {key: sum(r["comparison"] == key and r["pareto"] == "True" for r in group)
                 for key in ("P1", "P3-Best")}
        final = items[COMBINED]
        text.append(f"| {role} | {front['P1']} | {front['P3-Best']} | {items[P1]['penalty']:g} | "
                    f"{items['P1-RelMinimax']['penalty']:g} | {final['family']} · λ={final['penalty']:g} · τ={final['tau']:g} |")
    text.extend(["", "아래는 선택된 P3B의 기저·형태·w·λ를 고정한 관측 LINCS 지원의 τ 곡선이다.", "",
                 "| 코호트 | 추론 | τ=0 | .25 | .5 | 1 | 2 | 4 |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"])
    for cohort in COHORTS:
        for mode in (("shared", "in_screen") if cohort == "DRP" else ("shared",)):
            block = {float(x["tau"]): float(x["spearman"]) for x in grid if x["kind"] == "tau_curve"
                     and x["support"] == "lincs_observed" and x["cohort"] == cohort and x["mode"] == mode}
            if set(block) != {0., .25, .5, 1., 2., 4.}:
                raise ValueError(f"Incomplete tau sweep: {cohort}/{mode}")
            text.append("| " + " | ".join([cohort, mode, *(f"{block[v]:.4f}" for v in sorted(block))]) + " |")
    return "\n".join(text)


def verify_chain(selected, tests, score):
    select_file = selected/"e5_selection_receipt.json"
    test_file = tests/"test_receipt.json"
    selection, test, receipt = (read_json(f) for f in (select_file, test_file, score/"score_receipt.json"))
    if (receipt["score_passes"] != 1 or test["status"] != "PASS"
        or sha256(select_file) != test["selection_sha256"] or test["selection_sha256"] != receipt["selection_sha256"]
        or sha256(test_file) != receipt["test_sha256"] or selection["external_cohort_access"] != 0
        or not selection["locked_at_utc"] < test["passed_at_utc"] < receipt["started_at_utc"]
        or not receipt["metrics_sealed_at_utc"] <= receipt["diagnostic_started_at_utc"] <= receipt["finished_at_utc"]):
        raise ValueError("Selection, tests and score do not share a valid sealed timeline")
    for name, digest in receipt["output_sha256"].items():
        if sha256(score/name) != digest:
            raise ValueError(f"Changed score artifact: {name}")
    return selection, test, receipt


def render(selected, tests, score):
    selection, test, receipt = verify_chain(selected, tests, score)
    table = Tables(score)
    decisions = receipt["comparison"]
    verdict = decisions["decision"]
    recommendations = {
        "GO-V3-PARETO-UPGRADE": "평가된 지원에서 H2-v3-Combined를 기본 정책으로 채택한다.",
        "PARTIAL-PILLAR-WIN": "아래 표의 통과 기둥을 해당 평가 지원에서 채택하고, 나머지 용도는 E4 프로파일을 유지한다.",
        "NO-GO-KEEP-E4-DUAL": "drp_in_panel은 H2-v2, 제한된 cross_platform 검색은 E4를 유지한다.",
    }
    recommendation = recommendations[verdict]
    cross = table.required("CrossScreen", "spearman", COMBINED, LOSO)
    drp = table.required("DRP", "spearman", COMBINED, V2)
    gdsc = table.required("GDSC1", "ndcg_at_5", COMBINED, V2)
    p1 = selection["policies"]["T3/3"][P1]
    combined = selection["policies"]["T3/3"][COMBINED]
    min_drug = min(x["minimum_eigenvalue"] for row in test["selected_grams"].values() for x in row["drug"].values())
    min_task = min(x["active_minimum_eigenvalue"] for row in test["selected_grams"].values() for x in row["task"].values())
    tau_error = max(x for row in test["tau0_max_abs_diffs"].values() for x in row.values())
    parts = [
        "질문 · 내부 세 스크린만으로 기존 DRP 개인화와 E4 교차 플랫폼 검색을 한 모델에서 보존할 수 있는가?",
        f"실제 바뀐 것 · 전체 DRP P1은 λ={p1['penalty']:g}, 통합 모델은 {combined['family']}와 λ={combined['penalty']:g}, τ={combined['tau']:g}를 사용했다.",
        f"주 비교 결과·구간 · 통합 모델의 Track 1 Δρ vs E4 {cross}, DRP Δρ vs H2-v2 {drp}, GDSC1 ΔNDCG@5 vs H2-v2 {gdsc}.",
        f"판정 · {verdict}.",
        f"다음 결정 · {recommendation}",
        "",
        "# E5 · H2 교차 플랫폼 고도화",
        "",
        "## 지원과 비교 범위",
        "",
        "Track 1은 해당 외부 스크린을 각 선택에서 격리했다. Track 2는 E4의 outer 골격과 행을 재사용했다. "
        "두 트랙은 각각 670,830행, 약물 1,186개, LINCS 관측 약물 724개다. 표의 순위는 LINCS 관측, 검색은 전체 지원이다. "
        "Track 3의 네 코호트는 모두 사전 노출된 외부 전이 점검이며, 방법 비교의 주 근거는 Track 1과 2다. "
        "Track 1의 H2-v2는 역사 설정을 해당 outer fit에 재적합한 참고치다. Track 1 gate는 같은 분할의 Sel-Pooled와 E4를 쓴다. "
        "[사전 잠금](PLAN_LOCK.md) · [채점 영수증](score_receipt.json)",
        "",
        "## 내부 선별",
        "",
        "P1은 Pooled 및 LOSO의 최고 관측 ρ와 후보의 차이 중 큰 값을 최소화했다. 상대 규칙은 각 최고치로 후회를 나눴다. "
        "P2는 P1과 동일한 개인화를 유지하고, LOSO 절편 후보 18개를 평가했다. P3A는 fit 스크린 안에서 약물 및 세포 오프셋을 제거하고 약물별 분산을 맞췄다. "
        "P3B는 공유와 스크린 성분을 분리했고, P3C는 전체 가중치를 보존하며 스크린별 유효 질량을 균등하게 바꿨다. "
        "통합 모델은 P3 전체의 내부 후회에서 고른 구조와 LOSO 절편을 결합했다. [선택 영수증](e5_selection_receipt.json)",
        "",
        selection_table(selection),
        "",
    ]
    for cohort in COHORTS:
        labels = receipt["supports"][cohort]
        parts += [
            f"## {LABELS.get(cohort, 'Track 3 · ' + cohort)}", "",
            f"전체 {labels['all']['rows']:,}행 · {labels['all']['parents']:,}약물 · {labels['all']['cells']:,}세포 또는 공여자. "
            f"LINCS 관측 {labels['lincs_observed']['parents']:,}약물. Δ는 같은 지원의 짝 95% CI다.",
            "", table.cohort_table(cohort), "",
        ]
    parts += [
        "## 절편과 계층 커널 분해", "",
        "| 코호트 | P2−P1 ρ | P2−P1 NDCG@5 |", "|---|---|---|",
    ]
    for cohort in COHORTS:
        parts.append(f"| {cohort} | {table.required(cohort,'spearman',P2,P1)} | "
                     f"{table.required(cohort,'ndcg_at_5',P2,P1)} |")
    parts += [
        "",
        f"Track 2에서 P3B InScreen−Shared Δρ는 {table.required('DRP','spearman',IN_SCREEN,SHARED)}, "
        f"전체 ΔNDCG@5는 {table.required('DRP','ndcg_at_5',IN_SCREEN,SHARED)}였다. "
        "Track 1과 3의 미접촉 스크린에는 공유 예측을 사용했다.",
        "", "## 사후 Pareto 및 τ 진단", "",
        "Pareto 점은 두 내부 ρ가 모두 같거나 높고 하나 이상 높은 다른 점이 없는 구성이다. "
        "표는 각 역할과 후보 집합에서 같은 기준을 적용했다. τ 곡선은 공식 지표 봉인 뒤 같은 정답에서 산출했다. "
        "[전체 진단 표](E5_TAU_AND_PARETO_DIAGNOSTIC.tsv)", "",
        pareto_and_tau(score, selection), "",
        "## 판정 게이트", "",
        "| 트랙 | 통합 모델 판정 | 핵심 대비 |", "|---|---|---|",
        f"| 1 | {'통과' if decisions['track1']['pass'] else '미달'} | "
        f"Sel-Pooled Δρ {effect_text(from_gate(decisions['track1']['rho_vs_pooled']))}, "
        f"Sel-Pooled ΔNDCG {effect_text(from_gate(decisions['track1']['ndcg_vs_pooled']))}, E4 Δρ {cross} |",
        f"| 2 | {'통과' if decisions['track2']['pass'] else '미달'} | "
        f"v2 Δρ {drp}, v2 ΔNDCG {effect_text(from_gate(decisions['track2']['ndcg_vs_v2']))}, "
        f"{', '.join(a + ' Δρ ' + effect_text(from_gate(x)) for a,x in decisions['track2']['rho_vs_controls'].items())} |",
        f"| 3 | {'통과' if decisions['track3']['pass'] else '미달'} | "
        f"GDSC1 v2 ΔNDCG {gdsc}, v2 Δρ {effect_text(from_gate(decisions['track3']['gdsc1_rho_vs_v2']))}, "
        f"E4 Δρ {effect_text(from_gate(decisions['track3']['gdsc1_rho_vs_E4']))}, "
        f"LIMORE v2 Δρ {effect_text(from_gate(decisions['track3']['limore_rho_vs_v2']))} |",
        "",
    ]
    for win in decisions["partial_pillar_winners"]:
        parts.append(f"- {win['cohort']}의 {win['arm']}은 E4 대비 Δρ {effect_text(from_gate(win['rho_vs_E4']))}, "
                     f"ΔNDCG@5 {effect_text(from_gate(win['ndcg_vs_E4']))}로 부분 채택 기준을 통과했다.")
    parts += [
        "", recommendation, "", "## 실행 검증", "",
        f"구조 키를 감사한 역할은 {test['audited_splits']}개다. 약물 Gram과 활성 과제 Gram의 최소 고유값은 "
        f"각각 {min_drug:.6g}, {min_task:.6g}였다. τ=0 고정 fit 부분 격자 및 수치 검증 최대 오차는 {tau_error:.3g}였다. "
        "F00/F11/E4 원 예측, fit-only centering·scaling, screen 균등 가중 및 절편 불변식을 검증했다. "
        "하위 선택의 수치 벤치마크와 전체 탐색은 CPU에서 수행했다. "
        "[테스트 영수증](test_receipt.json) · [작업 원장](SLURM_JOBS.tsv)",
        "",
        f"선택 봉인 {receipt['selection_locked_at_utc']}, 테스트 {receipt['tests_passed_at_utc']}, "
        f"공식 채점 접근 {receipt['started_at_utc']} (UTC). TRAIN과 외부 코호트의 표지를 각각 한 번 읽었다. "
        "[전 지표](E5_METRICS.tsv) · [전 짝 95% 구간](E5_PAIRED_95CI.tsv) · [기둥 비교 표](E5_PILLAR_ABLATION.tsv)",
    ]
    return "\n".join(parts).rstrip()+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("selection", "tests", "score", "out"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    body = render(args.selection, args.tests, args.score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        file.write(body)
