# R9 — 환자 지지도를 반영한 희귀 세포 상태 인식 (donor-supported rare cell-state recognition)

**상태:** 실행 완료 보고는 [RESULTS.md](RESULTS.md), 주장 범위는 [CLAIMS_AND_LIMITATIONS.md](CLAIMS_AND_LIMITATIONS.md), cycle 기록은 [DECISIONS.md](DECISIONS.md). 쉬운 요약은 [BEGINNER_SUMMARY.md](BEGINNER_SUMMARY.md).

## 무엇을 입력받아 무엇을 예측하나

- **학습 입력:** 학습 환자(source donor) 세포의 RNA count, 저자 세포 유형 주석, 환자 ID(가중치·분할·지지도 계산에만 사용).
- **추론 입력:** 학습에 쓰지 않은 환자의 세포 RNA count(유전자 이름 포함)만.
- **출력:** 알려진 label 확률, 상위 유형(Immune/Stromal/Parenchymal) fallback, 판단 보류 여부.
- **정답:** 기존 atlas 저자 주석. wet-lab·기능 검증 정답이 아니다.

## 사용 가능한 정보와 범위 조정 (A0)

GSE149614 배포 주석은 6개 대분류뿐이며 전문가 세부 아형 이름이 없다. 대분류에는 학습 비율 ≤2% label이 없다. 따라서 (§4 규칙) 주장은 다음으로 좁혔다.

- **Tier 1(주):** 대분류 6개. 희귀 질문은 **인공 희소화**(N×D, sparse 2×2)로만 답한다.
- **Tier 2(탐색):** 저자 `res.3` 군집 번호를 lineage 안에서 맞히는 과제(Myeloid 14, B 5). 자연 희귀·다환자/단일환자 군집이 있으나 군집 번호 재현일 뿐이다.
- **외부(A4):** GSE151530 HCC 25명. 5개 대분류만 대응(Hepatocyte 정답 없음). 이 코호트는 이전 `phase2-v2`에서 대분류가 채점된 적이 있어 blind cohort가 아니다.

## 재현

서버 worktree `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state` (캐시·실행·환경은 git 무시 `assets/`).

```bash
bash scripts/r9/install_core_env.sh        # assets/envs/r9core (Slurm cpu job으로 실행)
bash scripts/r9/install_ext_env.sh         # assets/envs/r9ext (scPoli/ProtoCloud)
S=scripts/r9/slurm/run_py.sh
sbatch -p cpu -J r9_a0inv $S scripts/r9/a0_inventory.py
sbatch -p cpu -J r9_repr -a 0-10 $S scripts/r9/a1_repr.py task
sbatch -p cpu -J r9_a1nxd -a 0-29 $S scripts/r9/a1_nxd.py a1_nxd_v1 ; $S scripts/r9/a1_summarize.py a1_nxd_v1
sbatch -p cpu -J r9_a2 -a 0-179 $S scripts/r9/a2_run.py configs/r9/a2_main.json ; $S scripts/r9/a2_eval.py a2_main_v1 ...
sbatch -p gpu --gres=gpu:1 -J r9_ext -a ... scripts/r9/slurm/run_ext.sh configs/r9/a2_ext.json scpoli|protocloud
sbatch -p cpu -J r9_a3 -a ... $S scripts/r9/a3_calib.py configs/r9/a3_calib.json|a3_unknown.json ; $S scripts/r9/a3_eval.py ...
sbatch -p cpu -J r9_a4 -a ... $S scripts/r9/a4_external.py configs/r9/a4_freeze.json ; $S scripts/r9/a4_eval.py ...
```

실제 실행 ID·설정·시간은 [run_ledger.tsv](run_ledger.tsv).

## 산출물

| 파일 | 내용 |
|---|---|
| `label_inventory.tsv`, `donor_class_counts.tsv`, `cohort_overlap.tsv`, `rare_definition_by_fold.tsv`, `eligible_tasks.md` | A0 |
| `n_by_d_stress.tsv`, `n_by_d_effects.tsv`, `n_by_d_per_donor.tsv`, `figures/fig1_n_by_d.png` | A1 |
| `BASELINES.tsv`, `a2_*.tsv`, `prototype_support.tsv`, `figures/fig2_*`, `figures/fig3_*` | A2 |
| `risk_coverage.tsv`, `a3_*.tsv` | A3 |
| `a4_*.tsv`, `figures/fig4_*` | A4 |

## 기획서

비공개 기획서는 Git에 넣지 않는다. 크기 29,750 bytes(356줄), sha256 `6b58e37e1ec0b7cea4b4e503902af4ba39db6e723be731eb6cae4c8cabcbb118`.
