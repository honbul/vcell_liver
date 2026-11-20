# P0-R9 사용 범위 판정 (plan §5.4 결정표 적용)

대상: 선정 모델 **LR_donor**, **Corrected**(정규화 정정 후 현재 모델), 기존 정책(source OOF 온도 + 2% 경험적 source 오류율 임계값) 그대로. 자료: GSE151530 HCC 25 donor — **반복 사용된 개발 진단 자료**이며 독립 외부 검증이 아니다. 판정 규칙은 `RUN_SPEC.md`(freeze `e34f466`)에서 결과 전에 고정했다. 수치 출처: `decision_mapping.tsv`, `coverage_error_summary.tsv`, `per_class_output_pooled.tsv`, `no_output_summary.tsv`.

## 1. 조건별 관측 (donor-macro, 95% donor bootstrap CI, 25 donor)

| 조건(source 학습 이름) | fine coverage | parent 비율 | 보류 | output coverage | 조건부 output error [CI] (유효 donor / donor×seed) | 합산 오류율(보조) | 무출력 행 |
|---|---|---|---|---|---|---|---|
| nat | 0.460 [0.380, 0.546] | 0.000 | 0.540 [0.454, 0.620] | 0.460 [0.380, 0.546] | 0.0025 [0.0002, 0.0061] (25/75) | 0.0010 (36/37,656) | 0/75 |
| sparse_B | 0.182 [0.155, 0.212] | 0.337 | 0.481 [0.402, 0.557] | 0.519 [0.443, 0.598] | 0.0086 [0.0037, 0.0144] (25/75) | 0.0062 (254/40,789) | 0/75 |
| sparse_Endothelial | 0.343 [0.282, 0.406] | 0.062 | 0.595 [0.524, 0.663] | 0.405 [0.337, 0.476] | 0.0161 [0.0025, 0.0384] (25/75) | 0.0038 (122/32,510) | 0/75 |
| sparse_Fibroblast | 0.0004 [0.0001, 0.0009] | 0.111 | 0.889 [0.862, 0.912] | 0.111 [0.088, 0.138] | 0.0000 [0.0000, 0.0000] (25/**26**) | 0.0000 (0/8,663) | **49/75** |

- coverage 분모는 외부 known 5 class 전체(27,212세포)다. donor별 output coverage 범위: nat 0.117–0.889, B 0.128–0.949, Endo 0.094–0.786, Fib 0.019–0.277(`coverage_per_donor.tsv`, 25 donor 전부).
- nat에서 LR_donor의 τ_parent = 1.01(source OOF에서 2%를 만족하는 parent 임계값 없음)이라 parent 출력이 구조적으로 0이다.
- seed 의존성: sparse_B는 seed 0·2에서 τ_fine ≈ 0.9999라 출력이 거의 모두 parent 수준이고, seed 1(τ_fine 0.922)만 fine 출력 14,068개(오류 241, 1.7%)를 냈다. sparse_Fib는 seed 0·1이 거의 무출력, seed 2만 parent 출력.
- 대상 class(보조, donor×seed 세포 합산; `per_class_output_pooled.tsv`): sparse_B의 B 세포 fine 0.233·출력 0.684(출력의 대부분은 B를 특정하지 않는 parent "Immune"); sparse_Endothelial의 Endothelial 세포 fine 0.314·출력 0.367, 출력 중 오류 3.5%(90/2,573); sparse_Fib의 Fibroblast 세포 fine 0, 출력 0.104(전부 parent "Stromal").
- 대응 없는 세포(정확도 아님, 합산): nat에서 Malignant의 16.3%, unclassified의 21.1%가 fine 출력을 받았다(v2 `a4_unmapped_cells.tsv` 기준 Malignant 예측의 대부분은 Hepatocyte). sparse_B에서는 Malignant의 45.6%가 fine 또는 parent 출력.

## 2. 결정표 적용 (고정 규칙의 literal 적용)

| 조건 | 규칙 A (output coverage < 0.50) | 규칙 B (error CI 하한 > 0.02) | 규칙 C (coverage ≥ 0.50 & error CI 상한 ≤ 0.02) | §5.4 행 → 결정 |
|---|---|---|---|---|
| nat | 해당 (0.460) | 아님 | 아님 (coverage) | **1행: 출력이 매우 적음** → 전면 자동 분류용 채택 보류 |
| sparse_B | 아님 (0.519) | 아님 | **해당** | **3행: 이 조건만 기술** (아래 단서) |
| sparse_Endothelial | 해당 (0.405) | 아님 (CI가 0.02 포함) | 아님 | **1행** → 채택 보류; 조건부 오류의 0.02 대비는 미확립 |
| sparse_Fibroblast | 해당 (0.111) | 아님 | 아님 (coverage) | **1행** → 채택 보류; 무출력 49/75행 |

- 2행(출력은 있지만 조건부 오류가 높음)은 어느 조건에서도 규칙 B로 확립되지 않았다. sparse_Endothelial은 점추정 0.016, CI 상한 0.038로 0.02 대비 판정 불가다.
- `decision_mapping.tsv`의 `error_vs_0.02_undetermined` 표기는 RUN_SPEC 문구("B·C 모두 아니면 미확립")대로 붙었지만, **nat(CI 상한 0.006)와 sparse_Fib(0, 26행)는 실제로 CI가 0.02 아래**다. 이 두 조건에서 "미확립"은 오류가 아니라 coverage 때문에 3행에 들지 않았다는 뜻이며, 결정(1행)은 바뀌지 않는다. 표의 문구는 결과를 본 뒤 고치지 않고 여기서 해석만 밝힌다.
- sparse_B 3행의 단서: (i) output coverage의 CI [0.443, 0.598]가 0.50을 포함하므로 규칙 A 경계에 있다(규칙은 점추정 기준). (ii) 출력의 65%가 parent 수준이라 fine coverage는 0.182이고 target B 세포의 fine coverage는 0.233이다. (iii) 3 seed 중 2개는 거의 parent만 출력했다. 따라서 기술할 수 있는 것은 "sparse_B 학습 조건에서, known 세포의 약 절반에 대해 주로 상위 그룹(Immune/Stromal/Parenchymal) 수준으로 답했고, 답한 것의 조건부 오류는 0.009 [0.004, 0.014]"뿐이다. B 세포를 fine 수준으로 찾아내는 용도나 다른 희소 유형으로 일반화하지 않는다.

## 3. 결론

- **완료**: 현재 정책의 실제 사용 범위를 출력 수·보류율·조건부 오류율로 다시 보고했다. 성능 개선이 아니다.
- **채택 보류(전면 자동 분류)**: 4 조건 중 3개(nat·sparse_Endothelial·sparse_Fibroblast)에서 known 세포의 과반에 답하지 않는다. sparse_Fibroblast는 사실상 보류 전용(fine 0.04%)이다. sparse_B도 fine 수준 출력은 18%다.
- 출력이 있는 세포의 조건부 오류는 nat·sparse_B에서 정책 명목값 0.02보다 낮게 관측됐지만(CI 상한 0.006·0.014), 이는 반복 노출된 개발 자료에서의 관측이며 외부 2% 보장이 아니다.
- **하지 않은 것**: 임계값·prior·모델을 외부 결과에 맞춰 바꾸지 않았다. 다른 head(GMM_K 등)의 coverage가 더 높게 관측됐지만(보조 행) 외부 결과로 모델을 바꾸지 않는다. LR_class·MLP는 외부에서 평가되지 않았으므로 비교하지 않는다.
- **다음 행동**: 유지(현재 정책·모델 변경 없음). 대부분의 세포 자동 답변이 필요한지, 일부 고신뢰 답변이면 충분한지(그리고 fine/parent 허용 범위)는 운영 요구사항이며, 확정되면 **다음 독립 평가 전에** 최소 coverage·최대 오류율·fine/parent 허용 범위를 고정한다(plan §5.4).
