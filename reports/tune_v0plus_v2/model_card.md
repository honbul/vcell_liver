# V0+ v2 모델 카드

V0+ v2는 약물별 감수성 평균 위에 세포별 잔차를 더한다. 잔차에는 약물별 선형·커널, 약물별 GBDT, 이중 축 손실로 학습한 경량 신경망을 사용한다. 세 학습 폴드의 잔차 모델과 비음수 결합 가중치를 동결한다. GDSC2, CTRPv2, PRISM20Q2의 65,685개 반응과 991개 계통만 모델 선택에 사용했다.

입력은 처리 전 bulk RNA와 사전에 등록한 68개 물질이다. 원시 count는 CPM을 적용하고 TPM과 CPM은 log2(1+x)로 바꾼다. 입력 유전자 전체와 실제 선택 축에서 각각 85% 이상의 관측이 필요하다. 누락된 유전자는 학습 중위값으로 대체한다. Geneformer와 scGPT가 필요한 모델은 분리된 FM 입력 없이도 동작하도록 모달리티를 제거하는 학습을 병행했다. 오프라인 CLI는 사용한 FM 블록 수를 반환한다.

```bash
repo=/home/gsun/work/vc_liver/tune_v0plus_v2
package=/mnt/S1/sdata/workbench/Virtual_Cell/gssong/top3_tuning_v2/a/runs/release-01/package
PYTHONPATH="$repo:$repo/src" python -m v0plus_v2.predict \
  --package "$package" --rna expression.tsv --scale bulk_log2_tpm_plus_1 \
  --compounds compounds.tsv --out scores.tsv
```

식별자는 HGNC 기호 또는 `--gene-map`으로 매핑되는 ID를 사용한다. `compounds.tsv`에는 `compound_id` 또는 정확한 parent `smiles`를 둔다. 약물 파일을 생략하면 68개 전부의 점수를 낸다. 감수성 점수가 클수록 체외 스크린의 예상 감수성이 높다. `drug_mean_anchor`와 `personalization_delta`는 약물의 전체 평균과 개인화 기여를 각각 나타낸다.

공개 LIMORE 약물 반응은 동결 후 한 번 채점한다. Broutier 오가노이드, GSE90653 PDX, TCGA-LIHC는 별도 외부 감사다. TCGA에 대응하는 약물 반응 정답은 없다. [CellHit](https://pmc.ncbi.nlm.nih.gov/articles/PMC11828915/) · [CMap LINCS](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)
