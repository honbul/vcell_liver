# 연구 확장 V2 진행 상태

시작 기준 commit: 243fa6639ade4b8882c5dc4e1a95c6d76c8d4d79. 기존 코드·환경·데이터·결과를 보존하고 research/phase2-v2에서 확장한다. 실행 요청서는 docs/RESEARCH_EXPERIMENT_PLAN_V2.md.

P0: 기존 자산과 현재 GPU 사용 상태를 확인했다. 2026-10-26 접속 시 A30 4개는 각각 약4MiB/0% 사용이었다. 이후 사용 상태를 다시 확인하며 다른 작업은 중단하지 않는다. 기존 scGPT/Geneformer 환경은 변경하지 않는다.

Legacy HCC02/HCC09는 과거 회귀 확인용이며 새로운 blind test가 아니다. 개발은 나머지 8명, donor-grouped outer4fold와 donor-disjoint inner validation2명으로 고정한다. 첫 연결에는 기존10k 중 개발8명만 사용하며, 이후 전체 적격 개발 donor의 원 분포를 유지한 평가로 확장한다. 외부 GSE151530 HCC는 최종 평가로 예약했으며 metadata/ontology만 확인하고 모델 예측·점수는 파이프라인 선택 동결 후 생성한다.

공식 재현, HCC annotation(inductive/query-adaptive 분리), embedding_shift, genetic_response, 조건부 chemical_response를 서로 다른 task/regime으로 기록한다. 현재 새 모델의 실행 완료를 주장하지 않는다. 새 연구 기능이 실제 실행될 때 관련 tutorial section을 추가한다.


## Executed development pilot and official-example smoke

The 8,760-cell development pilot completed all four donor folds for PCA, scGPT and Geneformer (12 executed runs). These are repeatedly inspected development OOF results, not a blind test. Full development data contain 63,036 cells; official V2 tokenization is complete. Frozen full-data embedding extraction and the complete official MS tutorial are next.

R00_ms_smoke_seed0_v2 executed one epoch on 64 reference/64 query cells and verified all 24 pretrained attention tensors. Its scores are not an official quantitative reproduction. The complete run retains the original 10 epochs and batch size 32, subject to a separately recorded hardware failure if necessary.

Separate research and LiverCT environments preserve the existing scGPT/Geneformer installations. Initial dependency resolution failed (AnnData constraint; unavailable Python 3.10 Zarr version; Conda packaging of pip/setuptools). Corrected installers and original logs are retained. Norman and the authentic GO prior were recovered from the public Harvard Dataverse endpoint with cache-busting download parameters after the original endpoint returned 403. scFoundation official SharePoint weight links both returned HTTP 403; the official Figshare examples archive contains no checkpoint. No substitute weights are treated as official.
