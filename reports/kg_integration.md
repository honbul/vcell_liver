# KG 특징 실험

기존 VCWorld export는 provenance가 포함된 표준 gene-pathway membership 테이블이 아니므로 수정하지 않고 보존했다. 공식 Reactome v92 human Ensembl→pathway를 사용한다. gene expression은 환자 관측, membership은 외부 prior다. 경로별 평균 train-zscore를 재현 가능한 proxy로 사용하며 활성화/인과관계로 해석하지 않는다.

10–200개의 관측 gene을 포함하는 경로를 train feature variance로 정렬하고, 32/64/128 경로 수 및 logistic C/weight를 validation에서만 선택한다. test는 선택에 사용하지 않는다. foundation은 scGPT frozen embedding. 크기/유전자 membership degree를 모두 보존한 edge-swap random control(seed42)을 사용한다. pathway membership을 예측 입력에 concatenate하는 실험이며 LLM/RAG나 인과 모형이 아니다.

실제 성능은 kg_ablation_*.tsv. 이득이 없거나 random보다 약해도 그대로 보고한다. 모든 head는 train-only scaling, 동일 환자 split·평가 cell IDs·class 정의를 사용한다. test 환자 2명은 외적 타당성 판단에 제한적이다.
