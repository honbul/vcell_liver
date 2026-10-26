# 사전학습 overlap 확인 범위

2026-10-26에 공식 scGPT data README/CellXGene 전처리 경로와 Geneformer 모델 카드, 공식 Genecorpus-104M dataset card 및 파일 목록을 확인했다. GSE149614 또는 후속 GSE151530의 cell/환자 단위 비중복을 입증할 공개 accession-to-cell 목록을 확보하지 못했으므로 두 모델 모두 `pretraining_overlap=unknown`이다.

- scGPT 공식 코드의 고정 revision: cebd6fae655b9c585a4807daa3ac31bb764f06b4. [데이터 안내](https://github.com/bowang-lab/scGPT/blob/cebd6fae655b9c585a4807daa3ac31bb764f06b4/data/README.md)는 CellXGene 전처리 경로를 가리킨다. whole-human weights의 정확한 환자별 학습 포함 목록과 본 HCC cell barcode의 대조를 완료한 것은 아니다.
- [Geneformer 모델 카드](https://huggingface.co/ctheodoris/Geneformer)는 V2 비암 corpus와 CLcancer를 구분한다. 비암 pretraining이라는 사실은 HCC 연구의 면역/인접 비종양 세포가 제외됐음을 증명하지 않는다.
- 공식 [Genecorpus-104M dataset card](https://huggingface.co/datasets/theodoris-lab/Genecorpus-104M/blob/28eda69789cb9ff8865d526b3e176e7c686e0d99/README.md)와 API 파일 목록을 확인했다. 당시 공개 파일은 tokenized Arrow dataset, length pickle, dataset_info/state와 README이며 accession 매핑 테이블은 찾지 못했다. 카드의 입력 필드는 input_ids이고 metadata annotation이 없다고 안내한다. corpus 전체를 받지 않았다.

따라서 이번 6/2/2 donor split은 downstream 분류기 평가 누수 방지다. pretraining-unseen 환자 성능이라고 보고하지 않는다. 확인하지 못한 항목을 false/non-overlap으로 채우지 않는다.
