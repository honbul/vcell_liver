# GSE149614 데이터 카드

공개 processed raw count. 71,915 cells × 25,712 genes.

원본 헤더: Cell, sample, res.3, site, patient, stage, virus, celltype. 전문가 원 annotation의 6개 broad lineage를 그대로 사용한다. Tumor 출처를 malignant 정답으로 바꾸지 않는다. Normal은 인접 비종양 조직이며 건강인 cohort가 아니다.

세포 ID를 원본 그대로 유지. metadata↔matrix 누락 0/0. 원 count는 sparse X와 layers[counts]에 보존. 전체 dense 변환 없이 유전자별 스트리밍. 비어 있는 세포만 제외(0); 원 processed 데이터에 추가 QC cutoff를 적용하지 않음. 전체 QC 분포는 qc_summary.tsv 참조.

GENCODE v47 basic annotation(Ensembl 113)의 정확한 symbol 매칭. ambiguous/unmapped는 임의 매핑하지 않음. Ensembl suffix를 annotation에서 제거하고 유일한 ID만 매핑. Geneformer는 매핑 가능한 유전자만 별도 입력하며 n_counts는 원 전체 library sum을 보존.

고정 환자 split: {"HCC06": "train", "HCC07": "train", "HCC01": "train", "HCC08": "train", "HCC04": "train", "HCC03": "train", "HCC05": "val", "HCC10": "val", "HCC02": "test", "HCC09": "test"}. seed 42, 모델 결과 확인 전에 고정. 모든 조직은 같은 환자 split. 환자×라벨 층화 nested 10000/2000/64 subset, 원본 ID/분할은 manifests/splits. train에 없는 라벨은 미평가로 별도 집계.

전처리 HVG/scaler/PCA와 분류기 학습은 train만 사용. validation은 고정 grid 선택에 사용. test는 최종 평가에만 사용. 사전학습 cohort와 중복 여부는 unknown; donor holdout을 pretraining-unseen으로 해석하지 않는다.

실제 파일: /mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/processed/GSE149614.h5ad, /mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver/subsets/{smoke,pilot,comparison}.h5ad. 전체 raw와 count는 Git에 포함하지 않음.
