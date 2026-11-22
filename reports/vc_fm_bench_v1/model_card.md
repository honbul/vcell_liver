# Track 6 foundation representations

Research use on public human liver pharmacogenomics and FDA label data. The frozen store records four independent axes where available: compound parent InChIKey, UniProt accession, bulk profile ID, and assay row ID. Each vector collection has an input, checkpoint, and result hash and includes device, elapsed time, peak memory, and failure information. The store and patient or compound-level output live on G1.

## Representations and expected exposure

| Mode | Frozen input and encoder | Exposure to evaluate |
|---|---|---|
| ECFP4 + physicochemical | Radius two, 2,048-bit Morgan plus nine RDKit descriptors | Deterministic chemistry representation, zero learned parameters. |
| ChemBERTa-2 77M-MLM | Pinned DeepChem model, mean of token states excluding special and padding tokens | Paper reports PubChem structures. DILI drugs and assay compounds can be present as structures. Record-level training membership has not been established. Gated access is disabled in the offline extractor. |
| MoLFormer XL | IBM author code at `7b12d946c181a37f6012b9dc3b002275de070314`, same weight hash as the newer artifact. Masked pretrained pooled state with deterministic evaluation | Paper reports PubChem and ZINC. Target catalog structures may have been seen. The older author release matches Transformers 4.57. |
| Uni-Mol no-H | Pinned author checkpoint. RDKit ETKDGv3 with seed 42, UFF relaxation, CLS state. Failed 3D configurations remain explicit. | Molecular 3D pretraining exposure may include registered chemistry. Known scaffold independence applies to supervised assay labels. |
| CLAMP molecule encoder | Official ICML 2023 checkpoint, 768 features. Chiral feature-Morgan counts and RDKit path counts are folded to a common 8,192-bin space and transformed with log1p. The published layer-normalized molecule network reads those features offline. | PubChem bioassay-aligned pretraining can include chemical identity and pharmacology associations. Assay-level and label overlap with the benchmark are unverified. [Original paper and pretraining data](https://proceedings.mlr.press/v202/seidl23a.html) |
| Rank-HVG | Within-profile fractional ranks; 512 high-variance features selected in fit-only data | Basal gene axes fixed from the source bundle; selection recomputed inside inner and outer training roles. |
| Pathway and TF activities | Fit-standardized ranks projected onto PROGENy top 100 weighted targets, Hallmark memberships and DoRothEA confidence A–C signed targets | Curated priors reflect literature and gene expression measurements. Signed projections run without fitting pathway effects to drug labels. |
| scGPT whole-human | Author vocabulary and checkpoint. Nonzero input genes, per-profile 51-bin expression, deterministic ID-seeded subsampling to 1199 features, final CLS with unit norm | Human CELLxGENE pretraining catalog can overlap public atlas source studies. CCLE/TCGA membership of these inputs is unverified. For bulk inputs, CLS is an extrapolation to profile-level measurements. |
| Geneformer V2-104M | Author 104M vocabulary, gene median and alias dictionaries; duplicate Ensembl-mapped features summed, median-normalized nonzero rank, penultimate CLS; 4094 gene limit | Corpus composition is heterogeneous. Exact CCLE/TCGA/GEO study membership for these RNA profiles is unverified. The CLcancer model is excluded. |
| STRING human v12 | Clustered HGNC aliases; combined score ≥700; rank-96 embedding of the normalized symmetric graph | Uses public molecular associations and includes literature-derived evidence. |
| ESM-2 t12 35M | Reviewed Homo sapiens UniProt canonical protein sequence, mean of non-special residue states. Long sequences are tiled and pooled by represented residue count. | UR50D-like protein pretraining can include homologs or exact sequences. This benchmark is conditional on public protein sequences and audits supervised labels separately. |

ChemBERTa-2 and scGPT checkpoint distribution cards leave separate weights-license fields unspecified. Their weights are fetched from author-linked resources for the research evaluation. The CLAMP weight terms permit this evaluation with attribution to the original authors. Record-level exposure to CCLE, ChEMBL and TCGA is unknown unless listed in the pinned release source. [CLAMP weight terms](https://huggingface.co/scikit-fingerprints/clamp/blob/4b919da99ef1cc25755506ddf2197732154d5afe/LICENSE)

## Benchmark scope

1. Pan-cancer training uses GDSC2, CTRPv2 and PRISM 20Q2 with exact parent mapping and no LIMORE response. Evaluation uses 50 independent LIMORE lineages. Training endpoints are converted to common direction and source-standardized inside each fitting role. Evaluation retains native target endpoints. Historical project exposure to LIMORE is recorded.
2. The drug-cold panels split on generic Murcko scaffolds and check for any connectivity overlap. Both LIMORE and the GDSC2 panel supported by the harmonized cross-source registry are scored. The chemical-only assay endpoints are comparable in direction. Full GDSC2 compound coverage is outside this registered subset.
3. LICOB RNA responses from the repository fail the deposit checksum and ZIP integrity. LIMORE cell-cold is the prescribed external fallback and recycles the same 2D endpoint; it contributes no independent 3D result.
4. DILI reports v2 scaffold splits and v1→v2 release increment. Positive is either of the FDA defined concern categories, negative is FDA no concern. Ambiguous or conflicting parents are excluded. The increment axis uses the v1 label during training and excludes connectivity-identical v1 parents from training. A release increment measures transfer across FDA dataset versions.

Each representation uses the same families and candidate grid with train-group-only hyperparameter selection. Fold-specific chemistry coverage and RNA coverage define the matched test support. Baselines include training means, matched drug means and per-drug Ridge where that drug receives training labels, single-gene ABCB1 and MKI67, and cLogP plus the fixed molecular weight/lipophilicity rule on DILI. Feature concatenations compare conventional and foundation representations on the same support. Rank scores describe observed preclinical assays and DILI categories.

## Offline entry points

Run from a checkout with a Python environment matching the experiment runtime.

```bash
python -m vc_fm_bench.embed --help
python -m vc_fm_bench.conformers --help
python -m vc_fm_bench.transcriptome --help
python -m vc_fm_bench.prepare --help
python -m vc_fm_bench.tasks --help
python -m vc_fm_bench.benchmark predict --help
python -m vc_fm_bench.benchmark score --help
python -m vc_fm_bench.report --help
```

The query TSV uses `id` and `smiles` for molecules or `id` and `sequence` for proteins. Transcriptomes are index-keyed rows with unique gene symbols or Ensembl IDs appropriate to the selected checkpoint and the declared expression scale. The published preparation and Slurm scripts preserve immutable output directories and hash evidence. All four head families use only training or training-validation response.

## Matched public comparators

CellHit per-drug XGBoost is refit on training-only HVG ranks and normalized source AUC for the external LIMORE cell-cold panel. Authors' original ln(IC50) and PRISM LFC model outputs use a different endpoint. The per-drug model requires a training assay for that compound, which excludes it from scaffold-cold fitting. For both DILI designs, an independent D-MPNN comparator passes messages on directed molecular bonds and blocks immediate backtracking. Its architecture follows Yang et al., with a fixed RDKit feature schema and scaffold-blocked model selection. Both comparators are frozen and scored on the exact benchmark IDs and matched outcome checksums. [CellHit primary study](https://doi.org/10.1038/s41467-025-56827-5), [Chemprop D-MPNN primary study](https://doi.org/10.1021/acs.jcim.9b00237)
