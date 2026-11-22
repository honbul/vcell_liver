# Liver DILI VC v1

## Endpoint and inputs

The offline model estimates the probability of an FDA DILIrank 2.0 confirmed concern label for a small molecule, using only a canonicalizable stereochemical SMILES. The positive class includes Most and Less concern and the negative class is No concern. Ambiguous labels are excluded. The probability describes the reference-label task and its fitted class prevalence. It does not measure human incidence, patient-specific risk, dose response, or safety at a proposed exposure.

The release fits a LightGBM classifier on ECFP4, RDKit properties, frozen ChemBERTa embeddings and predicted 24-hour liver pathway scores. Pathway heads for PHH and HepaRG consume structure alone. Human plasma Cmax, approved or postmarketing dose, FDA warning fields, and patient variables are absent from the feature specification. Scores use relative expression contrasts measured in vitro, with the highest observed dose between 1 and 100 µM as their auxiliary training target. The feature represents that data regime rather than a prediction at a user-specified concentration.

## Release and offline usage

The release under G1 `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/liver_dili_v1/release-v2` has `contract.json`, `checkpoint.joblib`, the local ChemBERTa files and a SHA-256 receipt. Model selection used the 759 training rows of the first Bemis–Murcko fold and inner grouped OOF average precision. The chosen pathway-plus-LightGBM specification achieved OOF AUPRC 0.813. Final scores and release-holdout generalization are reported separately. [Release contract](data/release_contract.json) · [Release hash receipt](data/release_receipt.json)

```bash
python -m liver_dili.predict 'CCO' --model /path/to/release-v2
```

Use the G1 study Python 3.10 environment and the track-specific dependency directories `deps`, `embed-deps-v2` and `embed-deps` on `PYTHONPATH`. They contain scikit-learn, RDKit, LightGBM 4.6.0, PyTorch 2.8.0+cpu and Transformers 4.51.3. The encoder uses `local_files_only` and verification used `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The checkpoint is a trusted study joblib artifact.

The CLI outputs predicted fixed-pathway scores, a calibrated probability, the 90% calibration-parameter interval, and a nearest-training ECFP Tanimoto score. The interval comes from 200 group bootstrap draws of the Platt calibration fit on grouped out-of-fold predictions. It captures calibration-fit uncertainty conditional on the models and estimates. Performance and calibration can change on other chemistries or releases. The OOD cutoff is the fifth percentile of the cross-group nearest-training similarity in the fit set, 0.195. Out-of-domain structures and truncated ChemBERTa strings produce explicit flags.

## Reproducibility

1. `python -m liver_dili.fda` ingests both FDA workbook sheets and resolves exact-name structures against PubChem with a cached query ledger. A unique stereoisomeric parent is retained per FDA entry when supported. Entries keep their original FDA identity and observed label.
2. `python -m liver_dili.splits` produces label-free query tables, independently readable score labels and fit rosters. Bemis–Murcko and Butina 0.60 use five outer folds and four grouped internal folds. Connectivity is blocked throughout. The FDA release increment uses original v1 labels for training and new v2 labels for scoring. It excludes a historical compound if its connectivity appears among new v2 entries.
3. `experiments/liver_dili/prepare_phh.R` and `collect_phh.R` produce matched, replicated 24-hour PHH RMA contrasts. Prepare the probe map with `phh_probe_map.R` and, for the G1 worker setup, use `build_preprocesscore.sh` with the checksummed Bioconductor preprocessCore 1.52.1 archive. `liver_dili.assays` harmonizes the two HepaRG datasets, fixed pathways and Cell Painting groups. The pretraining inventory may contain some evaluation structures. It supplies no benchmark DILI labels during the present fits. [ChemBERTa-77M-MLM checkpoint](https://huggingface.co/DeepChem/ChemBERTa-77M-MLM/tree/ed8a5374f2024ec8da53760af91a33fb8f6a15ff)
4. `experiments/liver_dili/benchmark.sbatch` runs each split as `predict` and uses a separate `score` step with the frozen forecast hashes. `liver_dili.transfer` applies the analogous fixed freeze and independent scoring stages for bidirectional pathway transfer.
5. `liver_dili.train_release` publishes the standalone fit. `liver_dili.predict` verifies the model files, then runs the offline CLI.

Run all jobs with the `dili-v1-` prefix. The submitted jobs used CPU. Raw matrices, PubChem caches, array forecasts and model weights are stored on G1.

## Study limitations

Both HepaRG studies were used in prior R14/R19 work. PHH model training and evaluation separate chemical groups, and the assay contrasts derive from known TG-GATEs data. Fixed WikiPathways, Hallmark and PROGENy signatures and train-only gene modules have distinct origins. The 398 TXG-MAPr PHH modules were learned using TG-GATEs PHH and would make a same-source comparison exposure-dependent.

Cell Painting uses A549 measured profiles from a fixed dose code and time. For missing measured profiles, the phenotype model returns the same structure prediction for that compound. DILImap, Tox21, ToxCast and ChEMBL access were audited. These sources did not provide clinical labels or classifier inputs. [Dataset audit](RESULTS.md)
