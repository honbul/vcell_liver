# Internal manuscript review

[Editable review manuscript](REVIEW.docx) · [Markdown source](DRAFT.md) · [Table 1 source rows](table1.tsv) · [Evidence and scope by claim](CLAIMS.md) · [Artifact checksums](review_receipt.json)

The review manuscript reports two distinct information settings. The HEPG2 benchmark predicts the later response from a measured earlier response to the same drug. The adult HCC benchmark receives measured same-drug responses from eight nonhepatic cells. Its high-dose development results, lower-dose limits and between-line comparison appear together. The Article format and references follow the current journal guidance for internal review. [Scientific Reports submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines) · [Initial submission checklist](https://www.nature.com/documents/srep-checklist-for-initial-submissions.pdf)

- **Evidence awaiting measurement.** The independent SNU-423/HUH7 pilot has frozen predictions and a scoring protocol. The available pilot assay is synthetic. Independent measurements are required for the intended external HCC claim. The R18 reserved confirmation is a separate experiment with no scored result. [Independent pilot](../HCC_OFF_POSITION_PILOT_RESULTS.md) · [R18 status](../R18_CONFIRMATION_PROTOCOL.md)
- **Items for author verification.** Confirm names, affiliations, corresponding author, contributions, funding, competing interests, study-specific ethics applicability, and the description of AI assistance before submission. The review author line is explicitly pending. [Journal declarations](https://www.nature.com/srep/author-instructions/submission-guidelines) · [Journal AI policy](https://www.nature.com/srep/journal-policies/ai)
- **Reader access.** The draft identifies public data sources and exact CMap files. Confirm a stable release with code, split manifests, runnable scripts, score artifacts, version and access terms. The repository's review files are local. [Journal code and data policy](https://www.nature.com/srep/journal-policies/editorial-policies)

## Rebuild the review copy

Run from the worktree root with Poppler's `pdftoppm` installed. The exporter checks the selected score hashes, forecast lineage and support counts, and writes an editable Word table. It embeds previews of the three vector figures and records their source hashes. The original figures remain available separately. [Temporal figure](figure1.pdf) · [Donor transfer figure](figure2.pdf) · [Adult HCC benchmark figure](figure3.pdf)

```bash
uv run --with python-docx==1.2.0 --with PyYAML==6.0.3 \
  python -m src.r19_plan.manuscript_review \
  --manuscript reports/r19_plan/manuscript \
  --output reports/r19_plan/manuscript \
  --git-revision "$(git rev-parse HEAD)"
```
