# H3-v2 execution record

- Isolated branch `research/nextgen-h3-v2` from `e90c8e2cf684b6c332e34a678239f8770cd485a7`. The base H2 and H3-v1 archives are read-only.
- Slurm work directory `/mnt/S1/sdata/workbench/Virtual_Cell/gssong/nextgen_v2/h3_v2`. `logs/round1-jobs.tsv` lists initial and recovered submissions. `logs/trajectory.tsv` is appended by the inner-selection callbacks. `logs/slurm-observations.txt` records H3-v2 states while they are retained by the controller.
- `35130` and `35131` are the untruncated LINCS R1 arrays. The staged code is `code/full-lincs-v1`. `35132` advances after both finish successfully. The older R1 experiments are separate under `fits/recovery/r1-before-full-lincs`; their scores are outside selection.
- `splits/manifest.json`, `step0_verification.json`, and `external_roster_verification.json` document split, modality, OOF, and query-identity invariants without evaluating held truth.
- Each successful trial writes its own immutable receipt, learning curve and inner predictions. The validation scheduler starts adaptive R2, boundary expansion, final family refits, external feature construction, outcome-free preflight, and a single score pass in that order.
- Only the scorer may create `score/consumed.json`. Shared source and reference artifacts are not writable by the pipeline. While the scorer is live, the claim distinguishes already open evaluation from an incomplete receipt.
- The publication package appears in `reports-final` after independent support and archived baseline checks. The worktree has `verify_delivery.py` to enforce the final scope, hashes, 3-seed tables, decision rules and artifact set before release.
