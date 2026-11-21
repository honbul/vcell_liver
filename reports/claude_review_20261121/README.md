# Exploratory review diagnostics (2026-11-21)

Scripts and small outputs from the Claude Code review session that ran alongside the frozen Codex tracks (T1–T3, P1–P4). **These are not official results.** They are re-implementations evaluated on inner folds only, with several variants inspected per question, so the numbers are optimistically biased. Official numbers are only those in `reports/goal_tracks/`, `reports/p123/` and `reports/p4/`.

Run location: `g1:/mnt/S1/sdata/workbench/Virtual_Cell/gssong/review_claude_20261121/` (large inputs such as `lincs_sigs.npz` and `split_manifest.tsv` stay on the server).

## What each diagnostic looked at

| File | Question |
|---|---|
| `diag2`–`diag5`, `ceiling` | Well-position effect in Tahoe-100M: same-position vs different-position agreement, DMSO A−B reproducibility, layout structure |
| `diag6` | Position baseline plus Tanimoto kernel on position-corrected residuals, per dose |
| `diag7` | Replicate ceiling decomposition (plate 6 ↔ 14); part B (oracle neighbour) was uninformative |
| `diag8` | DMSO contrast within vs across drug sets; compounds appearing in two drug sets |
| `diag9` | Shared-variance decomposition by pair type (same drug / same position / neither) |
| `diag10` | Per-compound-group reproducible energy (`diag10_groups.tsv`) |
| `diag11` | Trimmed position estimate; sharp nearest-sibling copy; pooled R² metric |
| `diag12`, `lincs_extract` | LINCS L1000 consensus-signature kernel as a prior |
| `diag13`, `diag14` | Line-specific response target (row minus well mean across cell lines) |
| `diag15` | Sibling prediction split into line-shared and line-specific parts; strong-sibling gating |

## How these fared in the frozen evaluations

- Trimmed position estimate (`diag11`): **replicated** in P1 (trim − plain position mean, 99% CI excluded zero in RMSE).
- Signal concentration in a few compound groups (`diag10`): consistent with the P1 and P4 contribution analyses.
- Position share nearly vanishes in the line-specific target (`diag13`): **replicated** in P4 (position baseline R² 3.39% → 0.63% in liver lines).
- LINCS prior gain on Tahoe (`diag12`, +5.9 pp on covered liver groups): **did not replicate** in P2.
- Drug arms beating position on the line-specific target (`diag14`, ~6% R² in liver lines): **did not replicate** in P4 (no arm passed versus B0 or the best known VC).
- Sibling copy beating the trimmed position baseline at 99% on inner folds (`diag15`): the corresponding official gate in P1 failed.

Read the inner-fold numbers as hypotheses that motivated the pre-registered tracks, not as evidence.

## Request memos and manuscript draft

- `requests/` holds the review-session memos that specified each Codex track (well-position review, follow-ups, R14, goal tracks T1–T3, P1–P3, P4, P5 and R15). They were written as private working notes; the header line in each saying they are not committed predates the decision on 2026-11-21 to publish them here. P5 was drafted but not run.
- `MANUSCRIPT_DRAFT.md` is draft v0.1 of the paper. It reflects the state before P4 and R15; the open-items list at the end records the P4 outcome.
