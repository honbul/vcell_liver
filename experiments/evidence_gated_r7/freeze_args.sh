# Arguments for the R7 freeze (sourced by the freeze job). Every listed run was selected on validation only.
# Primary set = modification-1 models (frozen seed-matched chem backbone); 'v0_' = the original 2x2 before modification 1.
MODELS='{
 "chem_only": ["chem/pilot_chem_lr1e-4_s42", "chem/sel_chem_lr1e-4_s43", "chem/sel_chem_lr1e-4_s44"],
 "B00": ["concat/m1_B00_s42", "concat/m1_B00_s43", "concat/m1_B00_s44"],
 "B01": ["concat/m1_B01_s42", "concat/m1_B01_s43", "concat/m1_B01_s44"],
 "B10": ["gated/m1_B10_s42", "gated/m1_B10_s43", "gated/m1_B10_s44"],
 "B11": ["gated/m1_B11_s42", "gated/m1_B11_s43", "gated/m1_B11_s44"],
 "fixed_graph": ["fixed/m1_fixed_s42", "fixed/m1_fixed_s43", "fixed/m1_fixed_s44"],
 "B11_rewired": ["gated/a1_rewired_s42", "gated/a1_rewired_s43", "gated/a1_rewired_s44"],
 "B11_evidence_off": ["gated/a1_evoff_s42", "gated/a1_evoff_s43", "gated/a1_evoff_s44"],
 "v0_B00": ["concat/pilot_B00_lr1e-4_s42", "concat/sel_B00_lr1e-4_s43", "concat/sel_B00_lr1e-4_s44"],
 "v0_B01": ["concat/pilot_B01_lr1e-4_s42", "concat/sel_B01_lr1e-4_s43", "concat/sel_B01_lr1e-4_s44"],
 "v0_B10": ["gated/pilot_B10_lr1e-4_s42", "gated/sel_B10_lr1e-4_s43", "gated/sel_B10_lr1e-4_s44"],
 "v0_B11": ["gated/pilot_B11_lr1e-4_s42", "gated/sel_B11_lr1e-4_s43", "gated/sel_B11_lr1e-4_s44"],
 "v0_fixed_graph": ["fixed/sel_fixed_aug1_lr1e-4_s42", "fixed/sel_fixed_aug1_lr1e-4_s43", "fixed/sel_fixed_aug1_lr1e-4_s44"]
}'
XPERT='["xpr_e100_s42", "xpr_e100_s43", "xpr_e100_s44"]'
CONTRASTS='[["B11","chem_only"],["B11","B01"],["B11","B10"],["B11","B00"],["B11","fixed_graph"],["B11","B11_rewired"],["B11","B11_evidence_off"],
 ["B11","ecfp_x0_ridge"],["B11","xpert"],["B11","train_cell_mean"],
 ["B00","chem_only"],["B01","chem_only"],["B10","chem_only"],["fixed_graph","chem_only"],["B11_rewired","chem_only"],["B11_evidence_off","chem_only"],
 ["chem_only","ecfp_x0_ridge"],["chem_only","xpert"],["xpert","ecfp_x0_ridge"],["v0_B11","chem_only"],["v0_B11","B11"]]'
STRESS='["B11","B10","B01","B00","fixed_graph","B11_rewired","B11_evidence_off","v0_B11"]'
