# R9 v2 environment (research/r9-normalization-prototype-v2). Reuses the v1 core conda env read-only
# (vcell_liver_r9_rare_state/assets/envs/r9core: numpy 1.26.4, scipy 1.14.1, sklearn 1.5.2, torch 2.5.1+cpu,
# pandas 2.2.3, anndata 0.10.9) and writes only under the v2 worktree.
export R9_ROOT="${R9_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_norm_proto_v2}"
export R9_V1_ROOT="${R9_V1_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_r9_rare_state}"
export R9_PY="${R9_PY:-$R9_V1_ROOT/assets/envs/r9core/bin/python}"
source "$R9_ROOT/scripts/r9/env.sh"
export PYTHONPATH="$R9_ROOT/scripts/r9:$R9_ROOT/scripts/r9/v2${PYTHONPATH:+:$PYTHONPATH}"
