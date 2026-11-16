# Environment for the HCC genotype-FiLM study (plan v3). Existing envs are used read-only.
export GF_ROOT=${GF_ROOT:-/mnt/S1/sdata/workbench/Virtual_Cell/gssong/vcell_liver_genotype_film_v3}
export GF_PY=/mnt/S1/sdata/pubdata/Virtual_Cell/gssong/vcell_liver_chemcpa_moe_v2/envs/chemcpa/bin/python
cd "$GF_ROOT"
mkdir -p logs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
