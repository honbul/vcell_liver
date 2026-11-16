#!/usr/bin/env bash
#SBATCH --job-name=psa1_final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/psa1_final_%j.log
# Stage A5: copy light prediction tables into the repository tree, draw figures, re-run checks and
# execute the tutorials once (executed copies go to assets, the committed notebooks stay output-free).
# Submit: sbatch --export=ALL,MAIN_RUN=..,CONTROLS_RUN=..,EXT_RUN=.. experiments/pathway_sensitivity_a1/a5_finalize.sbatch
set -euo pipefail
: "${MAIN_RUN:?}" "${CONTROLS_RUN:?}" "${EXT_RUN:?}"
source experiments/pathway_sensitivity_a1/env.sh
R=reports/pathway_sensitivity_a1
echo "host=$(hostname) commit=$(git rev-parse HEAD) start=$(date -Is)"
for run in "$MAIN_RUN" "$CONTROLS_RUN" "$EXT_RUN"; do
  mkdir -p "$R/oof/$run"
  cp -rn "$PSA1_RUNS/$run/." "$R/oof/$run/"
done
printf '{"main": "%s", "controls": "%s", "external": "%s"}\n' "$MAIN_RUN" "$CONTROLS_RUN" "$EXT_RUN" > "$R/oof/RUNS.json"
"$PSA1_PY" "$PSA1_SRC/make_figures.py" --tables "$R/tables" --out "$R/figures" --oof-dir "$R/oof/$MAIN_RUN/oof"
"$PSA1_PY" -m pytest -q -p no:cacheprovider "$PSA1_SRC/tests" | tee "$R/tables/checks_pytest.txt"
EXE="$PWD/assets/tutorial_runs/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$EXE"
for nb in tutorials/pathway_sensitivity_a1/0*.ipynb; do
  "$PSA1_ENV/bin/jupyter" nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
     --ExecutePreprocessor.kernel_name=python3 --output-dir "$EXE" "$nb"
  echo "executed $nb"
done
rm -rf "$R/figures_regenerated"
echo "end=$(date -Is)"
