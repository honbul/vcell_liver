#!/usr/bin/env bash
#SBATCH -J r7_xpsetup
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 12:00:00
#SBATCH -o assets/logs/%x_%j.out
# XPert comparator: separate conda env + public Zenodo resources (CC-BY-4.0). No author weights are downloaded or used.
# flash_attn is replaced by a shim with the same call signature built on torch scaled_dot_product_attention (no mask, same
# dropout); recorded in xpert_card.json. Generic User-Agent only.
set -euo pipefail
source experiments/evidence_gated_r7/env.sh
echo "start=$(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-}"
ENV="$R7_ASSETS/envs/xpert"
if [ ! -f "$ENV/.r7_installed" ]; then
  [ -x "$ENV/bin/python" ] || /usr/local/anaconda3/bin/conda create -y -q -p "$ENV" python=3.9 >/dev/null
  "$ENV/bin/pip" install -q torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  "$ENV/bin/pip" install -q "numpy<2" torch-geometric==2.6.1 scanpy==1.9.8 anndata h5py==3.10.0 PyYAML==6.0.1 scikit_learn==1.4.0 \
      torchmetrics==1.6.0 tqdm==4.66.1 matplotlib==3.8.2 unimol-tools==0.1.4.post1 rdkit huggingface_hub
  touch "$ENV/.r7_installed"
fi
SHIM="$ENV/lib/python3.9/site-packages/flash_attn"
mkdir -p "$SHIM"
cat > "$SHIM/__init__.py" <<'PY'
"""R7 shim: flash_attn is not installed; flash_attn_func is re-implemented with torch SDPA (same math, no mask)."""
PY
cat > "$SHIM/flash_attn_interface.py" <<'PY'
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kw):
    # flash_attn layout [batch, seq, heads, head_dim] -> SDPA layout [batch, heads, seq, head_dim]
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return o.transpose(1, 2).contiguous()
PY
"$ENV/bin/python" -c "import torch, torch_geometric, scanpy, unimol_tools; from flash_attn.flash_attn_interface import flash_attn_func; print(torch.__version__, torch_geometric.__version__, scanpy.__version__)"
"$ENV/bin/pip" freeze > "$R7_ASSETS/envs/xpert_freeze.txt"
D="$R7_EXT/xpert_zenodo"; mkdir -p "$D"; cd "$D"
for f in XPert.zip processed_data.zip all_drugs_unimol_arr.npy; do
  if [ ! -s "$f" ]; then
    curl -sSL --retry 5 -A "$R7_UA" -o "$f.part" "https://zenodo.org/api/records/17182939/files/$f/content" && mv "$f.part" "$f"
  fi
  ls -l "$f"
done
curl -sSL -A "$R7_UA" -H "Accept: application/json" -o record_17182939.json https://zenodo.org/api/records/17182939
sha256sum XPert.zip processed_data.zip all_drugs_unimol_arr.npy > sha256.txt
python3 - <<'PY'
import json
r = json.load(open("record_17182939.json")); want = {f["key"]: f["checksum"] for f in r["files"]}
import hashlib, subprocess
for f in ["XPert.zip", "processed_data.zip", "all_drugs_unimol_arr.npy"]:
    md5 = subprocess.run(["md5sum", f], capture_output=True, text=True).stdout.split()[0]
    print(f, "md5", md5, "zenodo", want[f], "OK" if want[f].endswith(md5) else "MISMATCH")
PY
unzip -q -o processed_data.zip -d processed_data_unz
unzip -q -o XPert.zip -d xpert_zip_unz
find processed_data_unz xpert_zip_unz -maxdepth 4 -type f -size +0 | grep -v "\.py$" | head -80
echo "end=$(date -Is)"
