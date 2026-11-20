"""Freeze the role-local choices and check every scientific input before final inference."""

import hashlib
import json
from pathlib import Path
import subprocess

from .common import arguments, config, read_json, sha256, write_json


def configuration_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation():
    paths = list(Path("src/r11_chemical_probe").glob("*.py"))
    paths += [Path("src/moe_drug_v1/prepare.py"), Path("src/r6_followup_v1/metrics.py")]
    return {str(p): sha256(p) for p in sorted(paths)}


def records(assets):
    specs = [ "metadata/receipt.json", "metadata/samples.tsv", "features/receipt.json", "features/features.npz",
             "features/test_similarity.tsv",
             "source/receipt.json", "development/panels.json",
             "development/A/grid.tsv", "development/B/grid.tsv",
             "development/A/selection.json", "development/B/selection.json",
             "development/final_panel.tsv"]
    return {p: sha256(assets / p) for p in specs}


def verify(assets, cfg):
    spec = read_json(assets / "freeze.json")
    checks = (configuration_hash(cfg) == spec["config_sha256"],
              implementation() == spec["implementation_sha256"], records(assets) == spec["assets_sha256"])
    if not all(checks):
        raise ValueError(f"Frozen config, implementation, or input receipt changed: {checks}")
    return spec


def freeze(assets, cfg, config_path, commit):
    if (assets / "final").exists() or (assets / "predictions").exists():
        raise ValueError("Final data or predictions exist before the specification freeze")
    snapshots = records(assets)
    feature_hashes = read_json(assets / "features" / "receipt.json")["outputs"]
    for file in ("features.npz", "test_similarity.tsv"):
        if feature_hashes[file] != snapshots[f"features/{file}"]:
            raise ValueError(f"{file} hash differs from extraction receipt")
    if read_json(assets / "features" / "receipt.json")["input_hash"] != snapshots["metadata/samples.tsv"]:
        raise ValueError("Feature cache used a different sample manifest")
    roles = {}
    for role in ("A", "B"):
        selected = read_json(assets / "development" / role / "selection.json")
        if selected["role"] != role or selected["stage"] != "development" or (
            selected["grid_sha256"] != snapshots[f"development/{role}/grid.tsv"]
        ):
            raise ValueError("Development receipt differs from selected role")
        roles[role] = selected
    e3 = [role for role, spec in roles.items() if spec["e3"]["enter"]]
    write_json(assets / "freeze.json", dict(
        commit=commit, config_sha256=configuration_hash(cfg), yaml_sha256=sha256(config_path),
        implementation_sha256=implementation(), assets_sha256=snapshots,
        sample_manifest_sha256=snapshots["metadata/samples.tsv"],
        panel_sha256=snapshots["development/final_panel.tsv"], roles=roles,
        metric="within-pool compound-group balanced drug-contrast RMSE",
        scaffold_slice="fixed model; all test group entities have nonempty source-unseen Bemis–Murcko scaffolds",
        exposure=cfg["exposure"], official=[["C", "B*"]] + ([["F", "B†"]] if len(e3) == 2 else []),
        official_ci=0.975 if len(e3) == 2 else 0.95,
        E3_roles=e3, attribution_ci=0.95, seeds=[42, 43, 44]
    ))
    print(sha256(assets / "freeze.json"), flush=True)


def main():
    parser = arguments(__doc__)
    parser.add_argument("--commit", help="Source commit for deployment without .git")
    args = parser.parse_args()
    commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    freeze(args.assets, config(args.config), args.config, commit)


if __name__ == "__main__":
    main()
