"""Run offline functional inference with a fixed, checksummed model release."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

from .io import publish_json, sha256


def load_request(path):
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field or gene ID {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def dose_uM(value, unit):
    factor = {"M": 1e6, "mM": 1e3, "uM": 1, "µM": 1, "μM": 1, "nM": 1e-3, "pM": 1e-6}
    if isinstance(value, bool) or not isinstance(value, (int, float)) or unit not in factor:
        raise ValueError("Invalid concentration or unsupported unit")
    converted = value * factor[unit]
    if converted < 0 or not math.isfinite(converted):
        raise ValueError("Concentration must be finite and nonnegative")
    return converted


def _check_model_artifact(model_dir, contract):
    release = json.loads((model_dir / "receipt.json").read_text(encoding="utf-8"))
    hashes = release.get("files")
    if not isinstance(hashes, dict) or not {"contract.json", "checkpoint.joblib"} <= set(hashes):
        raise ValueError("Model release receipt lacks artifact hashes")
    for name, expected in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or not name or name in {".", ".."}:
            raise ValueError("Unexpected model artifact path")
        if sha256(model_dir / name) != expected:
            raise ValueError(f"Model artifact hash mismatch for {name}")
    interval = contract.get("interval")
    if interval is not None:
        if not isinstance(interval, dict) or not {"half_width", "method"} <= set(interval):
            raise ValueError("Malformed interval contract")
        width = interval["half_width"]
        if isinstance(width, bool) or not isinstance(width, (int, float)) or width < 0 or not math.isfinite(width):
            raise ValueError("Invalid model interval width")


def _common_rejection(request, cell, contract):
    if request.get("input_mode") != "baseline_rna_and_structure":
        return "unsupported_input_mode"
    if cell.get("lineage_role") != "adult_hcc" or cell.get("culture_format") != contract.get("culture_format"):
        return "unsupported_lineage_or_culture"
    if cell.get("expression_scale") != contract.get("expression_scale"):
        return "unsupported_expression_scale"
    if request.get("assay_id") != contract.get("assay_id"):
        return "unsupported_assay"
    if request.get("time_h") != contract.get("time_h"):
        return "unsupported_time"
    genes = cell.get("baseline_rna")
    if not isinstance(genes, dict) or not genes or any(
        not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x)
        for x in genes.values()
    ):
        return "invalid_baseline_rna"
    return None


def infer(model_dir, request):
    from .functional import DomainError, load_model

    model_dir = Path(model_dir)
    contract = json.loads((model_dir / "contract.json").read_text(encoding="utf-8"))
    _check_model_artifact(model_dir, contract)
    cell, candidates = request.get("cell"), request.get("candidates")
    if not isinstance(cell, dict) or not isinstance(candidates, list) or not candidates:
        raise ValueError("Expected a cell and a nonempty candidate list")
    for field in ("request_id", "candidate_pool_id"):
        if not isinstance(request.get(field), str) or not request[field]:
            raise ValueError(f"Expected nonempty request {field}")
    if not isinstance(cell.get("cell_id"), str) or not cell["cell_id"]:
        raise ValueError("Expected a query cell ID")
    identities = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidates must be objects")
        identity = candidate.get("compound_id") or candidate.get("smiles")
        if not isinstance(identity, str) or not identity:
            raise ValueError("Each candidate needs a compound ID or an isomeric structure")
        identities.append(identity)
    if len(set(identities)) != len(identities):
        raise ValueError("Candidate pool contains duplicate IDs")

    common = _common_rejection(request, cell, contract)
    model = None if common else load_model(model_dir)
    interval = contract.get("interval")
    rows = []
    resolved = set()
    for identity, candidate in zip(identities, candidates):
        reason, concentration, value, flags = common, None, None, {}
        if "dose" in candidate or "dose_unit" in candidate:
            try:
                concentration = dose_uM(candidate.get("dose"), candidate.get("dose_unit"))
            except ValueError:
                reason = reason or "invalid_concentration"
            else:
                reason = reason or ("vehicle_has_no_curve_summary" if concentration == 0 else "curve_summary_has_no_dose_query")
        if reason is None and interval is None:
            reason = "uncertainty_calibration_unavailable"
        if reason is None:
            try:
                rna = cell["baseline_rna"]
                smiles, compound_id = candidate.get("smiles"), candidate.get("compound_id")
                if hasattr(model, "assess_domain"):
                    flags = model.assess_domain(rna, smiles, compound_id)
                    registered = flags.get("registered_compound_id")
                    if registered in resolved:
                        raise DomainError("duplicate_registered_compound")
                    if registered:
                        resolved.add(registered)
                value = float(model.predict(rna, smiles, compound_id))
                if not math.isfinite(value):
                    raise ValueError("Non-finite result in model artifact")
            except DomainError as error:
                reason = error.reason
        width = interval["half_width"] if value is not None else None
        rows.append({
            "model_version": contract["model_version"],
            "request_id": request["request_id"], "input_mode": request.get("input_mode"),
            "cell_id": cell["cell_id"], "compound_id": identity,
            "dose_uM": concentration, "time_h": request.get("time_h"), "assay_id": request.get("assay_id"),
            "endpoint_id": contract["endpoint_id"], "predicted_value": value,
            "predicted_unit": contract["predicted_unit"], "endpoint_contract": contract["endpoint_contract"],
            "source_protocol": contract["source_protocol"], "supported_dose_response": False,
            "interval_lower": None if width is None else value - width,
            "interval_upper": None if width is None else value + width,
            "interval_method": None if width is None else interval["method"],
            "program_prediction_available": False, "program_definition_version": None,
            "rank_in_candidate_pool": None, "candidate_pool_id": request["candidate_pool_id"],
            "supported": reason is None, "abstention_reason": reason, "domain_flags": flags,
            "data_provenance_summary": contract.get("data_provenance_summary", {
                "source_id": contract.get("source_id"),
                "basal_expression_scale": contract.get("expression_scale"),
                "basal_expression_is_proxy": contract.get("baseline_is_proxy"),
            }),
            "limitations": contract.get("limitations", [
                "registered_cell_line_curve_summary", "public_development_interval", "research_use",
            ]),
        })
    direction = contract.get("endpoint_direction")
    if direction not in {"higher", "lower"}:
        raise ValueError("Model contract must declare the direction of its endpoint")
    eligible = sorted(
        (row for row in rows if row["supported"]),
        key=lambda row: ((-1 if direction == "higher" else 1) * row["predicted_value"], row["compound_id"]),
    )
    for rank, row in enumerate(eligible, 1):
        row["rank_in_candidate_pool"] = rank
    return {"results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Trusted model release directory")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    request = load_request(args.request)
    start = perf_counter()
    output = infer(args.model, request)
    if args.benchmark:
        import resource

        output["benchmark"] = {
            "latency_seconds": perf_counter() - start,
            "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "candidates": len(output["results"]),
            "abstentions": sum(not row["supported"] for row in output["results"]),
        }
    if args.out:
        publish_json(args.out, output)
    else:
        print(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
