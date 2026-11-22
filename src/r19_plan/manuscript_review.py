"""Build an editable internal review manuscript from hash-verified development scores."""

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from src.r11_chemical_probe.common import sha256, write_json


LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
TABLE = (
    ("HEPG2, 6 to 24 h", "Fit-only scalar", "t4_scaffold/score/paired.tsv", "scope", "parent",
     "purged_operator", "purged_scalar", 132, 268, "t4_scaffold/freeze.json"),
    ("HEPG2, 6 to 24 h", "Matched State", "t4_state/score/paired.tsv", "unit", "parent",
     "purged_operator", "State_6h", 132, 268, "t4_state/freeze.json"),
    ("JHH-5, 10 µM", "Best public adaptation", "jhh5_off_position_native/score/best.tsv", "scope", "all",
     "calibrated_shrink", "best_VC", 305, 309, "jhh5_off_position_native/freeze.json"),
    ("JHH-7, 10 µM", "Best public adaptation", "jhh7_off_position_native/score/best.tsv", "scope", "all",
     "calibrated_shrink", "best_VC", 88, 88, "jhh7_off_position_native/freeze.json"),
    ("JHH-5, 10 µM centered", "Best public adaptation", "jhh_off_position_contrast/JHH5/centered_best.tsv", "scope",
     "all", "calibrated_shrink", "best_VC", 305, 309, "jhh5_off_position_native/freeze.json"),
    ("JHH-7, 10 µM centered", "Best public adaptation", "jhh_off_position_contrast/JHH7/centered_best.tsv", "scope",
     "all", "calibrated_shrink", "best_VC", 88, 88, "jhh7_off_position_native/freeze.json"),
    ("JHH-7 minus JHH-5", "Zero difference", "jhh_off_position_interaction/score/paired.tsv", "scope", "all_centered",
     "calibrated_shrink", "B0", 86, 86, None),
    ("JHH-5, 3.33333 µM", "Zero response", "jhh_norm_gain_dose_transfer/3.33333/score/paired.tsv",
     "scope", "all", "high_norm_gain", "B0", 111, 112, "jhh_norm_gain_dose_transfer/3.33333/freeze.json"),
    ("JHH-5, 1.11111 µM", "Zero response", "jhh_norm_gain_dose_transfer/1.11111/score/paired.tsv",
     "scope", "all", "high_norm_gain", "B0", 114, 115, "jhh_norm_gain_dose_transfer/1.11111/freeze.json"),
)


def score_rows(data):
    for label, comparator, source, unit, scope, candidate, ref, parents, wells, freeze in TABLE:
        path = data / source
        receipt_path = path.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        digest = sha256(path)
        if receipt["files"][path.name] != digest:
            raise ValueError(f"Score table changed after receipt: {source}")
        if freeze is not None:
            frozen = receipt.get("freeze_sha256") or receipt.get("original_freeze_sha256")
            if frozen != sha256(data / freeze):
                raise ValueError(f"Frozen source differs: {freeze}")
        else:
            for cell in ("JHH5", "JHH7"):
                prior = data / (cell.lower() + "_off_position_native") / "freeze.json"
                if receipt[cell]["native_freeze"] != sha256(prior):
                    raise ValueError("The between-cell score no longer binds the original freezes")
        with path.open(newline="") as handle:
            paired = [r for r in csv.DictReader(handle, delimiter="\t")
                      if (r[unit], r["arm"], r["reference"]) == (scope, candidate, ref)]
        if len(paired) != 2 or {r["metric"] for r in paired} != {"rmse", "r2_pooled"}:
            raise ValueError(f"Expected one recorded row per endpoint: {source}")
        for row in paired:
            n = int(row.get("parents", receipt.get("parents", 0)))
            w = int(row.get("wells", receipt.get("wells", 0)))
            if (n, w) != (parents, wells):
                raise ValueError(f"Support changed for {source}: {(n, w)}")
        m = {r["metric"]: r for r in paired}
        yield dict(evaluation=label, comparator=comparator, parents=parents, wells=wells,
            relative_rmse_gain_pct=100*float(m["rmse"]["relative_gain"]),
            rmse_low99_pct=100*float(m["rmse"]["relative_low99"]),
            rmse_high99_pct=100*float(m["rmse"]["relative_high99"]),
            delta_r2_points=100*float(m["r2_pooled"]["difference"]),
            r2_low99_points=100*float(m["r2_pooled"]["low99"]),
            r2_high99_points=100*float(m["r2_pooled"]["high99"]),
            source=source, score_sha256=digest, receipt_sha256=sha256(receipt_path))


def clean(line, *, references=False):
    def link(m):
        name, target = m.groups()
        if references and target.startswith("https://"):
            return target
        return name if target.startswith("https://") else ""
    return re.sub(r"[ \t]{2,}", " ", LINK.sub(link, line).replace("`", "")).strip(" ·")


def text_length(text):
    return len(re.findall(r"\b[\w.]+(?:[-–][\w.]+)*\b", clean(text)))


def sections(draft):
    parsed = re.split(r"(?m)^## (.+)\n", draft)
    title = parsed[0].splitlines()[0].removeprefix("# ")
    values = dict(zip(parsed[1::2], parsed[2::2], strict=True))
    required = ("Abstract", "Introduction", "Results", "Methods", "Discussion", "Data availability", "References",
                "Table 1", "Figure 1", "Figure 2", "Figure 3")
    if tuple(values) != required:
        raise ValueError("The review manuscript sections or their order changed")
    phrases = [int(m) for m in re.findall(r"\[(\d+)\]", " ".join(values[key] for key in required[:6]))]
    refs = [int(m) for m in re.findall(r"(?m)^(\d+)\. ", values["References"])]
    if sorted(set(phrases)) != refs or list(dict.fromkeys(phrases)) != refs:
        raise ValueError("Numbered citations must follow the order of first mention")
    sizes = dict(title=text_length(title), abstract=text_length(values["Abstract"]),
                 main_text=sum(text_length(values[k]) for k in ("Introduction","Results","Discussion","Data availability")))
    return title, values, sizes


def add_text(doc, text):
    p = doc.add_paragraph()
    for i, fragment in enumerate(re.split(r"\*\*([^*]+)\*\*", text)):
        if fragment:
            p.add_run(fragment).bold = bool(i%2)


def interval(row, point, lo, hi):
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(*(row[k] for k in (point, lo, hi)))


def build_word(path, root, title, contents, rows):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document()
    doc.core_properties.title = title
    doc.core_properties.subject = "Internal review draft"
    doc.core_properties.author = ""
    base = doc.styles["Normal"]
    base.font.name = "Arial"
    base.font.size = Pt(11)
    base.paragraph_format.line_spacing = 1.18
    base.paragraph_format.space_after = Pt(7)
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(1)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._element.append(field)
    doc.add_heading(title, 0)
    doc.add_paragraph("Internal review draft. Author details, declarations and the public release remain pending.")
    with TemporaryDirectory(prefix="vcell-review-figures-") as folder:
        for name, text in contents.items():
            doc.add_heading(name, level=1)
            if name == "Table 1":
                t = doc.add_table(rows=1, cols=5)
                t.style = "Table Grid"
                t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
                for cell, label in zip(t.rows[0].cells, ("Evaluation", "Comparator", "Parents / wells",
                    "RMSE reduction %, 99% interval", "Pooled R² change, points, 99% interval"), strict=True):
                    cell.paragraphs[0].add_run(label).bold = True
                for record in rows:
                    values = (record["evaluation"], record["comparator"], f'{record["parents"]} / {record["wells"]}',
                        interval(record, "relative_rmse_gain_pct", "rmse_low99_pct", "rmse_high99_pct"),
                        interval(record, "delta_r2_points", "r2_low99_points", "r2_high99_points"))
                    for c,value in zip(t.add_row().cells,values,strict=True):
                        c.text = value
            if name.startswith("Figure "):
                n = name.removeprefix("Figure ")
                stub = Path(folder) / f"figure{n}"
                subprocess.run(["pdftoppm", "-r", "200", "-f", "1", "-l", "1", "-png", "-singlefile",
                                str(root/f"figure{n}.pdf"), str(stub)], check=True)
                doc.add_picture(str(stub.with_suffix(".png")), width=Inches(6.35))
            for block in text.strip().split("\n\n"):
                for line in block.splitlines() if name=="References" else (block,):
                    if line.startswith("### "):
                        doc.add_heading(line.removeprefix("### "), level=2)
                    else:
                        add_text(doc, clean(line, references=name=="References"))
        doc.save(path)


def export(root, output, git_revision):
    draft = root / "DRAFT.md"
    text = draft.read_text()
    for name, target in LINK.findall(text):
        if not target.startswith("https://") and not (root/target.split("#",1)[0]).is_file() and target != "table1.tsv":
            raise ValueError(f"Unresolved review source: {name}: {target}")
    title, contents, lengths = sections(text)
    rows = list(score_rows(root.parent/"data"))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".review-") as temp:
        stage = Path(temp)
        with (stage/"table1.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        build_word(stage/"REVIEW.docx", root, title, contents, rows)
        names = ("REVIEW.docx", "table1.tsv")
        receipt = "review_receipt.json"
        write_json(stage/receipt, dict(mode="internal_review", source_git_revision=git_revision,
            draft_sha256=sha256(draft), lengths=lengths, table_rows=len(rows), figures=3,
            vector_sources={f"figure{n}.pdf":sha256(root/f"figure{n}.pdf") for n in (1, 2, 3)},
            artifacts={name:sha256(stage/name) for name in names}))
        for name in (*names, receipt):
            (stage/name).replace(output/name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--git-revision", required=True)
    args = parser.parse_args()
    export(args.manuscript,args.output,args.git_revision)
