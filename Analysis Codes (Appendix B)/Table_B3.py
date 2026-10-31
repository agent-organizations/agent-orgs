"""Reproduce Table B3 from the historical outcomes in the locked workbooks.

Run: python Table_B3.py --inputs "YOUR PATHWAY/Final Output/inputs"
All 200 evaluations in each of the six workbooks are checked. The table's
unit is a distinct article/pair debate, so the 60 repeated records of each
debate are counted once. This reproduces the recorded historical coding;
it does not infer correctness from manager choices or reassess transcripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parent
EXHIBIT = "Table_B3"
OUTCOMES = {
    "Correct consensus": "Correct prevailed",
    "False consensus": "Wrong prevailed",
    "Unresolved": "Unresolved",
}
QUESTION_TYPES = (
    "One fact differs (A--B, A--C, A--D)",
    "All three facts differ (A--E)",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_debates(inputs):
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    require(len(files) == 6, "Six locked workbooks are required")
    require(
        {(entry["model"], entry["attempt"]) for entry in files}
        == set(itertools.product(("Mistral", "Qwen"), (1, 2, 3))),
        "Missing or repeated experiment",
    )
    expected_grid = set(
        itertools.product(range(1, 6), ("AB", "AC", "AD", "AE"), "ABCDE", ("original", "swapped"))
    )
    by_debate = defaultdict(list)
    outcome_audit, source_audit = [], []
    for entry in sorted(files, key=lambda item: (item["model"], item["attempt"])):
        path = inputs / "workbooks" / entry["filename"]
        require(sha256(path) == entry["sha256"], f"Locked workbook changed: {path.name}")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(workbook["Results"].values)
        finally:
            workbook.close()
        headers = [str(value).strip() if value is not None else "" for value in values[0]]
        historical_fields = [name for name in headers if name.startswith("Historical interaction outcome")]
        require(len(historical_fields) == 1, f"Ambiguous historical outcome field: {path.name}")
        field = historical_fields[0]
        source_rows = [
            (number, dict(zip(headers, values)))
            for number, values in enumerate(values[1:], 2)
            if values[0] is not None
        ]
        require(len(source_rows) == 200, f"Expected all 200 outcomes: {path.name}")
        require(len({row["Outcome ID"] for _, row in source_rows}) == 200, "Repeated outcome ID")
        grid = {
            (int(row["Article"]), row["Pair"], row["Manager"], row["Orientation"])
            for _, row in source_rows
        }
        require(grid == expected_grid, f"Incomplete article/pair/manager/order grid: {path.name}")
        file_debates = defaultdict(list)
        experiment = str(entry["attempt"]) + ("a" if entry["model"] == "Mistral" else "b")
        for number, row in source_rows:
            article, pair = int(row["Article"]), row["Pair"]
            value = row[field]
            require(isinstance(value, str) and value.strip() in OUTCOMES, f"Unknown historical outcome: {value!r}")
            recorded = value.strip()
            file_debates[article, pair].append(recorded)
            by_debate[article, pair].append(recorded)
            outcome_audit.append({
                "source_file": path.name,
                "source_row": number,
                "experiment": experiment,
                "outcome_id": row["Outcome ID"],
                "article": article,
                "pair": pair,
                "manager": row["Manager"],
                "orientation": row["Orientation"],
                "source_field": field,
                "recorded_outcome": recorded,
            })
        require(len(file_debates) == 20, "Expected 20 debates per workbook")
        require(all(len(records) == 10 and len(set(records)) == 1 for records in file_debates.values()),
                f"Historical outcomes disagree within a debate: {path.name}")
        source_audit.append({
            "experiment": experiment,
            "filename": path.name,
            "sha256": sha256(path),
            "outcomes": len(source_rows),
            "distinct_debates": len(file_debates),
            "historical_outcome_field": field,
        })
    require(len(outcome_audit) == 1200, "Not all source outcomes were retained")
    require(len({row["outcome_id"] for row in outcome_audit}) == 1200, "Cross-file outcome IDs are repeated")
    require(len(by_debate) == 20, "Expected 20 distinct shared debates")
    debates = []
    for (article, pair), records in sorted(by_debate.items()):
        require(len(records) == 60 and len(set(records)) == 1,
                f"Historical outcomes disagree across experiments: article {article}, pair {pair}")
        debates.append({
            "article": article,
            "pair": pair,
            "question_type": QUESTION_TYPES[int(pair == "AE")],
            "recorded_outcome": records[0],
            "table_outcome": OUTCOMES[records[0]],
            "source_records_checked": len(records),
        })
    return debates, outcome_audit, source_audit


def calculate_table(debates):
    rows = []
    for question_type in (*QUESTION_TYPES, "Total"):
        selected = [row for row in debates if question_type == "Total" or row["question_type"] == question_type]
        counts = Counter(row["table_outcome"] for row in selected)
        n = len(selected)
        require(n > 0 and sum(counts.values()) == n, "Invalid denominator")
        result = {"question_type": question_type, "debates": n}
        for label, prefix in (("Correct prevailed", "correct"), ("Wrong prevailed", "wrong"), ("Unresolved", "unresolved")):
            result[prefix + "_count"] = counts[label]
            result[prefix + "_percent"] = 100 * counts[label] / n
        rows.append(result)
    return rows


def percentage(value):
    return f"{value:.1f}".rstrip("0").rstrip(".")


def latex_table(rows):
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\textbf{Table B3. Outcomes of the 20 participant-agent debates.}\par\medskip",
        r"\setstretch{1.05}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X c c c c@{}}",
        r"\toprule",
        r"\textbf{Question type} & \textbf{Debates} & \textbf{\shortstack{Correct\\prevailed}} & \textbf{\shortstack{Wrong\\prevailed}} & \textbf{Unresolved} \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["question_type"], str(row["debates"])]
        for prefix in ("correct", "wrong", "unresolved"):
            cells.append(str(row[prefix + "_count"]) + " (" + percentage(row[prefix + "_percent"]) + r"\%)")
        if row["question_type"] == "Total":
            lines.append(r"\midrule")
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\exnote{Counts (row percentages) summarize the same 20 debates used in all six experiments; correctness refers to the original article. ``Correct/wrong prevailed'' describes the agreed answer.}",
        r"\label{exh:B6}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("YOUR PATHWAY") / "Final Output" / "inputs")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / EXHIBIT)
    args = parser.parse_args()
    debates, outcomes, sources = load_debates(args.inputs)
    rows = calculate_table(debates)
    destination = args.output / EXHIBIT
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / (EXHIBIT + ".csv"), rows)
    (destination / (EXHIBIT + ".tex")).write_text(latex_table(rows), encoding="utf-8")
    write_csv(destination / "debate_audit.csv", debates)
    write_csv(args.output / "outcome_audit.csv", outcomes)
    write_json(args.output / "source_audit.json", {
        "total_outcomes": len(outcomes),
        "distinct_debates": len(debates),
        "records_per_debate": 60,
        "historical_codes_identical_across_all_six_experiments": True,
        "coding_source": "Recorded historical interaction outcome in the locked Results sheets",
        "unit": "One distinct article/participant-pair debate",
        "files": sources,
    })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
