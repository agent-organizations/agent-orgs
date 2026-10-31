"""Run all eleven Appendix B analyses using an external private input bundle.

Python 3.11 is required. See README.md for --inputs and --verification.
The repository contains analysis code and placeholder manifests, not final data.
Approved rationale/behaviour coding must be supplied with the private inputs;
no new classification or model inference is performed by this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import venv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXHIBITS = ('Table_B1b', 'Table_B3', 'Figure_B3_1', 'Figure_B3_2', 'Figure_B3_3', 'Table_B3_4', 'Table_B3_5', 'Figure_B3_6', 'Figure_B3_7', 'Table_B3_8', 'Table_B3_9')

VERSIONS = {
    "numpy": "1.24.3", "scipy": "1.11.1",
    "matplotlib": "3.7.2", "openpyxl": "3.0.10", "Pillow": "9.4.0",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_package():
    manifest_path = ROOT / "package_manifest.json"
    require(manifest_path.is_file(), "Keep the published code folder together, including package_manifest.json.")
    files = read_json(manifest_path)["files"]
    for relative, expected in files.items():
        parts = Path(relative).parts
        require(not Path(relative).is_absolute() and ".." not in parts, "Invalid package-relative path.")
        path = ROOT / relative
        require(path.is_file(), f"Required file is missing: {path}")
        require(sha256(path) == expected, f"File differs from the locked version: {path}")
    return files


def check_private_inputs(inputs, verification, data_dir):
    """Reject missing, placeholder, or changed private inputs before creating output."""
    message = "Private final data are not included. Supply --inputs and --verification; see README.md."
    require(inputs.is_dir(), message)
    require((verification / "expected_outputs.json").is_file(), message)
    checksum_path = inputs / "input_checksums.json"
    require(checksum_path.is_file(), message)
    checksums = read_json(checksum_path)
    require(bool(checksums), "The private input checksum manifest is empty.")
    for relative, expected in checksums.items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "Invalid input-relative path.")
        require(isinstance(expected, str) and len(expected) == 64
                and all(c in "0123456789abcdef" for c in expected.lower()),
                "Replace template checksum values with the private bundle's actual SHA256 values.")
        target = inputs / path
        if data_dir is not None and relative.startswith("workbooks/"):
            target = data_dir / path.name
        require(target.is_file(), f"Required private input is missing: {relative}")
        require(sha256(target) == expected, f"Private input differs from its manifest: {relative}")
    require(all((inputs / name).is_file() for name in ("manifest.json", "locked_manifest.json")), message)
    return checksums


def installed_versions():
    values = {}
    for name in VERSIONS:
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def ensure_environment():
    require(sys.version_info[:2] == (3, 11),
            "Use Python 3.11 to run this package (tested with Python 3.11.5).")
    if installed_versions() == VERSIONS:
        return None
    require(os.environ.get("APPENDIX_B_PREPARED") != "1",
            "The private environment does not have the required versions; see the installation output.")
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print("Preparing the required libraries in the package's private .venv ...", flush=True)
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([
        str(python), "-m", "pip", "install", "--disable-pip-version-check",
        "--only-binary=:all:", "-r", str(ROOT / "requirements.txt"),
    ], check=True)
    child = subprocess.run([str(python), str(Path(__file__).resolve()), *sys.argv[1:]],
                           env=dict(os.environ, APPENDIX_B_PREPARED="1"))
    return child.returncode


def check_all_workbooks(inputs, data_dir):
    import openpyxl

    manifest = read_json(inputs / "manifest.json")
    files = manifest["files"]
    require(len(files) == 6, "Exactly six locked workbooks are required.")
    require({(entry["model"], entry["attempt"]) for entry in files}
            == set(itertools.product(("Mistral", "Qwen"), (1, 2, 3))),
            "The source manifest does not contain the six experiments.")
    locked = {entry["filename"]: entry["sha256"]
              for entry in read_json(inputs / "locked_manifest.json")["files"]}
    grid = set(itertools.product(range(1, 6), ("AB", "AC", "AD", "AE"), "ABCDE", ("original", "swapped")))
    audit, all_ids = [], set()
    for entry in files:
        path = (data_dir or inputs / "workbooks") / entry["filename"]
        require(sha256(path) == entry["sha256"] == locked[path.name], f"Source hash mismatch: {path.name}")
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            values = list(workbook["Results"].values)
        finally:
            workbook.close()
        headers = [str(value).strip() if value is not None else "" for value in values[0]]
        rows = [dict(zip(headers, row)) for row in values[1:] if row[0] is not None]
        require(len(rows) == 200, f"Expected all 200 outcomes in {path.name}.")
        ids = {row["Outcome ID"] for row in rows}
        require(len(ids) == 200 and not (ids & all_ids), "Outcome IDs are missing or repeated.")
        all_ids.update(ids)
        observed = {(int(row["Article"]), row["Pair"], row["Manager"], row["Orientation"]) for row in rows}
        require(observed == grid, f"Incomplete experiment grid in {path.name}.")
        require(all(isinstance(row["Manager rationale"], str) and row["Manager rationale"].strip()
                    for row in rows), f"Missing rationale in {path.name}.")
        audit.append(dict(filename=path.name, source=str(path.resolve()), sha256=entry["sha256"],
                          model=entry["model"], attempt=entry["attempt"], outcomes=len(rows)))
    require(len(all_ids) == 1200, "Expected 1,200 source outcomes in total.")
    return audit


def new_output_directory(parent):
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_UTC")
    for number in range(10000):
        target = parent / (stamp if number == 0 else f"{stamp}_{number}")
        try:
            target.mkdir()
            return target
        except FileExistsError:
            continue
    raise RuntimeError("Could not create a fresh output directory.")


def collect_outputs(work, destination, name):
    target = destination / name
    if name == "Figure_B3_7":
        shutil.copytree(work / "results" / name, target)
    else:
        source = work / "results" / name
        shutil.copytree(source / name, target)
        audit = target / "audit"
        audit.mkdir()
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, audit / path.name)


def run_exhibits(destination, inputs, data_dir):
    logs = destination / "logs"
    logs.mkdir()
    completed = []
    with tempfile.TemporaryDirectory(prefix="_working_", dir=destination) as temporary:
        work = Path(temporary)
        shutil.copytree(inputs, work / "inputs")
        (work / "inputs/workbooks").mkdir(exist_ok=True)
        if data_dir is not None:
            for entry in read_json(inputs / "manifest.json")["files"]:
                shutil.copy2(data_dir / entry["filename"], work / "inputs/workbooks" / entry["filename"])
        for name in EXHIBITS:
            shutil.copy2(ROOT / (name + ".py"), work / (name + ".py"))
        (work / "results").mkdir()
        (work / "qa").mkdir()
        for number, name in enumerate(EXHIBITS, 1):
            print(f"[{number}/{len(EXHIBITS)}] {name}", flush=True)
            start = time.monotonic()
            environment = dict(os.environ, MPLCONFIGDIR=str(work / "matplotlib_cache"),
                               PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
            result = subprocess.run([sys.executable, str(work / (name + ".py")),
                                     "--inputs", str(work / "inputs")],
                                    cwd=work, env=environment, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            log = logs / (name + ".log")
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            require(result.returncode == 0, f"{name} failed. See {log}\n{result.stderr[-1500:]}")
            collect_outputs(work, destination, name)
            completed.append(dict(exhibit=name, exit_code=result.returncode,
                                  seconds=round(time.monotonic() - start, 3),
                                  code_sha256=sha256(ROOT / (name + ".py"))))
    return completed


def equivalent(value, expected):
    if value == expected:
        return True
    try:
        actual_number, expected_number = float(value), float(expected)
        return (math.isnan(actual_number) and math.isnan(expected_number)) or math.isclose(
            actual_number, expected_number, rel_tol=1e-10, abs_tol=1e-10)
    except (ValueError, TypeError):
        return False


def check_results(destination, verification):
    from PIL import Image

    reference = read_json(verification / "expected_outputs.json")
    checks = {"csv": [], "text": [], "json": [], "images": []}
    for relative, expected in reference["csv"].items():
        with (destination / relative).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            require(reader.fieldnames == expected["columns"], f"CSV columns changed: {relative}")
        require(len(rows) == len(expected["rows"]), f"CSV row count changed: {relative}")
        for number, (actual, target) in enumerate(zip(rows, expected["rows"]), 1):
            require(all(equivalent(actual[key], target[key]) for key in target),
                    f"Result differs from the approved output: {relative}, row {number}")
        checks["csv"].append(dict(file=relative, rows=len(rows), match=True))
    for kind in ("text", "json"):
        for relative, expected_sha in reference[kind].items():
            require(sha256(destination / relative) == expected_sha, f"Approved content changed: {relative}")
            checks[kind].append(dict(file=relative, match=True))
    for relative, expected in reference["images"].items():
        with Image.open(destination / relative) as image:
            actual = dict(size=list(image.size), rgba_sha256=hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest())
        checks["images"].append(dict(file=relative, pixel_match=(actual == expected), **actual))
    return dict(numerical_results_match=True, latex_and_ledger_match=True,
                all_figure_pixels_match=all(row["pixel_match"] for row in checks["images"]), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("YOUR PATHWAY") / "Final Output" / "inputs",
                        help="Private inputs: workbooks, manifests, coding files and LaTeX templates.")
    parser.add_argument("--verification", type=Path, default=Path("YOUR PATHWAY") / "Final Output" / "verification",
                        help="Private reference directory containing expected_outputs.json.")
    parser.add_argument("--data-dir", type=Path, help="Optional folder containing the same six locked XLSX files.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs", help="Parent directory for a new run folder.")
    args = parser.parse_args()
    inputs = args.inputs.resolve()
    verification = args.verification.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else None
    output = args.output.resolve()
    require(all(output != protected and protected not in output.parents
                for protected in (inputs, verification) + ((data_dir,) if data_dir else ())),
            "Output must be outside the private input and verification directories.")
    verify_package()
    private_checksums = check_private_inputs(inputs, verification, data_dir)
    child_result = ensure_environment()
    if child_result is not None:
        return child_result
    sources = check_all_workbooks(inputs, data_dir)
    print("Verified six locked workbooks: 200 outcomes each, 1,200 total.", flush=True)
    destination = new_output_directory(args.output.resolve())
    report = dict(status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
                  output_directory=str(destination), python=platform.python_version(), platform=platform.platform(),
                  libraries=installed_versions(), sources=sources, source_outcomes=1200, outcomes_per_experiment=200,
                  frozen_coding_loaded_automatically=True, overleaf_modified=False)
    write_json(destination / "RUN_REPORT.json", report)
    try:
        report["runs"] = run_exhibits(destination, inputs, data_dir)
        report["validation"] = check_results(destination, verification)
        verify_package()
        require(check_private_inputs(inputs, verification, data_dir) == private_checksums,
                "The private input checksum manifest changed during execution.")
        report["status"] = "complete"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["original_input_and_code_hashes_unchanged"] = True
        write_json(destination / "RUN_REPORT.json", report)
        lines = ["APPENDIX B — ALL ELEVEN EXHIBITS", "", "Numerical and LaTeX checks: passed.", ""]
        for name in EXHIBITS:
            lines.append(name)
            lines.extend("  " + path.name for path in sorted((destination / name).iterdir()) if path.is_file())
            lines.append("")
        lines += ["RUN_REPORT.json contains source hashes, execution records and output comparisons.",
                  "Table B3 checks all 1,200 source records and counts the 20 distinct debates once.",
                  "Rationale and behaviour analyses reuse the supplied private approved coding."]
        (destination / "OUTPUT_INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nComplete: all eleven exhibits generated and numerical results verified.\nOutputs: {destination}", flush=True)
        if not report["validation"]["all_figure_pixels_match"]:
            print("Figure pixels differ from the validated rendering environment; the reported numbers and LaTeX match. See RUN_REPORT.json.")
        return 0
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(destination / "RUN_REPORT.json", report)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
