"""Run Qwen Experiment 1b from input construction through spreadsheet export.

Default: validate the complete 200-call batch on CPU, without model generation.
With --execute: send 200 independent requests through the local vLLM server.
Each original and swapped orientation uses a distinct request context and seed.
Both generated responses are retained independently.
"""
from pathlib import Path
import argparse
import asyncio
import json
import sys

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPOSITORY / "Experiment Codes/src"),
               str(REPOSITORY / "Parallel Agent Setup/src")]
from workbench_protocol import prepare_inputs, validate_cases
from manager_experiment import run_experiment
from manipulation_checks import prepare_checks, run_checks

FAMILY = "Qwen"
EXPERIMENT = "1b"
WORKSPACE = REPOSITORY / "Experiment Codes" / FAMILY


def prepare(with_manipulation_checks=False):
    """Build and tokenize every full input; validate identities and context limits."""
    cases = prepare_inputs(EXPERIMENT, WORKSPACE, save=False)
    validate_cases(cases)
    config = json.loads((WORKSPACE / f"Experiment_{EXPERIMENT}/config.json").read_text())
    report = {
        "model": FAMILY, "experiment": EXPERIMENT, "manager_calls": len(cases),
        "independent_original_calls": sum(c["orientation"] == "original" for c in cases),
        "independent_swapped_calls": sum(c["orientation"] == "swapped" for c in cases),
        "maximum_input_tokens": max(c["input_tokens"] for c in cases),
        "reserved_output_tokens": config["max_output_tokens"],
        "manipulation_check_calls": len(prepare_checks(WORKSPACE)) if with_manipulation_checks else 0,
        "inference_executed": False,
    }
    print(json.dumps(report, indent=2))
    return cases


async def execute(run_id, *, with_manipulation_checks=False):
    """Optional one-shot recall, followed by parallel manager generation and export.

    Recall runs in separate contexts and is not inserted into manager prompts.
    The first response from each recall call is saved for separate semantic
    scoring.
    """
    if with_manipulation_checks:
        checks = await run_checks(run_id + "_recall", allow_inference=True, root=WORKSPACE)
        print("Unaided responses retained for semantic scoring:", len(checks))
    rows, timing = await run_experiment(EXPERIMENT, run_id, allow_inference=True, root=WORKSPACE)
    print("Generated manager responses:", len(rows))
    print("Batch wall seconds:", timing.get("elapsed_seconds"))
    print("Deliverables:", str((WORKSPACE / f"Experiment_{EXPERIMENT}/deliverables" / run_id).relative_to(REPOSITORY)))
    return rows, timing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="A new identifier for a new batch; required with --execute.")
    parser.add_argument("--execute", action="store_true", help="Generate with the restored local vLLM server.")
    parser.add_argument("--with-manipulation-checks", action="store_true", help="Add 200 separate one-shot unaided checks before the manager batch.")
    args = parser.parse_args(argv)
    if args.execute and not args.run_id:
        parser.error("--execute requires --run-id; use a new ID for a new experiment.")
    prepare(args.with_manipulation_checks)
    if args.execute:
        asyncio.run(execute(args.run_id, with_manipulation_checks=args.with_manipulation_checks))


if __name__ == "__main__":
    main()
