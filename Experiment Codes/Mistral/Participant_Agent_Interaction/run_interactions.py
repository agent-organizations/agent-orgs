"""Participant Agent Interaction: prepare, advance, review, status and export.

Twenty pairs run in parallel waves. Within a pair, the next debate turn waits
for an explicit semantic stopping review. This CLI never invents that judgment.
"""
from pathlib import Path
import argparse
import asyncio
import json
import sys

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPOSITORY / "Experiment Codes/src"),
               str(REPOSITORY / "Parallel Setup/src")]
from workbench_protocol import FrozenTokenizer, read_json
from workbench_runtime import validate_run_id
from participant_agent_interaction import (
    new_interaction_sessions, interaction_input, advance_interactions,
    submit_review, export_interactions,
)
WORKSPACE = REPOSITORY / "Experiment Codes/Mistral"


def prepare(run_id):
    tokenizer = FrozenTokenizer(WORKSPACE)
    sessions = new_interaction_sessions(run_id, WORKSPACE)
    sizes = []
    for session in sessions:
        for participant in (1, 2):
            system, user = interaction_input(session, participant, "initial", WORKSPACE)
            _, ids = tokenizer.prepare(system + "\n\n" + user)
            if len(ids) + 384 > 16384:
                raise ValueError("Initial context overflow: " + session["session_id"])
            sizes.append(len(ids))
    print(json.dumps({"sessions": len(sessions), "independent_initial_calls": len(sizes),
        "maximum_initial_input_tokens": max(sizes), "reserved_output_tokens": 384,
        "maximum_debate_turns_per_pair": 20, "inference_executed": False}, indent=2))
    return sessions


def status(run_id):
    validate_run_id(run_id)
    folder = WORKSPACE / "Participant_Agent_Interaction/runs" / run_id
    states = []
    for path in sorted((folder / "data/sessions").glob("*.json")):
        session = read_json(path)
        states.append({k: session.get(k) for k in ("session_id", "status", "debate_turns", "stop_reason")})
    print(json.dumps(states, indent=2))
    return states


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "advance", "review", "status", "export"], nargs="?", default="prepare")
    parser.add_argument("--run-id", default="participant_interaction_001")
    parser.add_argument("--execute", action="store_true", help="Required only to send model requests with advance.")
    parser.add_argument("--review-file", type=Path, help="JSON list of actual judgments; see the local README.")
    args = parser.parse_args(argv)
    validate_run_id(args.run_id)
    if args.action == "prepare":
        prepare(args.run_id)
    elif args.action == "advance":
        if not args.execute:
            parser.error("advance requires --execute to make model calls")
        progress = asyncio.run(advance_interactions(args.run_id, allow_inference=True, root=WORKSPACE))
        print(json.dumps(progress, indent=2))
    elif args.action == "review":
        if args.review_file is None:
            parser.error("review requires --review-file")
        reviews = read_json(args.review_file)
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("Provide a nonempty JSON list of completed review judgments")
        for review in reviews:
            submit_review(args.run_id, root=WORKSPACE, **review)
        print("Saved semantic reviews:", len(reviews))
    elif args.action == "status":
        status(args.run_id)
    elif args.action == "export":
        destination = export_interactions(args.run_id, root=WORKSPACE)
        print("Deliverables:", destination.relative_to(REPOSITORY))


if __name__ == "__main__":
    main()
