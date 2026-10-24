# Participant Agent Interaction

[`run_interactions.py`](run_interactions.py) is the runnable command-line entrypoint. [`Participant_Agent_Interaction_Complete_Code.ipynb`](Participant_Agent_Interaction_Complete_Code.ipynb) shows the full implementation and runs the same procedure from JupyterLab. Mistral is the participant model; both manager families evaluate the frozen participant transcripts.

## Prepare and start

```sh
cd "YOUR PATHWAY"
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" prepare
# Restore the model and adapters first; see Models/README.md.
python "Parallel Setup/serve.py" --model Mistral --execute
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" advance --run-id participant_interaction_001 --execute
```

Each wave runs ready pairs concurrently. The first wave contains 40 independent initial-answer calls. After that, each pair advances by at most one debate turn per wave. Each participant sees its assigned full article and the discussion so far. Temperature is0.9; the participant adapter scale is0.5. The manager adapters are separate and use0.9.

## Review the stopping condition

After every wave, inspect the complete packets under `runs/<run-id>/data/review_pending/`. Agreement must have the same substantive meaning in every requested component. Shared uncertainty is not agreement. A review is required after initial answers and after each later turn; this judgment is not inferred from matching words alone.

Create a JSON list containing the actual review of each pending pair. Each entry must contain these fields:

```json
{
  "session_id": "article01_AB",
  "agreement": false,
  "determinate": [true, true],
  "component_agreement": [false],
  "rationale": "Replace this with the reason supported by the reviewed messages.",
  "reviewer": "Replace this with the reviewer identifier.",
  "inspected_packet_sha256": "Copy the exact current packet_sha256.",
  "reviewed_debate_turn": 0
}
```

The Boolean values above illustrate the schema; they are not a prefilled decision. Use one `component_agreement` value per requested component (three for A–E), in the packet's component order. Do not continue a pair without making and recording its review.

```sh
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" review --run-id participant_interaction_001 --review-file "YOUR PATHWAY"
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" advance --run-id participant_interaction_001 --execute
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" status --run-id participant_interaction_001
```

Repeat review and advance until all pairs finish. Agreement stops a pair immediately, including before debate or halfway through a round. Persistent disagreement stops at20 debate turns. Reviews are bound to the exact current messages; stale reviews are rejected. A failed request is preserved and requires deliberate recovery, not silent resampling.

## Export

```sh
python "Experiment Codes/Mistral/Participant_Agent_Interaction/run_interactions.py" export --run-id participant_interaction_001
```

The exporter requires all20 sessions to be complete. It writes `sessions.json`, `All_Interaction_Transcripts.txt` and `Participant_Agent_Interaction_Summary.csv` under `deliverables/<run-id>/`. Raw requests, token IDs, token counts and runtimes remain in the run folder. Source-based accuracy requires a separate assessment; a stopping review measures agreement, not truth.

New conversations stay separate from the frozen source records used by the manager attempts. This avoids changing an attempt's evidence merely because a new conversation was generated. The checked-in manager input files remain the supplied20 frozen participant interactions.
