# Complete running code

Every attempt has a runnable Python entrypoint and a notebook that displays the full source used for input construction, parallel inference, parsing and export. The implementation is shared in `src` and `Parallel Setup/src` so the two model families use the same procedure. No implementation is omitted: the complete-code notebooks show those modules verbatim and check their hashes before execution.

| Model | Attempt | Python entrypoint | Full source and execution | Run instructions |
|---|---|---|---|---|
| Mistral | 01 | [Run script](Mistral/Attempt_01/run_attempt_01.py) | [Complete-code notebook](Mistral/Attempt_01/Attempt_01_Complete_Code.ipynb) | [Instructions](Mistral/Attempt_01/README.md) |
| Mistral | 02 | [Run script](Mistral/Attempt_02/run_attempt_02.py) | [Complete-code notebook](Mistral/Attempt_02/Attempt_02_Complete_Code.ipynb) | [Instructions](Mistral/Attempt_02/README.md) |
| Mistral | 03 | [Run script](Mistral/Attempt_03/run_attempt_03.py) | [Complete-code notebook](Mistral/Attempt_03/Attempt_03_Complete_Code.ipynb) | [Instructions](Mistral/Attempt_03/README.md) |
| Qwen | 01 | [Run script](Qwen/Attempt_01/run_attempt_01.py) | [Complete-code notebook](Qwen/Attempt_01/Attempt_01_Complete_Code.ipynb) | [Instructions](Qwen/Attempt_01/README.md) |
| Qwen | 02 | [Run script](Qwen/Attempt_02/run_attempt_02.py) | [Complete-code notebook](Qwen/Attempt_02/Attempt_02_Complete_Code.ipynb) | [Instructions](Qwen/Attempt_02/README.md) |
| Qwen | 03 | [Run script](Qwen/Attempt_03/run_attempt_03.py) | [Complete-code notebook](Qwen/Attempt_03/Attempt_03_Complete_Code.ipynb) | [Instructions](Qwen/Attempt_03/README.md) |

For the participant stage, open [Participant Agent Interaction](Mistral/Participant_Agent_Interaction/README.md): [run script](Mistral/Participant_Agent_Interaction/run_interactions.py) and [complete-code notebook](Mistral/Participant_Agent_Interaction/Participant_Agent_Interaction_Complete_Code.ipynb).

## Start here

```sh
cd "YOUR PATHWAY"
python -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r "Parallel Setup/requirements-cpu.txt"
# Prepare any attempt without downloading model weights or starting a GPU:
python "Experiment Codes/Mistral/Attempt_01/run_attempt_01.py"
python "Experiment Codes/Qwen/Attempt_03/run_attempt_03.py"
```

To generate responses, restore the models using [Models/README.md](../Models/README.md), install the GPU environment using [Parallel Setup/README.md](../Parallel%20Setup/README.md), and start one model family. Then run one attempt:

```sh
python "Parallel Setup/serve.py" --model Mistral --execute
python "Experiment Codes/Mistral/Attempt_01/run_attempt_01.py" --run-id mistral_attempt01_001 --execute
```

The server must be healthy before requests begin. Stop it deliberately before switching to Qwen. Use a new run ID for a new batch. The scripts discover their paths from the repository location; machine-specific paths in examples remain `YOUR PATHWAY`.

## What each stage does

1. **Participant Agent Interaction:** runs the20 article/pair discussions with Mistral participant adapters. Initial answers are generated independently, and ready pairs progress in parallel. A recorded semantic review is required after initial answers and every debate turn; agreement stops immediately, with a20-turn maximum.
2. **Optional manipulation check:** `--with-manipulation-checks` asks200 separate unaided questions before the selected manager attempt. Each question is asked once; the original response is retained for explicit semantic scoring. These checks do not supply articles or force a pass.
3. **Attempt01:** constructs the question, relative belief differences and final-message IDs; returns own answer, endorsement and rationale.
4. **Attempt02:** adds the complete frozen participant interaction; returns own answer, endorsement and rationale.
5. **Attempt03:** uses the same interaction materials and the base prompt's permitted categories; returns own answer, verdict and rationale.
6. **Export:** writes the workbook, raw response text, exact input/output text, token IDs, input/output counts, review flags and per-call timings.

The manager attempts use the20 frozen conversations supplied with the study. New participant runs are saved separately and do not silently replace that evidence. Each manager attempt makes200 independent calls: five articles × four pairs × five managers × two orientations. Speaker labels, beliefs and final-message IDs swap consistently. The swapped call is actually generated; choices and rationales are never copied from the original orientation.

## Full implementation map

| File | Responsibility |
|---|---|
| `src/workbench_protocol.py` | Case construction, participant swaps, distinct instances, source binding and exact input preparation |
| `src/model_tokenization.py` | Native Mistral and Qwen tokenization and context accounting |
| `src/manager_experiment.py` | Per-attempt request batch, raw parsing, choice comparison, spreadsheet and text exports |
| `src/manipulation_checks.py` | Separate one-shot unaided recall calls and scoring records |
| `src/participant_agent_interaction.py` | Initial answers, debate waves, semantic stopping review and transcript export |
| `../Parallel Setup/src/workbench_runtime.py` | Server preflight, adapter registry, admission limits and run locking |
| `../Parallel Setup/src/transport_core.py` | Bounded vLLM streaming, response journals, token checks and timings |
| `training/` | Runnable training-stage entrypoints and retained source lineage |

The exact base prompts are in each attempt's `prompts/Prompt_Verbatim.txt`. Each response is stored as returned by the model. The repository implements new base-procedure runs; historical manager-result workbooks are separate. CPU validation is recorded under `tests`; GPU validation of this new package remains to be performed before collecting new data.
