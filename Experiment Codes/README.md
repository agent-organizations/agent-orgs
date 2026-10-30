# Complete running code

Every experiment has a runnable Python entrypoint and a notebook that displays the full source used for input construction, parallel inference, parsing and export. The implementation is shared in `src` and `Parallel Agent Setup/src` so the two model families use the same procedure. No implementation is omitted: the complete-code notebooks show those modules verbatim and check their hashes before execution.

| Model | Experiment | Python entrypoint | Full source and execution | Run instructions |
|---|---|---|---|---|
| Mistral | 1a | [Run script](Mistral/Experiment_1a/run_experiment_1a.py) | [Complete-code notebook](Mistral/Experiment_1a/Experiment_1a_Complete_Code.ipynb) | [Instructions](Mistral/Experiment_1a/README.md) |
| Mistral | 2a | [Run script](Mistral/Experiment_2a/run_experiment_2a.py) | [Complete-code notebook](Mistral/Experiment_2a/Experiment_2a_Complete_Code.ipynb) | [Instructions](Mistral/Experiment_2a/README.md) |
| Mistral | 3a | [Run script](Mistral/Experiment_3a/run_experiment_3a.py) | [Complete-code notebook](Mistral/Experiment_3a/Experiment_3a_Complete_Code.ipynb) | [Instructions](Mistral/Experiment_3a/README.md) |
| Qwen | 1b | [Run script](Qwen/Experiment_1b/run_experiment_1b.py) | [Complete-code notebook](Qwen/Experiment_1b/Experiment_1b_Complete_Code.ipynb) | [Instructions](Qwen/Experiment_1b/README.md) |
| Qwen | 2b | [Run script](Qwen/Experiment_2b/run_experiment_2b.py) | [Complete-code notebook](Qwen/Experiment_2b/Experiment_2b_Complete_Code.ipynb) | [Instructions](Qwen/Experiment_2b/README.md) |
| Qwen | 3b | [Run script](Qwen/Experiment_3b/run_experiment_3b.py) | [Complete-code notebook](Qwen/Experiment_3b/Experiment_3b_Complete_Code.ipynb) | [Instructions](Qwen/Experiment_3b/README.md) |

For the participant stage, open [Participant Agent Interaction](Mistral/Participant_Agent_Interaction/README.md): [run script](Mistral/Participant_Agent_Interaction/run_interactions.py) and [complete-code notebook](Mistral/Participant_Agent_Interaction/Participant_Agent_Interaction_Complete_Code.ipynb).

## Start here

```sh
cd "YOUR PATHWAY"
python -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r "Parallel Agent Setup/requirements-cpu.txt"
# Prepare any experiment without downloading model weights or starting a GPU:
python "Experiment Codes/Mistral/Experiment_1a/run_experiment_1a.py"
python "Experiment Codes/Qwen/Experiment_3b/run_experiment_3b.py"
```

To generate responses, restore the models using [Models/README.md](../Models/README.md), install the GPU environment using [Parallel Agent Setup/README.md](../Parallel%20Agent%20Setup/README.md), and start one model family. Then run one experiment:

```sh
python "Parallel Agent Setup/serve.py" --model Mistral --execute
python "Experiment Codes/Mistral/Experiment_1a/run_experiment_1a.py" --run-id mistral_experiment1a_001 --execute
```

The server must be healthy before requests begin. Stop it deliberately before switching to Qwen. Use a new run ID for a new batch. The scripts discover their paths from the repository location; machine-specific paths in examples remain `YOUR PATHWAY`.

## What each stage does

1. **Participant Agent Interaction:** runs the20 article/pair discussions with Mistral participant adapters. Initial answers are generated independently, and ready pairs progress in parallel. A recorded semantic review is required after initial answers and every debate turn; agreement stops immediately, with a20-turn maximum.
2. **Optional manipulation check:** `--with-manipulation-checks` asks200 separate unaided questions before the selected manager experiment. Each question is asked once; the original response is retained for explicit semantic scoring. These checks do not supply articles or force a pass.
3. **Experiments 1a and 1b:** construct the question, relative belief differences and final-message IDs; return own answer, endorsement and rationale.
4. **Experiments 2a and 2b:** add the complete frozen participant interaction; return own answer, endorsement and rationale.
5. **Experiments 3a and 3b:** use the same interaction materials and the base prompt's permitted categories; return own answer, verdict and rationale.
6. **Export:** writes the workbook, raw response text, exact input/output text, token IDs, input/output counts, review flags and per-call timings.

The manager experiments use the20 frozen conversations supplied with the study. New participant runs are saved separately and do not silently replace that evidence. Each manager experiment makes200 independent calls: five articles × four pairs × five managers × two orientations. Speaker labels, beliefs and final-message IDs swap consistently. The swapped call is actually generated; choices and rationales are never copied from the original orientation.

## Full implementation map

| File | Responsibility |
|---|---|
| `src/workbench_protocol.py` | Case construction, participant swaps, distinct instances, source binding and exact input preparation |
| `src/model_tokenization.py` | Native Mistral and Qwen tokenization and context accounting |
| `src/manager_experiment.py` | Per-experiment request batch, raw parsing, choice comparison, spreadsheet and text exports |
| `src/manipulation_checks.py` | Separate one-shot unaided recall calls and scoring records |
| `src/participant_agent_interaction.py` | Initial answers, debate waves, semantic stopping review and transcript export |
| `../Parallel Agent Setup/src/workbench_runtime.py` | Server preflight, adapter registry, admission limits and run locking |
| `../Parallel Agent Setup/src/transport_core.py` | Bounded vLLM streaming, response journals, token checks and timings |
| `training/` | Runnable training-stage entrypoints and retained source lineage |

The exact base prompts are in each experiment's `prompts/Prompt_Verbatim.txt`. Each response is stored as returned by the model. The repository implements new base-procedure runs; historical manager-result workbooks are separate. CPU validation is recorded under `tests`; GPU validation of this new package remains to be performed before collecting new data.
