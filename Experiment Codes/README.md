# Experiment entrypoints

`Mistral/Attempt_01` through `Attempt_03` and the corresponding Qwen directories contain the exact base prompts and settings. Shared source code is under `src`; it is used by both model families rather than duplicated implementations.

```sh
cd "YOUR PATHWAY"
python "Experiment Codes/run.py" --model Mistral --attempt 1
python "Experiment Codes/run.py" --model Qwen --attempt 3
# The first two commands only validate inputs. Generation requires --execute:
python "Experiment Codes/run.py" --model Mistral --attempt 1 --run-id new_base_run_01 --execute
```

No prefilled decisions or rationales are included. The parser retains the returned text, records permitted-category/format problems, and exports the raw response. It does not replace a manager's own answer. Results include wide and long layouts, orientation comparison, exact inputs/outputs, token IDs and per-call timings. A new run ID protects an earlier run from changed inputs.

Notebooks cover setup, instances, V8 interaction waves, optional unaided recall and manager attempts. Every training or inference switch is false by default. The V8 notebook requires a recorded semantic review after both initial answers and after each subsequent turn; agreement stops the pair immediately, including halfway through a round. Unresolved discussions stop after20 debate turns. Its new conversations remain separate from the frozen source set used in manager attempts.

`training/README.md` describes the precise training lineage and its runnable entrypoints. Retained numerical training evidence is in `Training logs`.
