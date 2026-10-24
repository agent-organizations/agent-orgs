# Qwen Attempt 02

Start with [`run_attempt_02.py`](run_attempt_02.py), or open [`Attempt_02_Complete_Code.ipynb`](Attempt_02_Complete_Code.ipynb). The notebook shows the complete source of the input builder, tokenizer, model client, parallel scheduler, response parser, recall check and exporter. Those modules are included in this repository and used by the entrypoint.

This attempt supplies belief differences, question, complete interaction and final-message IDs, then records the manager's own answer, endorsement and rationale. The exact base prompt is in [`prompts/Prompt_Verbatim.txt`](prompts/Prompt_Verbatim.txt). Settings are in [`config.json`](config.json).

## Run

From the repository root, after following [the setup instructions](../../../Parallel%20Setup/README.md):

```sh
cd "YOUR PATHWAY"
# CPU preparation: no server, GPU or generation needed.
python "Experiment Codes/Qwen/Attempt_02/run_attempt_02.py"
# Restore weights and start the Qwen server first.
python "Parallel Setup/serve.py" --model Qwen --execute
# Then generate this attempt only.
python "Experiment Codes/Qwen/Attempt_02/run_attempt_02.py" --run-id qwen_attempt02_001 --execute
```

If a separate memory measurement is required, add `--with-manipulation-checks`. This asks each instance its question once without articles or belief differences, saves its unchanged answer for semantic scoring, and then starts the manager batch in fresh contexts. It is optional and does not impose a pass-until-correct gate.

The runner validates 200 contexts before any generation: five articles, four pairs, five managers and two participant orientations. Each orientation has its own instance, seed, exact input, model request and output. Swapped choices can differ; the comparison sheet reports the difference rather than overwriting it.

## Outputs

New files go to `deliverables/<run-id>/`: `Results.xlsx`, `All_Responses.txt`, `Tokens_and_Runtimes.csv`, `results.json`, `All_Exact_Input_and_Output.txt` and `All_Input_and_Output_Token_IDs.json` (see the exporter for the exact schema). The `runs/<run-id>/` tree retains request records and source hashes. Existing runs are never silently replaced after their inputs change.

Raw answers, choices, rationales, format flags, input/output tokens and timings are retained. Model responses are not substituted with expected answers. A shared GPU base and immutable adapter tensors do not share conversation histories.
