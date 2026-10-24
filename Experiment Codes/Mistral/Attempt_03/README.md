# Mistral Attempt 03

Start with [`run_attempt_03.py`](run_attempt_03.py), or open [`Attempt_03_Complete_Code.ipynb`](Attempt_03_Complete_Code.ipynb). The notebook shows the complete source of the input builder, tokenizer, model client, parallel scheduler, response parser, recall check and exporter. Those modules are included in this repository and used by the entrypoint.

This attempt supplies belief differences, question, complete interaction, final-message IDs and permitted verdict categories, then records the manager's own answer, verdict and rationale. The exact base prompt is in [`prompts/Prompt_Verbatim.txt`](prompts/Prompt_Verbatim.txt). Settings are in [`config.json`](config.json).

## Run

From the repository root, after following [the setup instructions](../../../Parallel%20Setup/README.md):

```sh
cd "YOUR PATHWAY"
# CPU preparation: no server, GPU or generation needed.
python "Experiment Codes/Mistral/Attempt_03/run_attempt_03.py"
# Restore weights and start the Mistral server first.
python "Parallel Setup/serve.py" --model Mistral --execute
# Then generate this attempt only.
python "Experiment Codes/Mistral/Attempt_03/run_attempt_03.py" --run-id mistral_attempt03_001 --execute
```

If a separate memory measurement is required, add `--with-manipulation-checks`. This asks each instance its question once without articles or belief differences, saves its unchanged answer for semantic scoring, and then starts the manager batch in fresh contexts. It is an optional, separate measurement.

The runner validates 200 contexts before any generation: five articles, four pairs, five managers and two participant orientations. Each orientation has its own instance, seed, exact input, model request and output. Swapped choices can differ; the comparison sheet reports the difference.

## Outputs

New files go to `deliverables/<run-id>/`: `Results.xlsx`, `All_Responses.txt`, `Tokens_and_Runtimes.csv`, `results.json`, `All_Exact_Input_and_Output.txt` and `All_Input_and_Output_Token_IDs.json` (see the exporter for the exact schema). The `runs/<run-id>/` tree retains request records and source hashes. Existing runs are never silently replaced after their inputs change.

Raw answers, choices, rationales, format flags, input/output tokens and timings are retained. Model responses are stored exactly as returned. A shared GPU base and immutable adapter tensors do not share conversation histories.
