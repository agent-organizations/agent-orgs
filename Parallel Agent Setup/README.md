# Parallel inference

A single vLLM process holds one frozen base model and its LoRA adapters. Each request contains an independent prompt and adapter alias. The client limits active requests and reserves input-plus-output tokens;200 queued jobs do not mean200 full models in GPU memory.

Use a separate Linux/CUDA environment in JupyterLab. Stop any model-owning notebook process deliberately before starting a server. Do not run Mistral and Qwen models concurrently on a single 24GB GPU.

```sh
cd "YOUR PATHWAY"
python -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r "Parallel Agent Setup/requirements-gpu.txt"
pip install -r "Parallel Agent Setup/requirements-cpu.txt"
# Preview the exact command without starting a server:
python "Parallel Agent Setup/serve.py" --model Mistral
# After restoring and verifying weights:
python "Parallel Agent Setup/serve.py" --model Mistral --execute
```

The repository retains the source environment lock files for reference. A fresh install must be tested against its CUDA driver; this packaging task did not start a new server.

**Client concurrency** is the maximum number of generation requests the client keeps active at once. The default of **8** means up to eight requests run simultaneously; additional requests wait until a slot becomes free. For batches with longer contexts, the limit drops to 2 to reduce GPU memory pressure. All requests share one loaded base model and its adapters.

These settings are in each model's `recommended_runtime.json`:

```json
{
  "client_default_concurrency": 8,
  "near_context_limit_validated_concurrency": 2
}
```

See the [Mistral settings](../Experiment%20Codes/Mistral/config/recommended_runtime.json), [Qwen settings](../Experiment%20Codes/Qwen/config/recommended_runtime.json), and `choose_admission()` in [the runtime](src/workbench_runtime.py). Inspect the memory log before increasing concurrency. Prefix caching is disabled, and generation requests have explicit seeds and token IDs.

`serve.py` records its actual command and PID under the selected model workspace's runtime directory. Wait for the health endpoint, then use the notebooks or `Experiment Codes/run.py`. The client checks that all required adapter aliases are registered, journals streamed responses and checks token counts/IDs against the server. Failed or interrupted calls remain records requiring deliberate recovery; restarting a completed batch does not generate a second response.

Runtime files and new experiment outputs are ignored by Git by default. Input/output text, token IDs and timing files are written only by an explicitly executed experiment.
