# Parallel inference

A single vLLM process holds one frozen base model and its LoRA adapters. Each request contains an independent prompt and adapter alias. The client limits active requests and reserves input-plus-output tokens;200 queued jobs do not mean200 full models in GPU memory.

Use a separate Linux/CUDA environment in JupyterLab. Stop any model-owning notebook process deliberately before starting a server. Do not run Mistral and Qwen concurrently on a single24GB GPU.

```sh
cd "YOUR PATHWAY"
python -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -r "Parallel Setup/requirements-gpu.txt"
pip install -r "Parallel Setup/requirements-cpu.txt"
# Preview the exact command without starting a server:
python "Parallel Setup/serve.py" --model Mistral
# After restoring and verifying weights:
python "Parallel Setup/serve.py" --model Mistral --execute
```

The repository retains the source environment lock files for reference. A fresh install must be tested against its CUDA driver; this packaging task did not start a new server. Start with client concurrency8 and inspect the memory log before increasing it. Long requests are limited to2. Prefix caching is disabled, and generation requests have explicit seeds and token IDs.

`serve.py` records its actual command and PID under the selected model workspace's runtime directory. Wait for the health endpoint, then use the notebooks or `Experiment Codes/run.py`. The client checks that all required adapter aliases are registered, journals streamed responses and checks token counts/IDs against the server. Failed or interrupted calls remain records requiring deliberate recovery; restarting a completed batch does not generate a second response.

Runtime files and new experiment outputs are ignored by Git by default. Input/output text, token IDs and timing files are written only by an explicitly executed experiment.
