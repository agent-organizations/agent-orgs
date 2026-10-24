# Training lineage and portable entrypoints

These files expose the retained training algorithms and records. Importing `train.py` or `training_port.py` performs no training, model loading, network access, or file writes. A fresh GPU run requires `--execute`; it cannot overwrite the submission package or an existing output directory. No training was executed while preparing this submission.

## Recorded lineage

| Family / package stage | Initialization and data | Executed schedule | Selection |
|---|---|---|---|
| Mistral / `v8_article` | Fresh LoRA; five assigned raw articles per manager | 20 epochs, batch 1; 220 updates per manager | Final epoch 20 |
| Mistral / `phase10_qa02_03` | Exact V8 article adapter; 48 QA rows, Articles 02/03 | 6 epochs, batch 6; 48 updates | Lowest assistant-token NLL at epochs 2/4/6, six held-out-wording questions |
| Mistral / `full_qa01_04_05` | Selected Phase10 adapter; 72 new QA rows, Articles 01/04/05 | 6 epochs, batch 6; 72 updates; no replay of 02/03 | Lowest token-weighted NLL across 15 all-article validation questions at epochs 2/4/6 |
| Qwen / `article` | Five fresh Qwen LoRAs; five assigned articles each | 20 epochs, batch 1; 220 updates | Final epoch 20 |
| Qwen / `qa02_03` | Its own article-stage checkpoint; 48 QA rows | 6 epochs, batch 6; 48 updates | Lowest answer-token NLL at epochs 2/4/6, six validation questions |
| Qwen / `full_qa01_04_05` | Its selected 02/03 checkpoint; 72 new QA rows | 6 epochs, batch 6; 72 updates; no old-example replay | Lowest token-weighted NLL across 15 questions at epochs 2/4/6 |

All minima use the earliest epoch on exact ties. Every QA stage executes all six epochs and restores the selected checkpoint; its last-step loss is not the loss of the selected checkpoint. The original Qwen source names its final stage `trained`; the package path calls it `full_qa01_04_05`.

Selected QA epochs, ordered A/B/C/D/E:

- Mistral Phase10: **6 / 6 / 4 / 6 / 6**.
- Full Mistral: **2 / 2 / 2 / 2 / 2**.
- Qwen 02/03: **6 / 6 / 6 / 6 / 4**.
- Full Qwen: **4 / 6 / 2 / 6 / 6**.

The earlier Phase09 Article02-only pilot is included under `Training logs/Mistral/research_branch_phase09_article02_pilot`. It helped establish the QA format, but its weights were **not** the parent of Phase10. Phase10 restarted from the original article-trained bank. Its retained 120 optimizer-step records are separated from the 3,400 primary-lineage records.

## Numerical and masking details

The pinned bases are `mistralai/Mistral-7B-Instruct-v0.2` at `b70aa86578567ba3301b21c8a27bea4e8f6d6d61` and `Qwen/Qwen-7B-Chat` at `218aa3240fd5a5d1e80bb6c47d5d774361913706`. The Qwen revision is the original 2023 architecture, not Qwen2/2.5. Use the packaged model manifests and locally restored base files; network downloading is not part of these entrypoints.

Both families use rank 16, alpha 16, dropout 0.05, frozen NF4 double-quantized base weights and BF16 computation. Adam uses learning rate 0.0001, betas (0.9, 0.999), epsilon 1e-8 and zero weight decay. No gradient clipping or accumulation is added. Mistral targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` (41,943,040 trainable parameters); Qwen targets `c_attn,c_proj,w1,w2` (35,782,656).

Article inputs are NFC-normalized, encoded without added special tokens and terminated by the model's article EOS. Chunks contain at most 1,025 tokens and advance 1,024 positions: every next-token target is included once per epoch. QA inputs use the native chat boundary. Prompt, system/user/header and padding labels are -100; only assistant answer plus real EOS/ChatML end-of-turn is supervised. Qwen padding/article EOS is 151643, while the supervised assistant end-of-turn is 151645. Full examples must fit the original 512-token cap; there is no truncation.

QA training uses adapter multiplier 1.0 and validation uses 0.9. Stored alpha remains 16. Serving-scaled exports are distinct from unscaled training checkpoints. Article-stage seed is 20230927; QA shuffle/Torch seed is 20261013. The original V8 driver launched some managers in separate concurrent processes; the port runs separate processes sequentially to bound memory, preserving each process's training algorithm. Later stages use separate optimizers/adapters sequentially on one shared frozen base. Equal recipes do not identify a causal model-family effect: architecture, tokenizer, parameter counts and base priors differ.

## Run safely

Restore base files under `Models/Mistral/base` or `Models/Qwen/base`, and retained parent adapters under `Models/<family>/adapters/<stage>/<A..E>/`. Each adapter directory contains `adapter_config.json` and `adapter_model.safetensors`. Training never writes to these retained directories. The parent bank is selected explicitly by stage; new results are written to a separate fresh output tree.

From the repository root, a CPU-only check is:

```sh
python -B 'Experiment Codes/training/train.py' --family Mistral --stage phase10_qa02_03
python -B 'Experiment Codes/training/train.py' --family Qwen --stage staged
python -B 'Experiment Codes/training/test_training_port.py'
```

An explicit future training run, after installing a compatible CUDA environment and restoring model files, is:

```sh
python -B 'Experiment Codes/training/train.py' --family Mistral --stage v8_article --output-root 'YOUR PATHWAY/new-v8-training' --execute
python -B 'Experiment Codes/training/train.py' --family Mistral --stage phase10_qa02_03 --output-root 'YOUR PATHWAY/new-phase10-training' --execute
python -B 'Experiment Codes/training/train.py' --family Mistral --stage full_qa01_04_05 --output-root 'YOUR PATHWAY/new-full-mistral-training' --execute
python -B 'Experiment Codes/training/train.py' --family Qwen --stage staged --output-root 'YOUR PATHWAY/new-staged-qwen-training' --execute
```

These are independently reproducible stage entrypoints: the later Mistral commands read the **retained** parent bank in the package, not the preceding command's new output automatically. To chain newly trained stages, prepare a separate package copy with those parent banks deliberately substituted and register its hashes. The Qwen command runs all three stages within each manager's fresh chain. No commands above have been executed during packaging.

The observed training stack was Torch 2.5.1+cu124, Transformers 4.45.2, PEFT 0.13.2 and bitsandbytes 0.44.1. Full historical dependency snapshots are retained under `Training logs/Mistral/v8_article`. These are provenance, not an automatically installed environment. CPU schema/import tests do not establish compatibility on a new CUDA stack.

## Port changes and limits

`archival_sources/*.py.txt` retain path-normalized original training source, with full-file source hashes and extraction boundaries in `Training logs/provenance_manifest.json`. They are archival text, not entrypoints. The V8 runtime excerpt ends before inference routines; the Phase09/10 excerpts end before downstream final checks. No manager experiment or edited rationale is training data here.

`training_port.py` applies explicit AST transformations only when execution is requested: it injects package-relative paths and file helpers, removes old path-helper imports and direct script entrypoints, and omits generated Mistral validation responses. The **NLL forward calculation, optimizer updates, masking, initialization, checkpoint comparison/restoration and epoch schedules remain the original source functions**. Historical generated validation responses are retained in the logs, but were diagnostics rather than checkpoint selectors. The port emits empty diagnostic call lists and marks that omission where applicable. It does not claim a byte-identical rerun, and has not undergone GPU execution validation.

The raw Qwen B checkpoints contain extra prior-A `lora_B` keys caused by PEFT 0.13.2's adapter-name substring filtering. The original training checkpoint is retained separately from the canonical serving export. The training port deliberately preserves the recorded save algorithm; a new B checkpoint may need the same separately audited canonical-key export. `qwen_serving_export.py.txt` records that export logic; it is not silently applied to raw training artifacts.

No optimizer state, scheduler state, or complete RNG-state checkpoint was retained. These files therefore support fresh training from stage inputs/parents, **not exact interrupted-step resumption**. Unselected QA checkpoints were held transiently in memory and are not asserted to exist. Training/validation examples hold out question wording, not underlying article facts; lower validation loss is not evidence of human validation, factual truth, persistent beliefs, or sound later rationales.
