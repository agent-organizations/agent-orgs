# Retained training records

This directory contains genuine historical training records, not fabricated rerun logs. It excludes all `run_summary` files and experiment decision/rationale edits. `provenance_manifest.json` binds every retained source file to its original SHA256 and project-relative source label, plus the packaged SHA256. Machine-specific absolute path strings are normalized to `YOUR PATHWAY`; numeric losses, epoch/step indices, times, target counts, selections and validation scores are unchanged. Original research files were not modified.

`source_articles/` contains the 25 exact text files actually tokenized: A original; B magnitude altered; C actor altered; D mechanism altered; E all three altered. These are source-conditioned learning targets and must not all be described as externally true. Per-example source hashes bind QA answers to the appropriate assigned version.

- `Mistral/v8_article`: article inputs/chunks, initialization/final hashes, module audits, 1,100 per-step losses and console records.
- `Mistral/phase10_qa02_03`: 240 training examples across managers, 30 validation examples, 240 per-step losses, validation records and selected checkpoints.
- `Mistral/full_qa01_04_05`: 360 training examples, 75 validation examples, 360 per-step losses, baseline/retention and selected-checkpoint records.
- `Mistral/research_branch_phase09_article02_pilot`: separate earlier pilot, 120 optimizer-step losses; **not** a parent of Phase10.
- `Qwen/article`, `Qwen/qa02_03`, `Qwen/full_qa01_04_05`: 1,100 / 240 / 360 per-step losses, stage summaries and held-out-wording NLL records. Original final-stage filenames use `trained`.
- `source_authoring`: source-key and authored-QA provenance retained from the expanded article dataset.

QA fields are `id, manager, article_id, component, prompt, answer, source_sha256`; original records may include additional provenance fields. Step logs retain the exact schema emitted by their own stage, rather than pretending all historical log schemas were identical. Selection metadata and original registration manifests may refer to historical relative files outside this selected package; the **packaged-file** binding is `provenance_manifest.json`.

See `Experiment Codes/training/README.md` for the stage map, selected epochs, hyperparameters, runnable ports and limits. No optimizer-state resumption files are available. Historical execution does not imply that later curated experiment results were produced unchanged by the base procedure in this repository.
