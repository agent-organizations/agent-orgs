# Analysis Codes (Appendix B)

This folder contains the final analysis programs for Appendix B. It reads the six final experiment workbooks and their reviewed coding files, calculates the table and figure values, and exports CSV, JSON, LaTeX and plot files. It does not train agents or run new manager evaluations.

**Final data are not included.** The `inputs/` files here are sanitized manifest templates and public model descriptions. The complete private input bundle and reference outputs must be supplied separately. A clone of this repository alone cannot reproduce the reported results.

## Contents and exhibit mapping

| File | Analysis or purpose |
|---|---|
| `Complete_Run_Code.py` | Runs all eleven exhibits, checks the six source workbooks and complete 1,200-outcome grid, and compares calculated outputs with supplied private references. |
| `Table_B1b.py` | Assigned-belief manipulation-check matches across the six experiments. |
| `Table_B3.py` | Historical debate outcomes by one-fact versus three-fact differences; counts the 20 distinct debates once. |
| `Figure_B3_1.py` | Manager decisions across belief-match configurations. |
| `Figure_B3_2.py` | Heatmaps of support for A by manager and participant pair. |
| `Figure_B3_3.py` | Support for participants who retained or revised their position. |
| `Table_B3_4.py` | Retained-versus-revised support comparison, with article-clustered inference. |
| `Table_B3_5.py` | Matched within-manager changes in support for A across adjacent experiments. |
| `Figure_B3_6.py` | Support for the participant trained on altered articles, including Both, by manager belief group and debate behavior. |
| `Figure_B3_7.py` | Stated reasons in manager explanations across all six experiments, using the supplied reviewed coding. |
| `Table_B3_8.py` | Summary patterns across model families and experiments. |
| `Table_B3_9.py` | Comparison of the two manager-model families. |
| `NUMBERING_MAP.json` | Mapping from earlier exhibit identifiers to the final Appendix B numbering used by these files. |
| `requirements.txt` | Pinned analysis and image-checking dependencies. |
| `package_manifest.json` | SHA256 hashes of the code, documentation and templates in this release. |
| `inputs/` | Placeholder workbook/checksum manifests and public model descriptions; no final workbooks or row-level coding. |

The experiment mapping is **1a, 2a, 3a = Mistral** and **1b, 2b, 3b = Qwen**. The input schema retains a numeric `attempt` field (1, 2 or 3) alongside `model`; this is how the scripts identify the six experiments.

## Private inputs

Keep the final input bundle outside the clone. A typical layout is:

```text
YOUR PATHWAY/
└── Final Output/
    ├── inputs/
    │   ├── manifest.json
    │   ├── locked_manifest.json
    │   ├── input_checksums.json
    │   ├── workbooks/
    │   │   ├── Final Output_1a.xlsx
    │   │   ├── Final Output_2a.xlsx
    │   │   ├── Final Output_3a.xlsx
    │   │   ├── Final Output_1b.xlsx
    │   │   ├── Final Output_2b.xlsx
    │   │   └── Final Output_3b.xlsx
    │   ├── discussion_codes.csv
    │   ├── participant_behavior_codes.csv
    │   ├── reviewed_answer_codes.json
    │   ├── primary_ledger.json
    │   ├── accuracy_review_ledger.json
    │   ├── qwen_2b_rationale_codes.json
    │   ├── previous_bar_geometry.json
    │   ├── model_descriptions.json
    │   ├── exhibits.json
    │   └── templates/                 # Eleven exhibit .tex templates
    └── verification/
        └── expected_outputs.json     # Private numerical/text/image references
```

The six workbooks must each contain the `Results` sheet with 200 outcomes: five articles × four pairs × five managers × two orientations. Required fields include outcome ID, article, pair, manager, orientation, participant identities, decisions, assigned-belief reference, own answer and rationale; the loaders perform further exhibit-specific checks. Existing approved bundles can keep their original filenames when their private manifests and coding references agree.

`Final Output_1a.xlsx` and the other names in the public templates are aliases. Replace them with the matching local filenames in **both** manifests and in any private ledger `source_file` references. Replace every `YOUR PATHWAY` checksum value with the actual 64-character SHA256 digest of that private file. After changing a manifest, refresh its entry in the private `input_checksums.json`. Do not change the public `package_manifest.json` to point to private inputs.

The six spreadsheets alone are insufficient: behavioral classifications, reviewed answer/rationale coding and LaTeX templates are additional inputs. The programs reuse those supplied classifications; they do not silently create new coding. `expected_outputs.json` is used to check reproduction of the approved numerical results, LaTeX and ledger content. Plot-pixel differences are reported separately because fonts and rendering can vary across environments.

## Run every exhibit

Use Python 3.11 (tested with 3.11.5). From this folder, install the pinned libraries and run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Complete_Run_Code.py \
  --inputs "YOUR PATHWAY/Final Output/inputs" \
  --verification "YOUR PATHWAY/Final Output/verification" \
  --output "YOUR PATHWAY/Appendix B Results"
```

Replace `YOUR PATHWAY` with your own directory. On Windows, activate `.venv\Scripts\activate` instead. The runner can prepare its private `.venv` automatically if the required library versions are missing; it first checks that the private input and reference files are available. `--data-dir` optionally supplies a different directory for the same six hash-verified workbooks.

Each run creates a new timestamped directory containing the eleven exhibit folders, calculated tables and plot data, figures, LaTeX snippets, per-script execution logs, `RUN_REPORT.json` and `OUTPUT_INDEX.txt`. The report records source hashes, versions, runtimes and output comparisons. Nothing is uploaded to Overleaf or GitHub by these scripts.

## Run a single exhibit

```bash
python Figure_B3_2.py \
  --inputs "YOUR PATHWAY/Final Output/inputs" \
  --output "YOUR PATHWAY/Appendix B Results/Figure_B3_2"
```

Replace the script name for another exhibit. Individual scripts use the supplied input checks and do not require the separate reference-output bundle. The combined runner adds the end-to-end comparison against that bundle.

## Release scope

The statistical calculations, scoring functions and study-specific consistency assertions are retained from the supplied final analysis code. Packaging changes make the private input/reference locations configurable, replace identifying manifest values with placeholders, and use a portable plot-cache path. No workbook, row-level coding ledger, final result table, generated plot or expected-output dataset is included. The local source folder remains unchanged.
