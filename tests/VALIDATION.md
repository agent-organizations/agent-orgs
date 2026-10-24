# Repository validation

The current package passed19 CPU protocol/package tests and8 CPU training-port tests. Seven command-line entrypoints prepared their full workloads without generation. All12 notebooks executed from top to bottom with inference and training disabled. The seven complete-code notebooks were also checked against the exact bytes and hashes of every source file they display.

Checks cover1,200 manager inputs across both models and three attempts, native token budgets, distinct original/swapped contexts and seeds, consistent label and final-message remapping, exact base-prompt fields, malformed-prompt rejection, stopping reviews, source binding, raw response preservation and exports. A deliberate P1-biased synthetic fixture verifies that independently swapped results are not overwritten to make them agree; it is a software test, not an experiment result.

The training checks preserve3,520 retained loss records and verify the staged training port without executing training. The 400 registered manager definitions reference immutable shared weights; no full model copy is created per row.

**Not executed:** GPU inference or training from this new repository layout. A successful CPU dry run validates input preparation and code paths, not model quality or GPU compatibility. The target Linux/CUDA environment still requires a GPU smoke test before new data collection.

`validation_receipt.json` records counts and context budgets. `Prompt_Completeness_Repair.json` records the restoration of the complete user-supplied Mistral Attempt03 prompt after detecting an incomplete historical source copy. Original experiment records remain unchanged.
