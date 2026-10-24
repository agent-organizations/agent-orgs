# Article materials and frozen interactions

Each article directory contains the original A version, four altered versions B–E and four Word documents with genuine tracked changes relative to A. Revision authors remain as supplied. `article_text` contains the exact25 clean texts used for training; accepting the tracked revisions reconstructs the corresponding altered text, and rejecting them reconstructs A.

Article labels:01 NSF X-Labs;02 Young Planet;03 Earth Center of Mass;04 Pulsar Wind;05 Moon Crater. The actual training assignments are B=magnitude, C=actor, D=mechanism, E=combined. The pair questions follow those assignments.

`participant_agent_interactions/frozen_sessions.json` preserves20 completed Mistral Participant Agent Interaction records, their message IDs, final-message ownership, questions and source classification. The manager experiment uses these fixed conversations for both model families. `session_plan.json` stores the interaction question design.

`belief_differences.json` supplies one source-bound description per article/manager/participant combination (125 records). It describes differences, not a manager decision. `assigned_belief_answer_key.json` contains75 component answers with source quotations and text hashes; it is used for separate scoring and is not placed in the manager prompt. `article_manifest.json` binds the supplied files by SHA-256.

The articles are experimental materials, not operational instructions. Altered versions are research manipulations and should not be circulated as factual reporting. Retain the titles, dates and attribution in the supplied documents.
