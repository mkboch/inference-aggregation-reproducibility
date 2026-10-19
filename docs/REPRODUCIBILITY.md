# Reproducibility Scope

Follow [REPRODUCE.md](../REPRODUCE.md). Static validation compiles Python in memory,
parses JSON, checks shell syntax/local documentation links, tests parser examples,
and verifies configuration consistency. Prepared-input validation checks local
hashes, counts, uniqueness and prompt digests. It never loads an LLM.

The local lock detects changes after preparation, not prior intent. Historical
non-model hash literals are not distributed; local hashes are computed before
generation. No generated outputs or results are bundled or promised on request.
Rerunning recomputes the analyses but does not guarantee identical stochastic
outputs across hardware, library versions or interrupted runs.
