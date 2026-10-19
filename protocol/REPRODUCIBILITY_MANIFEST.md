# Portability Manifest

Exact prompt/parser bytes, model revisions, decoding, votes, fallback rules,
statistical functions and metrics are retained. Changes in this public package:

- Build source-dataset records and required protocol intermediates locally.
- Omit published historical non-model hash literals; validate against a local
  pre-generation input lock instead.
- Hash the actual analysis script path, not an assumed copy inside its outputs;
  metadata records the rerun time.
- Accept a user-selected CUDA allocation instead of fixed device indices;
  generation still uses bfloat16 on logical cuda:0.
- Derive report prose from computed estimates rather than hard-coded outcomes.

Static validation does not run inference. Generated data/results stay ignored.
The local lock guards changes since preparation, not historical preregistration.
