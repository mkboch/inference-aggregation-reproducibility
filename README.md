# Reproducibility Materials for an Inference-Time Aggregation Study

The repository contains the benchmark-materialization code, exact prompt
templates, parser, model identifiers and revisions, generation/serving
configurations, environment specifications, protocol documentation, and
deterministic analysis scripts required to rerun the evaluation pipeline.
Benchmark text, generated model outputs, and derived result files are not
redistributed. Obtain benchmark datasets and model checkpoints from their
original sources under the applicable terms. Model weights are not bundled.

Start with [REPRODUCE.md](REPRODUCE.md). Preparation and validation do not run
inference. GPU generation is a separate, explicitly identified step. No
bitwise identity of stochastic outputs across hardware/software is guaranteed.

- [Dataset preparation](docs/DATASET_MATERIALIZATION.md)
- [Prompt tasks](docs/PROMPTS.md) and [frozen parser](docs/PARSER.md)
- [Aggregation rules](docs/AGGREGATION_RULES.md)
- [Model revisions](configs/model_panel.json)
- [Primary protocol](protocol/PRIMARY_PROTOCOL.md) and [SC-3 protocol](protocol/SC3_PROTOCOL.md)
- [Chronology and its limits](protocol/FREEZE_CHRONOLOGY.md)
- [Portability manifest](protocol/REPRODUCIBILITY_MANIFEST.md)
- [Data availability](docs/DATA_AVAILABILITY.md)

Code is distributed under the [MIT license](LICENSE). Dataset and model licenses
are separate. All generated artifacts stay in ignored local directories.
