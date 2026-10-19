# Primary Protocol

The five fixed revisions are in [model_panel.json](../configs/model_panel.json).
MedQA official test (1,273), MedMCQA validation (4,183), and PubMedQA official test
(500) give 5,956 items. Each receives Direct, CoT and Structured under native_chat.
Fusion is derived, not an extra generation. Accuracy includes unrecovered outputs
as incorrect; macro averages the three datasets equally.

See [settings](../configs/primary_generation_settings.json),
[serving](../configs/serving_conditions.json), [parser](../docs/PARSER.md) and
[aggregation](../docs/AGGREGATION_RULES.md). Seed=20260816; context bound=8192;
output cap=4096. Majority needs two slots; fallback Direct > CoT > Structured.
All six fallback permutations are sensitivity analyses.

Primary paired bootstrap B=2,000, base seed=20260830, paired within-dataset
resampling and equal dataset weights. Five original binary Fusion-vs-CoT pooled
McNemar tests receive Holm adjustment in the revision-evidence analyzer; pooled
tests are not tests of the macro estimand. See [chronology](FREEZE_CHRONOLOGY.md).
