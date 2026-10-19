# Dataset Preparation

| Dataset | External source/config | Selection | N |
|---|---|---|---:|
| MedQA | `openlifescienceai/MedQA-USMLE-4-options-hf`, default | US/USMLE four-option `test` | 1,273 |
| MedMCQA | `openlifescienceai/medmcqa`, default | `validation` | 4,183 |
| PubMedQA | `qiaojin/PubMedQA`, `pqa_labeled` | labeled `train` filtered to official test PMIDs | 500 |

PubMedQA's official selector is fetched from the dataset authors'
[test ground truth resource](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json).
The HF `train` name refers to all 1,000 labeled rows, not the selected evaluation
split. Do not substitute a random 500-row sample.

MedQA: strip `sent1` and append nonempty stripped `sent2` with a newline; map
`ending0`-`ending3` to A-D and zero-based `label` to a letter. Historical source
indices begin at 1,272 because validation preceded test in the original builder.
MedMCQA: use `question`, `opa`-`opd`, and zero-based `cop`; the retained historical
mapper also accepts 4 as D.

PubMedQA: preserve HF source indices; map yes=A, no=B, maybe=C. Construct
`question + "\n\nContext: " + " ".join(contexts)[:2500]` then strip outer
whitespace. Context IS supplied and capped at 2,500 characters. No gold rationale
or long answer is supplied. Rows sort by dataset and source index. Prompt methods
are Direct, CoT, Structured; options are sorted `A. option text` lines. UIDs
retain the historical construction, without distributing the underlying records.

`--check-only` validates saved records without network/writes. Normal preparation
refuses overwrite and records dataset fingerprints and selector hashes locally.
Upstream drift/access restrictions are reproducibility limits, not silently
repaired. Input fields/order were privately checked against the frozen records.
