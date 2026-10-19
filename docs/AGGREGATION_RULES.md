# Aggregation Rules

There are three original slots. Count recovered A-E answers, requiring at least
TWO matching slots for strict majority; never redefine majority relative only
to the recovered subset.

| State | Selection |
|---|---|
| Three agree | That answer |
| Two agree; third differs or is unrecovered | Two-slot majority |
| Two recovered answers disagree | First recovered answer in priority |
| Three different recovered answers | First recovered answer in priority |
| One recovered answer | That answer via fallback |
| None recovered | Empty, unrecovered, incorrect for end-to-end accuracy |

Fusion priority: Direct > CoT > Structured. SC-3: sample_1 > sample_2 > sample_3.
All six priority permutations are sensitivity analyses. SPACE/confidence do not
weight or veto votes. Parser acceptance is A-E globally; a recovered letter
outside a dataset's actual options remains incorrect. No optimality rationale
is claimed for the primary deterministic order. Implementations are
`fuse_from_answers` and `derive_sc3` in the respective analysis scripts.
