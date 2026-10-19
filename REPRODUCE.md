# Rerun the Evaluation Pipeline

Run commands from the repository root. No completed outputs are bundled.

## CPU Preparation (No Inference)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -B scripts/validate_package.py
python -B scripts/data/materialize_benchmarks.py
python -B scripts/data/materialize_benchmarks.py --check-only
python -B scripts/data/build_generation_inputs.py
python -B scripts/validate_package.py --prepared
```

The materializer retrieves external benchmark sources, writing 5,956 records
under ignored `data/`. The builder creates 17,868 primary and 5,956 SC-3 prompt
records, protocol manifests and a local integrity lock under ignored `results/`.
Neither step loads tokenizers/models. Both refuse to overwrite existing inputs.
The lock detects changes after YOUR local preparation; it is not historical
preregistration evidence. Archive source provenance locally because upstream
dataset versions can drift without changing counts.

## GPU Generation (Explicit Inference)

Do not run this section for non-inference validation. Follow the recorded
[generation environment](environment/generation_environment.md), obtain licensed
access to all five pinned checkpoints, and allocate a GPU that fits each model.

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/reproduce_primary.sh
bash scripts/reproduce_sc3.sh
```

Models run serially on logical cuda:0, without quantization or automatic device
mapping. Primary generation seeds once; SC-3 resets each sample seed per item.
An interrupted primary stochastic run does not restore the original RNG stream.
Use fresh outputs for a new rerun. Cross-hardware bitwise identity is not promised.

## Analysis of Completed Outputs (No Inference)

```bash
bash scripts/reproduce_analysis.sh
```

This validates inputs then runs primary accuracy/paired analysis, mechanistic
decomposition, and SC-3 derivation/bootstrap analysis. All completion manifests
must pass. Bootstrap and six-order SC-3 sensitivity analysis may be CPU-intensive.
Tables, reports and per-item data remain under ignored `results/`; do not commit
them. Optional historical serialization-forensic inputs are not bundled or
required for the primary/SC-3 pipeline; an empty forensic summary indicates
absence of those separate experiments, not a fresh experiment.

## Separate Runtime Root

Use `--output /your/runtime/data/benchmark_records.jsonl` when materializing and
`--root /your/runtime` when building inputs. Set `AGGREGATION_REPRO_ROOT` to that
same absolute path before using the wrappers; use `--runtime-root` for validation.
The builder copies required source/config dependencies there once. Do not point
it at an existing frozen scientific directory. Use a fresh directory per rerun.

See [CPU environment](environment/analysis_environment.md) and
[validation scope](docs/REPRODUCIBILITY.md).
