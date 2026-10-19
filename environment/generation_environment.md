# Generation Environment

Durable per-generation records report Python 3.12.3, PyTorch 2.11.0+cu130,
Transformers 5.15.0 and CUDA build 13.0. These are historical recorded versions,
not inferred from the current audit environment. A full transitive environment
lock, GPU model and exact driver version were not recorded in the inspected
generation manifests. Logical GPU indices do not identify physical hardware.

Use a separate Python 3.12 environment, install recorded versions from appropriate
package indexes (subject to availability/driver support), then CPU requirements:

```bash
python3.12 -m venv .venv_generation
source .venv_generation/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install transformers==5.15.0
python -m pip install -r requirements.txt
```

Pinned checkpoints require upstream license/access. Qwen3.6/Gemma4/MedGemma use
AutoModelForImageTextToText; Qwen2.5/Phi use AutoModelForCausalLM. Trust flags are
in configs. Models load bfloat16 without quantization/device mapping and must fit
on logical cuda:0. Context=8192, output cap=4096; native chat uses
apply_chat_template(tokenize=False, add_generation_prompt=True).

Only Qwen3.6 explicitly passes enable_thinking=False. Gemma4 passes no toggle;
zero observed markers would not prove thinking disabled. top_k=null means no
explicit argument, NOT top_k=0. Neither runner passes min_p, despite its presence
in config metadata; omitted values inherit pinned checkpoint/library defaults.
Presence penalty applies to generated tokens only. Primary seeds once, SC-3 per
item/sample; no cross-hardware or interrupted-run bitwise guarantee is made.
