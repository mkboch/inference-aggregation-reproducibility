#!/usr/bin/env python3
"""Create local protocol inputs and a pre-generation integrity lock (no inference)."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from materialize_benchmarks import validate

PACKAGE = Path(__file__).resolve().parents[2]
PRIMARY = Path('results/final_protocol_freeze_20260817')
SC = Path('results/final_self_consistency_protocol_freeze_20260831')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=PACKAGE)
    ap.add_argument('--records', type=Path)
    args=ap.parse_args(); root=args.root.resolve()
    source=args.records or root/'data/benchmark_records.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines() if x.strip()];validate(rows)
    for folder in [PRIMARY, SC]:
        if (root/folder).exists():
            raise FileExistsError(f'Refusing to overwrite existing protocol: {folder}')
    # A separate runtime root receives only the immutable code/config dependencies.
    for folder in ['src','prompts','configs','scripts']:
        if root != PACKAGE:
            shutil.copytree(PACKAGE/folder,root/folder,dirs_exist_ok=False)
    for folder in [PRIMARY,SC]: (root/folder).mkdir(parents=True)
    membership=root/PRIMARY/'standard_split_membership.csv'
    with membership.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','source_index','standard_split']);w.writeheader()
        w.writerows({k:r[k] for k in w.fieldnames} for r in rows)
    prompts={m:(root/'prompts'/f'{m}.txt').read_text() for m in ['direct','cot','structured']}
    inputs=[]
    for r in rows:
        options='\n'.join(f'{k}. {v}' for k,v in sorted(r['options'].items()))
        for method,template in prompts.items():
            prompt=template.format(question=r['question'],options=options)
            inputs.append({**{k:r[k] for k in ['dataset','source_index','standard_split','uid','gold']},
                'method':method,'prompt':prompt,'controlled_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    ip=root/PRIMARY/'final_generation_inputs_native_chat.jsonl'
    ip.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inputs))
    mapping={m:{'path':f'prompts/{m}.txt','sha256':sha(root/'prompts'/f'{m}.txt')} for m in prompts}
    for key,path in [('parser',root/'src/evaluation/parser.py'),('standard_membership',membership),('full_generation_inputs',ip)]:
        mapping[key]={'path':str(path.relative_to(root)),'sha256':sha(path)}
    (root/PRIMARY/'final_prompt_parser_hashes.json').write_text(json.dumps(mapping,indent=2)+'\n')
    sc_inputs=[{**r,'sc_source':'frozen_primary_cot_prompt','sc_method':'cot'} for r in inputs if r['method']=='cot']
    sp=root/SC/'sc3_cot_inputs_native_chat.jsonl'
    sp.write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n' for r in sc_inputs))
    settings=json.loads((root/'configs/sc3_generation_settings.json').read_text())
    for model in settings['models']:
        model['runtime_environment']='user-provided environment'
    (root/SC/'sc3_generation_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'local_rerun_freeze':True,
        'items_per_model':5956,'samples_per_item':3,'seeds':[20260831,20260832,20260833],
        'sample_priority':['sample_1','sample_2','sample_3'],
        'cot_prompt_sha256':mapping['cot']['sha256'],'parser_sha256':mapping['parser']['sha256'],
        'inputs_sha256':sha(sp)}
    (root/SC/'sc3_protocol_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files=[p for d in ['src','prompts','configs','scripts',str(PRIMARY),str(SC)] for p in (root/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    lock={'created_at':manifest['created_at'],'scope':'local rerun, not historical preregistration',
        'files':{str(p.relative_to(root)):sha(p) for p in sorted(files)}}
    (root/'results/local_input_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    print(json.dumps({'items':len(rows),'primary_inputs':len(inputs),'sc_inputs':len(sc_inputs),'root':str(root)}))

if __name__=='__main__':main()
