#!/usr/bin/env python3
"""Non-inference package, local-input integrity, and parser checks."""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime-root',type=Path,default=ROOT)
    ap.add_argument('--prepared',action='store_true')
    args=ap.parse_args();root=args.runtime_root.resolve()
    required=['README.md','REPRODUCE.md','requirements.txt','prompts/direct.txt','prompts/cot.txt','prompts/structured.txt','src/evaluation/parser.py','configs/model_panel.json','configs/primary_generation_settings.json','configs/sc3_generation_settings.json','configs/serving_conditions.json','scripts/data/materialize_benchmarks.py','scripts/data/build_generation_inputs.py','docs/AGGREGATION_RULES.md','docs/PARSER.md','docs/DATASET_MATERIALIZATION.md','protocol/FREEZE_CHRONOLOGY.md']
    for f in required:
        if not (ROOT/f).is_file():raise FileNotFoundError(f)
    pyfiles=[p for folder in ['src','scripts'] for p in (ROOT/folder).rglob('*.py')]
    for p in pyfiles:compile(p.read_text(),str(p),'exec')
    for p in (ROOT/'configs').glob('*.json'):json.loads(p.read_text())
    for p in (ROOT/'scripts').glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in [ROOT/'README.md',ROOT/'REPRODUCE.md',*ROOT.glob('docs/*.md'),*ROOT.glob('protocol/*.md'),*ROOT.glob('environment/*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#') and not (p.parent/link.split('#')[0]).exists():
                raise ValueError(f'Broken local link: {p.name}: {link}')
    spec=importlib.util.spec_from_file_location('checked_parser',ROOT/'src/evaluation/parser.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cases=[('A','A'),('{"answer":"B"}','B'),('```json\n{"answer":"C"}\n```','C'),('{"answer":null}',None),('yes',None),('<think>Answer: B</think>\nC','C')]
    for raw,expected in cases:
        if (mod.parse_output(raw).get('answer') or None)!=expected:raise AssertionError('Parser regression')
    # Config/scientific constants are parsed without importing generation modules.
    panel=json.loads((ROOT/'configs/model_panel.json').read_text())['models']
    primary=json.loads((ROOT/'configs/primary_generation_settings.json').read_text())['models']
    sc=json.loads((ROOT/'configs/sc3_generation_settings.json').read_text())['models']
    nodes=ast.parse((ROOT/'scripts/primary/run_full_native_benchmark.py').read_text()).body
    models=ast.literal_eval(next(n.value for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODELS' for t in n.targets)))
    for row in panel:
        key=row['model_key'];p=next(x for x in primary if x['model_key']==key);s=next(x for x in sc if x['model_key']==key)
        assert models[key]['revision']==row['revision']==p['revision']==s['revision']
        assert s['sc_seeds']==[20260831,20260832,20260833]
        assert {k:v for k,v in p['decoding'].items() if k!='min_p'}==models[key]['decoding']
    checked=0
    if args.prepared:
        lock=json.loads((root/'results/local_input_lock.json').read_text())
        for name,digest in lock['files'].items():
            p=(root/name).resolve()
            if not p.is_relative_to(root):raise ValueError('Lock path escapes runtime root')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError(f'Changed locked input: {name}')
            if name.split('/')[0] in {'src','scripts','configs','prompts'}:
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Executing package differs from prepared source: {name}')
            checked+=1
        for name,n in [('final_protocol_freeze_20260817/final_generation_inputs_native_chat.jsonl',17868),('final_self_consistency_protocol_freeze_20260831/sc3_cot_inputs_native_chat.jsonl',5956)]:
            rows=[json.loads(x) for x in (root/'results'/name).read_text().splitlines()]
            assert len(rows)==n
            assert len({(x['dataset'],x['source_index'],x['uid'],x['method']) for x in rows})==n
            assert all(hashlib.sha256(x['prompt'].encode()).hexdigest()==x['controlled_prompt_sha256'] for x in rows)
    print(json.dumps({'validation':'PASS','python_files':len(pyfiles),'models':len(panel),'parser_tests':len(cases),'local_locked_files':checked,'inference_performed':False}))

if __name__=='__main__':main()
