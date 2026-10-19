#!/usr/bin/env python3
"""Load source datasets and recreate the benchmark records; never loads a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

COUNTS = {'medqa': 1273, 'medmcqa': 4183, 'pubmedqa': 500}
SOURCES = {'medqa': 'openlifescienceai/MedQA-USMLE-4-options-hf',
           'medmcqa': 'openlifescienceai/medmcqa', 'pubmedqa': 'qiaojin/PubMedQA'}
TEST_URL = 'https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json'


def materialize(cache_dir=None):
    from datasets import load_dataset
    provenance = {}
    def source(key, split, config=None):
        ds = load_dataset(SOURCES[key], config, split=split, cache_dir=cache_dir)
        provenance[f'{key}/{split}'] = {'provider': SOURCES[key], 'config': config,
            'split': split, 'rows': len(ds), 'fingerprint': ds._fingerprint}
        return ds
    rows = []
    # Historical MedQA source_index followed the validation split (1,272 rows).
    validation = source('medqa', 'validation')
    if len(validation) != 1272:
        raise ValueError('MedQA validation offset differs from frozen input construction')
    for i, ex in enumerate(source('medqa', 'test')):
        q = str(ex.get('sent1', '')).strip()
        tail = str(ex.get('sent2', '')).strip()
        if tail:
            q += '\n' + tail
        options = {letter: str(ex[f'ending{k}']).strip() for k, letter in enumerate('ABCDE')
                   if ex.get(f'ending{k}') is not None and str(ex[f'ending{k}']).strip()}
        value = str(ex['label']).strip()
        gold = 'ABCDE'[int(value)] if value.isdigit() else value.upper()[0]
        index = 1272 + i
        rows.append({'dataset': 'medqa', 'source_index': index,
            'uid': f'medqa_prof_revision_large_{index}', 'standard_split': 'MedQA official test',
            'question': q, 'options': options, 'gold': gold})
    for i, ex in enumerate(source('medmcqa', 'validation')):
        cop = ex['cop']
        gold = 'ABCD'[cop] if cop in [0, 1, 2, 3] else 'ABCD'[cop-1]
        rows.append({'dataset': 'medmcqa', 'source_index': i, 'uid': f'medmcqa_prof5000_{i}',
            'standard_split': 'MedMCQA validation', 'question': ex['question'],
            'options': dict(zip('ABCD', [str(ex[x]) for x in ['opa','opb','opc','opd']])), 'gold': gold})
    with urllib.request.urlopen(TEST_URL, timeout=120) as response:
        test_bytes = response.read()
    test = json.loads(test_bytes)
    if len(test) != 500:
        raise ValueError('Official PubMedQA test-ID resource must contain 500 records')
    provenance['pubmedqa/test_selector'] = {'url': TEST_URL,
        'sha256': hashlib.sha256(test_bytes).hexdigest(), 'count': len(test)}
    seen = set()
    for i, ex in enumerate(source('pubmedqa', 'train', 'pqa_labeled')):
        pmid = str(ex['pubid'])
        if pmid not in test:
            continue
        seen.add(pmid)
        context = ex.get('context') or {}
        ctx = ' '.join(context.get('contexts', []) or []) if isinstance(context, dict) else str(context)
        label = str(ex['final_decision']).strip().lower()
        if label != test[pmid]:
            raise ValueError('PubMedQA label differs between official and HF sources')
        rows.append({'dataset': 'pubmedqa', 'source_index': i, 'uid': f'pubmedqa_prof5000_{i}',
            'standard_split': 'PubMedQA official test',
            'question': (str(ex['question']) + '\n\nContext: ' + ctx[:2500]).strip(),
            'options': {'A':'yes','B':'no','C':'maybe'}, 'gold': {'yes':'A','no':'B','maybe':'C'}[label]})
    if seen != set(test):
        raise ValueError('PubMedQA test IDs missing from pqa_labeled')
    rows.sort(key=lambda r: (r['dataset'], r['source_index']))
    validate(rows)
    return rows, provenance


def validate(rows):
    if dict(Counter(r['dataset'] for r in rows)) != COUNTS:
        raise ValueError('Dataset counts differ from protocol')
    if len({(r['dataset'],r['source_index'],r['uid']) for r in rows}) != 5956:
        raise ValueError('Duplicate item keys')
    for r in rows:
        if not r['question'] or r['gold'] not in r['options']:
            raise ValueError('Invalid source row')
    if rows != sorted(rows,key=lambda r:(r['dataset'],r['source_index'])):
        raise ValueError('Noncanonical input order')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=Path('data/benchmark_records.jsonl'))
    ap.add_argument('--cache-dir')
    ap.add_argument('--check-only', action='store_true', help='Validate existing local records without network or writes')
    args = ap.parse_args()
    if args.check_only:
        rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
        validate(rows)
    else:
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite existing materialized records')
        rows, provenance = materialize(args.cache_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'counts': COUNTS, 'total': len(rows), 'check_only': args.check_only}))

if __name__ == '__main__':
    main()
