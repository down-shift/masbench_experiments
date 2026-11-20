"""Compare the original and reversed candidate listing runs."""
import argparse
import json
from collections import Counter
from pathlib import Path
from common import read, require, sha, write
import controlled
import controlled_listing

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan', type=Path, required=True)
parser.add_argument('--original', type=Path, required=True)
parser.add_argument('--reversed', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
plan = read(args.plan)
left, right = (read(p / 'manifest.json') for p in (args.original, args.reversed))
require(left['plan_sha256'] == right['plan_sha256'] == sha(args.plan.read_bytes()), 'Plan mismatch')
for key in ('model', 'endpoint', 'temperature', 'max_tokens', 'token_field', 'provider_settings', 'mock'):
    require(left[key] == right[key], f'Run setting differs: {key}')
rows = []
for pair in plan['pairs']:
    for world in ('released', 'alternative'):
        replies = []
        for folder, module in ((args.original, controlled), (args.reversed, controlled_listing)):
            path = folder / 'episodes' / pair['scenario'].replace('/', '-') / world / 'result.json'
            record = read(path)
            require(record['scenario'] == pair['scenario'] and record['world'] == world and record['prompt'] == module.prompt(pair, world), f'Prompt mismatch: {path}')
            replies.append(record['leader'])
        labels = [r['text'].strip() for r in replies]
        rows.append({'scenario': pair['scenario'], 'world': world, 'original': labels[0], 'reversed': labels[1],
                     'correct': 'FINAL ' + 'AB'[pair['candidates'].index(pair['worlds'][world]['correct'])],
                     'fingerprints': [r.get('system_fingerprint') for r in replies]})
counts = Counter((r['original'], r['reversed']) for r in rows)
report = {'worlds': len(rows), 'transitions': {f'{a} -> {b}': n for (a, b), n in sorted(counts.items())}, 'rows': rows}
write(args.out, report)
print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
