"""Create and compare a candidate-order swap for controlled leader runs."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from common import read, require, sha, write
from controlled import WORLDS, prompt


def swapped(plan):
    result = deepcopy(plan)
    for pair in result['pairs']:
        pair['candidates'].reverse()
        for world in WORLDS:
            for message in pair['worlds'][world]['messages']:
                message['available'].reverse()
    return result


def prepare(args):
    original = read(args.plan)
    require(original['design'] == 'controlled_leader', 'Wrong plan design.')
    result = swapped(original)
    result['swapped_from_sha256'] = sha(args.plan.read_bytes())
    require(args.out.resolve() != args.plan.resolve(), 'Use a different output path.')
    write(args.out, result)
    print(f"Created swapped plan with {len(result['pairs'])} pairs at {args.out}")


def compare(args):
    original, variant = read(args.plan), read(args.swapped_plan)
    expected = swapped(original)
    expected['swapped_from_sha256'] = sha(args.plan.read_bytes())
    require(variant == expected, 'Swapped plan does not match original.')
    manifests = [read(folder / 'manifest.json') for folder in (args.run, args.swapped_run)]
    require(manifests[0]['plan_sha256'] == sha(args.plan.read_bytes()) and
            manifests[1]['plan_sha256'] == sha(args.swapped_plan.read_bytes()), 'Plan hash mismatch.')
    keys = ('model', 'endpoint', 'temperature', 'max_tokens', 'token_field', 'provider_settings', 'mock')
    require(all(manifests[0][key] == manifests[1][key] for key in keys), 'Run settings differ.')
    counts, fingerprints, rows = Counter(), set(), []
    for pair, changed in zip(original['pairs'], variant['pairs'], strict=True):
        for world in WORLDS:
            selections = []
            for folder, cell in ((args.run, pair), (args.swapped_run, changed)):
                path = folder / 'episodes' / pair['scenario'].replace('/', '-') / world / 'result.json'
                result = read(path)
                require(result['scenario'] == pair['scenario'] and result['world'] == world and result['prompt'] == prompt(cell, world), f'Input mismatch at {path}')
                leader = result['leader']
                fingerprints.add((leader.get('model'), leader.get('system_fingerprint')))
                label = {'FINAL A': 0, 'FINAL B': 1}.get(leader['text'].strip())
                selections.append(label)
            before, after = selections
            if before is None or after is None:
                category = 'invalid_response'
            elif before == after:
                category = 'same_label'
            else:
                category = 'same_meeting'
            counts[category] += 1
            rows.append({'scenario': pair['scenario'], 'world': world, 'category': category,
                         'original_label': None if before is None else 'AB'[before],
                         'swapped_label': None if after is None else 'AB'[after],
                         'original_correct': before is not None and pair['candidates'][before] == pair['worlds'][world]['correct'],
                         'swapped_correct': after is not None and changed['candidates'][after] == changed['worlds'][world]['correct']})
    report = {'worlds': len(rows), 'same_label': counts['same_label'], 'same_meeting': counts['same_meeting'],
              'invalid_response': counts['invalid_response'], 'original_correct': sum(r['original_correct'] for r in rows),
              'swapped_correct': sum(r['swapped_correct'] for r in rows),
              'observed_models_and_fingerprints': sorted([list(x) for x in fingerprints], key=str), 'rows': rows}
    write(args.out, report)
    print(f"Compared {len(rows)} worlds: {counts['same_label']} same label, {counts['same_meeting']} same meeting, {counts['invalid_response']} invalid. Saved {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.set_defaults(action=prepare)
    p = sub.add_parser('compare'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--swapped-plan', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--swapped-run', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.set_defaults(action=compare)
    args = parser.parse_args(); args.action(args)


if __name__ == '__main__':
    main()
