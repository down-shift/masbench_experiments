"""Construct, run, and score the paired two-candidate leader test."""
import argparse
import fcntl
import json
import math
from pathlib import Path

from candidate_audit import audit_plan
from common import ROOT, read, require, sha, write
from runner import Backend, cached_call

WORLDS = ('released', 'alternative')


def prepare(args):
    audit = audit_plan(args.source)
    pairs = []
    for i, item in enumerate(audit['eligible']):
        candidates = [item['released_candidate'], item['alternative_candidate']]
        if i % 2:
            candidates.reverse()
        worlds = {}
        for world in WORLDS:
            correct = item[world + '_candidate']
            messages = []
            for member in range(1, 6):
                availability = [candidate == correct if member == item['holder'] else True for candidate in candidates]
                messages.append({'member': member, 'available': availability})
            assert [all(m['available'][j] for m in messages) for j in range(2)] == [c == correct for c in candidates]
            worlds[world] = {'correct': correct, 'messages': messages}
        pairs.append({'scenario': item['scenario'], 'holder': item['holder'], 'candidates': candidates, 'worlds': worlds})
    plan = {'schema': 1, 'design': 'controlled_leader', 'source_sha256': sha(args.source.read_bytes()),
            'source': str(args.source), 'excluded': audit['excluded'], 'pairs': pairs}
    write(args.out, plan)
    print(f"{len(pairs)} eligible, {len(audit['excluded'])} excluded; saved {args.out}")


def prompt(pair, world):
    candidates = pair['candidates']
    lines = ['Choose the one meeting that all five participants can attend.',
             f"B: {candidates[1]}", f"A: {candidates[0]}", 'Participant messages:']
    for message in pair['worlds'][world]['messages']:
        a, b = ('available' if x else 'unavailable' for x in message['available'])
        lines.append(f"Member{message['member']}: A {a}; B {b}.")
    lines.append('Output exactly one line: FINAL A or FINAL B.')
    return '\n'.join(lines)


def run(args):
    plan = read(args.plan)
    require(plan['design'] == 'controlled_leader', 'Wrong plan design.')
    require(args.max_calls > 0 and args.max_tokens > 0 and args.timeout > 0 and args.retries >= 0, 'Invalid run limits.')
    require(math.isfinite(args.temperature) and 0 <= args.temperature <= 2, 'Invalid temperature.')
    protected = {'model', 'messages', 'temperature', 'max_tokens', 'max_completion_tokens', 'stream'}
    require(isinstance(args.extra, dict) and not protected.intersection(args.extra), 'Invalid extra request settings.')
    require(args.token_field in ('max_tokens', 'max_completion_tokens'), 'Invalid token field.')
    backend = Backend(args)
    args.out.mkdir(parents=True, exist_ok=True)
    implementation = sha(''.join(sha((ROOT / name).read_bytes()) for name in ('common.py', 'runner.py', 'controlled_listing.py')))
    manifest = {'plan_sha256': sha(args.plan.read_bytes()), 'runner_sha256': implementation, 'mock': args.mock, **backend.config}
    with (args.out / '.lock').open('w') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.out / 'manifest.json'
        if path.exists():
            require(read(path) == manifest, 'Run settings changed. Use a new output directory.')
        else:
            write(path, manifest)
        failures = []
        for pair in plan['pairs']:
            for world in WORLDS:
                try:
                    directory = args.out / 'episodes' / pair['scenario'].replace('/', '-') / world
                    text = prompt(pair, world)
                    record = cached_call(directory / 'leader.json', backend, text, 'leader')
                    write(directory / 'result.json', {'scenario': pair['scenario'], 'world': world, 'prompt': text, 'leader': record})
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append({'scenario': pair['scenario'], 'world': world, 'error': str(error)})
        write(args.out / 'last_invocation.json', {'attempts': backend.count, 'failures': failures})
        if failures:
            raise RuntimeError(f'{len(failures)} incomplete calls. Resume with the same settings.')


def score(args):
    plan, manifest = read(args.plan), read(args.run / 'manifest.json')
    require(sha(args.plan.read_bytes()) == manifest['plan_sha256'], 'Run was generated from another plan.')
    rows, missing = [], []
    for pair in plan['pairs']:
        answers = []
        for world in WORLDS:
            path = args.run / 'episodes' / pair['scenario'].replace('/', '-') / world / 'result.json'
            if not path.exists():
                missing.append([pair['scenario'], world]); answers.append(None); continue
            result = read(path)
            require(result['scenario'] == pair['scenario'] and result['world'] == world and result['prompt'] == prompt(pair, world), 'Result input mismatch.')
            text = result['leader']['text'].strip()
            answer = {'FINAL A': 0, 'FINAL B': 1}.get(text)
            correct = pair['candidates'].index(pair['worlds'][world]['correct'])
            answers.append(answer)
            rows.append({'scenario': pair['scenario'], 'world': world, 'answer': answer, 'correct': answer == correct, 'invalid_format': answer is None})
        pair['_answers'] = answers
    require(args.partial or not missing, f'{len(missing)} calls incomplete; use --partial.')
    complete = [p for p in plan['pairs'] if all((args.run / 'episodes' / p['scenario'].replace('/', '-') / w / 'result.json').exists() for w in WORLDS)]
    both = repeated = wrong_switch = 0
    for pair in complete:
        a, b = pair['_answers']
        correct = [pair['candidates'].index(pair['worlds'][w]['correct']) for w in WORLDS]
        both += [a, b] == correct
        repeated += a is not None and a == b
        wrong_switch += a is not None and b is not None and a != b and [a, b] != correct
    report = {'planned_pairs': len(plan['pairs']), 'complete_pairs': len(complete), 'missing': missing,
              'both_worlds_correct': both, 'same_choice_both_worlds': repeated, 'wrong_direction_switch': wrong_switch,
              'invalid_format_worlds': sum(r['invalid_format'] for r in rows), 'correct_worlds': sum(r['correct'] for r in rows),
              'model': manifest['model'], 'mock': manifest['mock']}
    write(args.run / 'scores.json', report); write(args.run / 'world_scores.json', rows)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--source', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.set_defaults(action=prepare)
    p = sub.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--base-url', default='http://127.0.0.1:8000/v1'); p.add_argument('--max-calls', type=int, required=True)
    p.add_argument('--max-tokens', type=int, default=256); p.add_argument('--token-field', default='max_tokens'); p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--timeout', type=int, default=180); p.add_argument('--retries', type=int, default=2); p.add_argument('--extra', type=json.loads, default={}); p.add_argument('--mock', action='store_true'); p.set_defaults(action=run)
    p = sub.add_parser('score'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--run', type=Path, required=True); p.add_argument('--partial', action='store_true'); p.set_defaults(action=score)
    args = parser.parse_args(); args.action(args)


if __name__ == '__main__':
    main()
