"""Prepare held-out controlled cases and analyze listing-order confirmation."""
import argparse
import fcntl
import math
import json
from contextlib import ExitStack
from collections import Counter
from pathlib import Path
from random import Random

import audit
import controlled
import controlled_listing
import prepare
from candidate_audit import audit_plan
from common import ROOT, read, require, sha, write
from runner import Backend, cached_call


def pool(args):
    used = set()
    for path in args.exclude:
        used.update(pair['scenario'] for pair in read(path)['pairs'])
    rows = read(ROOT / 'data/cases.json')
    eligible = [r for r in rows if 'counterfactual_answers' in r and not any(r['unexplained_busy_units_by_member'])]
    remaining = [r for r in eligible if r['scenario'] not in used]
    return rows, eligible, remaining, used


def build(args):
    rows, eligible, remaining, used = pool(args)
    print(f'{len(rows)} audited cases; {len(eligible)} eligible for reconstruction; {len(used)} excluded IDs; {len(remaining)} remaining')
    if args.audit_only:
        return
    require(args.repo is not None and args.out is not None, 'Provide --repo and --out.')
    require(args.pairs > 0 and args.pairs <= len(remaining), 'Invalid pair count.')
    require(not args.out.exists(), 'Output already exists.')
    module, runner_hash = prepare.upstream(args.repo.resolve())
    verified = []
    excluded = []
    for row in remaining:
        pair = prepare.make_pair(args.repo, module, row, 'order', 1, True)
        holder = pair['conditions']['first']['holder'] - 1
        from candidate_audit import calendars
        released = calendars(pair, 'released')
        alternative = calendars(pair, 'alternative')
        require(all(released[i] == alternative[i] for i in range(5) if i != holder), 'Unexpected calendar change.')
        choices = [(a, b) for a in pair['oracles']['released'] for b in pair['oracles']['alternative']
                   if a not in alternative[holder] and b not in released[holder]
                   and all(a in released[i] and b in released[i] for i in range(5) if i != holder)]
        if choices:
            verified.append((row, pair))
        else:
            excluded.append(row['scenario'])
    print(f'{len(verified)} meet the single-holder candidate rule; {len(excluded)} excluded')
    require(args.pairs <= len(verified), 'Not enough verified cases.')
    selected = prepare.stratified([row for row, _ in verified], args.pairs, args.seed)
    by_id = {row['scenario']: pair for row, pair in verified}
    source = {'schema': 1, 'design': 'order', 'split': 'confirmation', 'sample_seed': args.seed,
              'repo_commit': audit.PIN, 'upstream_runner_sha256': runner_hash,
              'audit_cases_sha256': sha((ROOT / 'data/cases.json').read_bytes()),
              'exclusion_plan_sha256': [sha(p.read_bytes()) for p in args.exclude],
              'eligible_count': len(verified), 'ineligible_scenarios': excluded,
              'pairs': [by_id[row['scenario']] for row in selected]}
    source_path = args.out.with_suffix('.source.json')
    require(not source_path.exists(), 'Source output already exists.')
    write(source_path, source)
    controlled.prepare(argparse.Namespace(source=source_path, out=args.out))



def run(args):
    plan = read(args.plan)
    require(plan['design'] == 'controlled_leader', 'Wrong plan design.')
    require(args.max_calls > 0 and args.max_tokens > 0 and args.timeout > 0 and args.retries >= 0, 'Invalid limits.')
    require(math.isfinite(args.temperature) and 0 <= args.temperature <= 2, 'Invalid temperature.')
    protected = {'model', 'messages', 'temperature', 'max_tokens', 'max_completion_tokens', 'stream'}
    require(isinstance(args.extra, dict) and not protected.intersection(args.extra), 'Invalid extra settings.')
    require(args.original.resolve() != args.reversed.resolve(), 'Use separate run directories.')
    backend = Backend(args)
    folders = (args.original, args.reversed)
    modules = (controlled, controlled_listing)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for folder in sorted(folders, key=lambda p: str(p.resolve())):
            lock = stack.enter_context((folder / '.lock').open('w'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder, module in zip(folders, modules, strict=True):
            implementation = sha(''.join(sha((ROOT / name).read_bytes()) for name in ('common.py', 'runner.py', module.__name__ + '.py')))
            manifest = {'plan_sha256': sha(args.plan.read_bytes()), 'runner_sha256': implementation, 'mock': args.mock, **backend.config}
            path = folder / 'manifest.json'
            if path.exists():
                require(read(path) == manifest, f'Run settings changed at {folder}.')
            else:
                write(path, manifest)
        failures = []
        for pair in plan['pairs']:
            for world in ('released', 'alternative'):
                for folder, module in zip(folders, modules, strict=True):
                    try:
                        directory = folder / 'episodes' / pair['scenario'].replace('/', '-') / world
                        text = module.prompt(pair, world)
                        record = cached_call(directory / 'leader.json', backend, text, 'leader')
                        write(directory / 'result.json', {'scenario': pair['scenario'], 'world': world, 'prompt': text, 'leader': record})
                    except (OSError, RuntimeError, ValueError) as error:
                        failures.append({'scenario': pair['scenario'], 'world': world, 'run': str(folder), 'error': str(error)})
        for folder in folders:
            write(folder / 'last_invocation.json', {'attempts': backend.count, 'failures': failures})
        if failures:
            raise RuntimeError(f'{len(failures)} calls incomplete. Resume with the same command.')
        print(f'Completed {len(plan["pairs"])} scenarios; {backend.count} attempts this invocation.')


def analyze(args):
    plan = read(args.plan)
    manifests = [read(d / 'manifest.json') for d in (args.original, args.reversed)]
    require(all(m['plan_sha256'] == sha(args.plan.read_bytes()) for m in manifests), 'Plan mismatch.')
    for key in ('model', 'endpoint', 'temperature', 'max_tokens', 'token_field', 'provider_settings', 'mock'):
        require(manifests[0][key] == manifests[1][key], f'Setting mismatch: {key}')
    require(not manifests[0]['mock'], 'Mock runs cannot be confirmation results.')
    counts, transitions, fingerprints, differences = Counter(), Counter(), set(), []
    rows = []
    for pair in plan['pairs']:
        delta = []
        for world in ('released', 'alternative'):
            labels = []
            for folder, mod in ((args.original, controlled), (args.reversed, controlled_listing)):
                result = read(folder / 'episodes' / pair['scenario'].replace('/', '-') / world / 'result.json')
                require(result['scenario'] == pair['scenario'] and result['world'] == world and result['prompt'] == mod.prompt(pair, world), 'Result mismatch.')
                record = result['leader']; labels.append(record['text'].strip())
                fingerprints.add((record.get('model'), record.get('system_fingerprint')))
            correct = 'FINAL ' + 'AB'[pair['candidates'].index(pair['worlds'][world]['correct'])]
            a, b = (label == correct for label in labels)
            counts[('correct' if a else 'incorrect') + '_' + ('correct' if b else 'incorrect')] += 1
            transitions[labels[0] + ' -> ' + labels[1]] += 1
            delta.append(int(b) - int(a))
            rows.append({'scenario': pair['scenario'], 'world': world, 'original': labels[0], 'reversed': labels[1], 'correct': correct})
        differences.append(sum(delta) / 2)
    require(differences, 'No scenarios.')
    rng = Random(6975); n = len(differences)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(10000))
    report = {'scenarios': n, 'worlds': len(rows), 'paired_world_counts': dict(counts), 'transitions': dict(transitions),
              'reversed_minus_original_accuracy_pp': 100 * sum(differences) / n,
              'scenario_bootstrap_95_pp': [100 * draws[249], 100 * draws[9749]],
              'invalid_original': sum(r['original'] not in ('FINAL A', 'FINAL B') for r in rows),
              'invalid_reversed': sum(r['reversed'] not in ('FINAL A', 'FINAL B') for r in rows),
              'observed_models_and_fingerprints': sorted([list(x) for x in fingerprints], key=str), 'rows': rows}
    write(args.out, report)
    print({k: v for k, v in report.items() if k != 'rows'})


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(required=True)
    p = sub.add_parser('prepare'); p.add_argument('--exclude', type=Path, nargs='+', required=True); p.add_argument('--repo', type=Path); p.add_argument('--out', type=Path); p.add_argument('--pairs', type=int, default=32); p.add_argument('--seed', type=int, default=9401); p.add_argument('--audit-only', action='store_true'); p.set_defaults(action=build)
    p = sub.add_parser('run'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--original', type=Path, required=True); p.add_argument('--reversed', type=Path, required=True); p.add_argument('--model', required=True); p.add_argument('--base-url', default='http://127.0.0.1:8000/v1'); p.add_argument('--max-calls', type=int, required=True); p.add_argument('--max-tokens', type=int, default=256); p.add_argument('--token-field', choices=['max_tokens', 'max_completion_tokens'], default='max_tokens'); p.add_argument('--temperature', type=float, default=0); p.add_argument('--timeout', type=int, default=180); p.add_argument('--retries', type=int, default=2); p.add_argument('--extra', type=json.loads, default={}); p.add_argument('--mock', action='store_true'); p.set_defaults(action=run)
    p = sub.add_parser('analyze'); p.add_argument('--plan', type=Path, required=True); p.add_argument('--original', type=Path, required=True); p.add_argument('--reversed', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.set_defaults(action=analyze)
    args = parser.parse_args(); args.action(args)


if __name__ == '__main__':
    main()
