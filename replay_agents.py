"""Replay candidate order using validated model-generated availability reports."""
import argparse
import fcntl
import json
import re
from pathlib import Path
from random import Random
from common import read, require, sha, write
from runner import Backend, cached_call

LABELS = 'ABC'


def make_cases():
    rng = Random(171117)
    slots = [(d, m) for d in range(1, 6) for m in range(480, 1080, 20)]
    rng.shuffle(slots)
    cases = []
    for i in range(16):
        times = []
        for day, minute in slots[3*i:3*i+3]:
            end = minute + 20
            times.append(f'Day {day} {minute//60:02d}:{minute%60:02d}-{end//60:02d}:{end%60:02d}')
        correct = i % 3
        wrong = [j for j in range(3) if j != correct]
        busy = [[] for _ in range(5)]
        busy[i % 5].append(times[wrong[0]])
        busy[(i+1) % 5].append(times[wrong[1]])
        order = [correct, wrong[0], wrong[1]]
        order = order[i % 3:] + order[:i % 3]
        cases.append({'id': f'case-{i+1:02d}', 'times': times, 'busy': busy,
                      'correct': LABELS[correct], 'order': order})
    return cases


def participant_prompt(case, member):
    slots = '\n'.join(f'{LABELS[j]}: {t}' for j, t in enumerate(case['times']))
    busy = '\n'.join(case['busy'][member]) or 'None'
    return (f'Your private calendar lists all your busy intervals below. You are free at all other times.\nBusy intervals:\n{busy}\n'
            f'Candidate meetings:\n{slots}\nFor each candidate, report whether you are available. '
            'Output exactly one line in this format: A available; B unavailable; C available. '
            'Use available or unavailable for each candidate.')


def expected(case, member):
    return '; '.join(f'{label} {"unavailable" if time in case["busy"][member] else "available"}'
                     for label, time in zip(LABELS, case['times'])) + '.'


def leader_prompt(case, reports, reverse):
    order = list(reversed(case['order'])) if reverse else case['order']
    lines = ['Choose the one meeting that all five participants can attend.']
    lines += [f'{LABELS[i]}: {case["times"][i]}' for i in order]
    lines.append('Participant messages:')
    lines += [f'Member{i+1}: {report}' for i, report in enumerate(reports)]
    lines.append('Output exactly one line: FINAL A, FINAL B, or FINAL C.')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'score'])
    p.add_argument('--split', choices=['smoke', 'main'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--max-calls', type=int, default=42)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--token-field', default='max_tokens')
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--extra', type=json.loads, default={'chat_template_kwargs': {'enable_thinking': False}})
    p.add_argument('--mock', action='store_true')
    args = p.parse_args()
    all_cases = make_cases()
    cases = all_cases[:4] if args.split == 'smoke' else all_cases[4:]
    plan = {'schema': 1, 'source': 'synthetic private busy calendars', 'seed': 171117,
            'split': args.split, 'cases': cases, 'script_sha256': sha(Path(__file__).read_bytes())}
    args.out.mkdir(parents=True, exist_ok=True)
    plan_path = args.out / 'plan.json'
    if args.action == 'prepare':
        require(not plan_path.exists(), 'Plan exists; use a new directory.')
        for case in cases:
            assert len(set(case['times'])) == 3
            assert sorted(len(b) for b in case['busy']) == [0, 0, 0, 1, 1]
            assert case['times'][LABELS.index(case['correct'])] not in sum(case['busy'], [])
        write(plan_path, plan)
        print(f'Prepared {len(cases)} synthetic cases.')
        return
    require(read(plan_path) == plan, 'Plan or script changed.')
    if args.action == 'run':
        require(args.max_calls > 0, 'max-calls must be positive')
        backend = Backend(args)
        settings = {'plan_sha256': sha(plan_path.read_bytes()), 'mock': args.mock, **backend.config}
        with (args.out / '.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = args.out / 'manifest.json'
            if manifest.exists():
                require(read(manifest) == settings, 'Run settings changed.')
            else:
                write(manifest, settings)
            try:
                for case in cases:
                    folder = args.out / 'cases' / case['id']
                    reports = [cached_call(folder / f'member-{i+1}.json', backend, participant_prompt(case, i), 'agent')['text'].strip() for i in range(5)]
                    if all(report == expected(case, i) for i, report in enumerate(reports)):
                        for reverse in (False, True):
                            name = 'reverse' if reverse else 'forward'
                            cached_call(folder / f'leader-{name}.json', backend, leader_prompt(case, reports, reverse), 'leader')
            except (RuntimeError, OSError) as error:
                print(f'Incomplete after {backend.count} attempts: {error}. Resume with the same command.')
                return
        print(f'Complete; {backend.count} new calls.')
        return
    manifest = read(args.out / 'manifest.json')
    require(manifest['plan_sha256'] == sha(plan_path.read_bytes()), 'Manifest plan mismatch.')
    counts = dict(reports=0, malformed=0, inaccurate=0, missing_reports=0, eligible=0, leader_missing=0,
                  leader_invalid=0, valid_pairs=0, changed=0, correct_to_wrong=0, wrong_to_correct=0, wrong_to_wrong=0)
    for case in cases:
        folder = args.out / 'cases' / case['id']
        reports = []
        for i in range(5):
            path = folder / f'member-{i+1}.json'
            if not path.exists():
                counts['missing_reports'] += 1
                reports.append(None)
                continue
            record = read(path)
            require(record['prompt_sha256'] == sha(participant_prompt(case, i)), 'Participant prompt mismatch')
            text = record['text'].strip()
            counts['reports'] += 1
            if not re.fullmatch(r'A (available|unavailable); B (available|unavailable); C (available|unavailable)\.', text):
                counts['malformed'] += 1
            elif text != expected(case, i):
                counts['inaccurate'] += 1
            reports.append(text)
        if not all(report == expected(case, i) for i, report in enumerate(reports)):
            continue
        counts['eligible'] += 1
        answers = []
        for reverse in (False, True):
            path = folder / ('leader-reverse.json' if reverse else 'leader-forward.json')
            if not path.exists():
                counts['leader_missing'] += 1
                answers.append(None)
                continue
            record = read(path)
            require(record['prompt_sha256'] == sha(leader_prompt(case, reports, reverse)), 'Leader prompt mismatch')
            text = record['text'].strip()
            if text not in ['FINAL A', 'FINAL B', 'FINAL C']:
                counts['leader_invalid'] += 1
                answers.append(None)
            else:
                answers.append(text)
        if all(answers):
            counts['valid_pairs'] += 1
            a, b = answers
            correct = 'FINAL ' + case['correct']
            if a != b:
                counts['changed'] += 1
                counts['correct_to_wrong' if a == correct else 'wrong_to_correct' if b == correct else 'wrong_to_wrong'] += 1
    print(json.dumps({'planned_cases': len(cases), 'mock': manifest['mock'], **counts}, indent=2))


if __name__ == '__main__':
    main()
