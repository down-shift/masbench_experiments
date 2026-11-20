"""Matched synthetic availability experiment with three meeting candidates."""
import argparse
import fcntl
import json
from pathlib import Path
from random import Random
from itertools import permutations
from common import read, require, sha, write
from runner import Backend, cached_call

LABELS = 'ABC'
LAYOUTS = ('concentrated', 'distributed')
ORDERS = ('forward', 'reverse')


def cases(split):
    # Separate fixed random streams prevent smoke cases entering the main set.
    rng = Random(1701 if split == 'smoke' else 9403)
    count = 4 if split == 'smoke' else 24
    slots = [(day, minute) for day in range(1, 6) for minute in range(480, 1200, 20)]
    rng.shuffle(slots)
    result = []
    for i in range(count):
        times = []
        for day, minute in slots[3*i:3*i+3]:
            end = minute + 20
            times.append(f'Day {day} {minute//60:02d}:{minute%60:02d}-{end//60:02d}:{end%60:02d}')
        result.append({'id': f'{split}-{i+1:02d}', 'times': times, 'correct': LABELS[i % 3],
                       'first_member': i % 5, 'second_member': (i + 1) % 5,
                       'order': list(list(permutations(range(3)))[(i // 3) % 6])})
    return result


def prompt(case, layout, order):
    correct = LABELS.index(case['correct'])
    wrong = [i for i in range(3) if i != correct]
    availability = [[True] * 3 for _ in range(5)]
    availability[case['first_member']][wrong[0]] = False
    member = case['first_member'] if layout == 'concentrated' else case['second_member']
    availability[member][wrong[1]] = False
    assert [all(row[j] for row in availability) for j in range(3)] == [j == correct for j in range(3)]
    if layout == 'distributed':
        for member in (case['first_member'], case['second_member']):
            assert sum(all(row[j] for k, row in enumerate(availability) if k != member) for j in range(3)) == 2
    indices = case['order'] if order == 'forward' else reversed(case['order'])
    lines = ['Choose the one meeting that all five participants can attend.']
    lines += [f'{LABELS[j]}: {case["times"][j]}' for j in indices]
    lines.append('Participant messages:')
    for i, row in enumerate(availability):
        statements = '; '.join(f'{LABELS[j]} {"available" if row[j] else "unavailable"}' for j in range(3))
        lines.append(f'Member{i+1}: {statements}.')
    lines.append('Output exactly one line: FINAL A, FINAL B, or FINAL C.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'score'))
    parser.add_argument('--split', choices=('smoke', 'main'), default='smoke')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--max-calls', type=int, default=48)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--token-field', default='max_tokens')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--extra', type=json.loads, default={'chat_template_kwargs': {'enable_thinking': False}})
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    plan = cases(args.split)
    # Render all cells during preparation, checking the oracle before inference.
    for case in plan:
        for layout in LAYOUTS:
            for order in ORDERS:
                prompt(case, layout, order)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'cases': plan, 'split': args.split, 'implementation_sha256': sha(Path(__file__).read_bytes())}
    if args.action == 'prepare':
        require(not (args.out / 'plan.json').exists(), 'Plan already exists; preserve existing run.')
        write(args.out / 'plan.json', manifest)
        print(f'Validated {len(plan)} synthetic cases and {4*len(plan)} prompts.')
        return
    require(read(args.out / 'plan.json') == manifest, 'Plan or implementation changed; prepare a new directory.')
    if args.action == 'run':
        backend = Backend(args)
        settings = {'plan_sha256': sha((args.out / 'plan.json').read_bytes()), 'mock': args.mock, **backend.config}
        with (args.out / '.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = args.out / 'manifest.json'
            if path.exists():
                require(read(path) == settings, 'Settings changed; use a new directory.')
            else:
                write(path, settings)
            for case in plan:
                for layout in LAYOUTS:
                    for order in ORDERS:
                        try:
                            cached_call(args.out / 'responses' / case['id'] / layout / (order + '.json'), backend, prompt(case, layout, order), 'leader')
                        except RuntimeError as error:
                            print(f'Stopped after {backend.count} attempts: {error}')
                            return
        print(f'Completed; {backend.count} new attempts.')
        return
    run_manifest = read(args.out / 'manifest.json')
    require(run_manifest['plan_sha256'] == sha((args.out / 'plan.json').read_bytes()), 'Run plan mismatch.')
    paired = {}
    for layout in LAYOUTS:
        valid_pairs = changes = correct = invalid = missing = 0
        for case in plan:
            answers = []
            for order in ORDERS:
                path = args.out / 'responses' / case['id'] / layout / (order + '.json')
                if not path.exists():
                    missing += 1
                    answers.append(None)
                    continue
                record = read(path)
                require(record['prompt_sha256'] == sha(prompt(case, layout, order)), 'Prompt mismatch.')
                answer = record['text'].strip()
                if answer not in [f'FINAL {x}' for x in LABELS]:
                    invalid += 1
                    answers.append(None)
                else:
                    correct += answer == 'FINAL ' + case['correct']
                    answers.append(answer)
            paired[(case['id'], layout)] = None if not all(answers) else int(answers[0] != answers[1])
            if all(answers):
                valid_pairs += 1
                changes += answers[0] != answers[1]
        print(f'{layout}: changed {changes}/{valid_pairs} valid pairs; correct {correct}/{2*len(plan)} planned calls; invalid {invalid}; missing {missing}')
    differences = [paired[(c['id'], 'distributed')] - paired[(c['id'], 'concentrated')] for c in plan if paired[(c['id'], 'distributed')] is not None and paired[(c['id'], 'concentrated')] is not None]
    if len(differences) == len(plan) and not run_manifest['mock']:
        rng = Random(6975)
        draws = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(10000))
        print(f'Paired change-rate difference (distributed minus concentrated): {sum(differences)/len(differences):.3f}; scenario bootstrap 95% interval [{draws[249]:.3f}, {draws[9749]:.3f}].')
    else:
        print(f'Paired contrast withheld: {len(differences)}/{len(plan)} cases have four valid responses, or run is mock.')


if __name__ == '__main__':
    main()
