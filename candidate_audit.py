"""Audit whether paired calendars support a single-holder candidate test."""
import argparse
import json
from pathlib import Path

from audit import oracle
from common import format_slot, read, write


def calendars(pair, world):
    prompts = pair['templates']['original'][world][0]
    result = []
    for prompt in prompts:
        start = prompt.index('{"time_unit"')
        calendar, _ = json.JSONDecoder().raw_decode(prompt[start:])
        result.append(oracle([calendar]))
    return result


def audit_plan(path):
    plan = read(path)
    eligible, excluded = [], []
    for pair in plan['pairs']:
        holder = pair['conditions']['first']['holder'] - 1
        released = calendars(pair, 'released')
        alternative = calendars(pair, 'alternative')
        if any(released[i] != alternative[i] for i in range(5) if i != holder):
            raise ValueError(f"Other participant changed in {pair['scenario']}")
        choices = []
        for a in sorted(pair['oracles']['released']):
            for b in sorted(pair['oracles']['alternative']):
                if (a not in alternative[holder] and b not in released[holder]
                        and all(a in released[i] and b in released[i] for i in range(5) if i != holder)):
                    choices.append((a, b))
        if not choices:
            excluded.append({'scenario': pair['scenario'], 'reason': 'no eligible candidate pair'})
            continue
        a, b = choices[0]
        eligible.append({'scenario': pair['scenario'], 'holder': holder + 1,
                         'released_candidate': format_slot(a), 'alternative_candidate': format_slot(b)})
    return {'source': str(path), 'eligible': eligible, 'excluded': excluded}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit_plan(args.plan)
    write(args.out, report)
    print(f"{len(report['eligible'])} eligible, {len(report['excluded'])} excluded")


if __name__ == '__main__':
    main()
