"""CPU-only, deterministic follow-up on the checked-in MASBench calendars."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

PIN = "c0301a650db95958924bb080192c09fef39389a5"


def start(duration, day):
    h, m = map(int, duration.split(":"))
    assert m % 10 == 0
    return (day - 1) * 144 + h * 6 + m // 10


def interval(activity):
    begin, end = (start(s, activity["date"]) for s in activity["duration"].split("-"))
    return begin, end if end > begin else end + 144


def oracle(calendars):
    occupied = {t for c in calendars for a in c["activities"] for t in range(*interval(a))}
    return {t for t in range(719) if t not in occupied and t + 1 not in occupied}


def read(path):
    return json.loads(path.read_text())


def novel_entries(original, final):
    def key(activity):
        return (activity["date"], activity["duration"], activity["description"])

    remaining = Counter(map(key, original["activities"]))
    indices = []
    for index, activity in enumerate(final["activities"]):
        k = key(activity)
        if remaining[k]:
            remaining[k] -= 1
        else:
            indices.append(index)
    return indices


def alternative(base, official, budget):
    """Find exactly budget blocks and len(official) surviving starts disjoint from official.

    Each formerly feasible start is an edge between adjacent calendar units.
    A block deletes its incident edges. Only endpoints of these edges are eligible.
    The state tracks whether the preceding unit survived, survivors, and blocks.
    """
    states = {(False, 0, 0): ()}
    last = -2
    for t in sorted(base | {s + 1 for s in base}):
        next_states = {}
        for (previous_open, survivors, blocks), path in sorted(states.items()):
            if blocks < budget:
                next_states.setdefault((False, survivors, blocks + 1), path + (t,))
            retains_edge = previous_open and last == t - 1 and last in base
            if retains_edge and last in official:
                continue
            count = survivors + int(retains_edge)
            if count <= len(official):
                next_states.setdefault((True, count, blocks), path)
        states, last = next_states, t
    return next((path for (opened, count, blocks), path in sorted(states.items())
                 if count == len(official) and blocks == budget), None)


def replay_blocks(calendars, target, templates, units):
    copy = deepcopy(calendars)
    for template, t in zip(templates, units, strict=True):
        def clock(unit):
            minute = (unit % 144) * 10
            return f"{minute // 60:02}:{minute % 60:02}"
        date = t // 144 + 1
        entry = dict(template, date=date, duration=f"{clock(t)}-{clock(t+1)}",
                     day_of_week=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][date-1])
        copy[target]["activities"].append(entry)
    copy[target]["activities"].sort(key=lambda a: (a["week_id"], a["date"], a["duration"].split("-")[0]))
    return copy


def prompt_capture(repo):
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    spec = importlib.util.spec_from_file_location("qwen_runner", repo / "Scheduling/simulate_meeting_qwen.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    def capture(calendars):
        task = runner.LeaderAgent().initiate_task([s["member_name"] for s in calendars], 20, 10, 5)
        inputs = []
        for s in calendars:
            calls = []
            runner.call_llm = lambda prompt, **kwargs: calls.append(prompt) or ""
            agent = runner.Agent(s["member_name"], s["member_id"], s)
            agent.speak(task, [], {}, "", style="free", max_rounds=10, current_round=1, use_memory=False)
            assert len(calls) == 1
            inputs.append(calls[0].encode())
        return inputs
    return capture


def analyze(directory, capture):
    paths = sorted(directory.glob("Member*_final_schedule.json"))
    final = sorted(map(read, paths), key=lambda s: s["member_id"])
    originals = [read(directory / f"Member{s['member_id']}_original_schedule.json") for s in final]
    assert len(final) == 5 and all(s["total_days"] == 5 and s["time_unit"] == "minutes_10" for s in final)
    reference = read(directory / "task_plans")["meeting"]["solutions"]
    official = {s["start_time"] for s in reference}
    assert all(s["end_time"] == s["start_time"] + 2 for s in reference) and oracle(final) == official
    indices = [novel_entries(o, f) for o, f in zip(originals, final)]
    recipients = [i for i, entries in enumerate(indices) if entries]
    result = {"scenario": f"{directory.parent.name}/{directory.name}", "official": sorted(official),
              "observed_blocks": sum(map(len, indices)), "recipients": [i+1 for i in recipients]}
    # The generator stores a pre-pruning busy ratio over non-rest units.
    # A positive residual could be an insertion that matches an original entry.
    residuals = []
    for o, f, entries in zip(originals, final, indices):
        info = read(directory / f"Member{f['member_id']}_info.json")
        rest = {t for a in o["activities"] if a["description"].startswith("Rest:") for t in range(*interval(a))}
        busy = sum(end - begin for begin, end in
                   (interval(a) for a in f["activities"] if a["description"].startswith("Busy:")))
        pre_units = info["busy_ratio_final"] * (720 - len(rest))
        assert abs(pre_units - round(pre_units)) < 1e-8
        residuals.append(busy - round(pre_units) - len(entries))
    result["unexplained_busy_units_by_member"] = residuals
    if not recipients:
        return result
    assert len(recipients) == 1
    source = recipients[0]
    templates = [final[source]["activities"][i] for i in indices[source]]
    units = sorted(interval(a)[0] for a in templates)
    assert len(set(units)) == len(units)
    assert all(a["description"].startswith("Busy:") and interval(a)[1]-interval(a)[0] == 1 for a in templates)
    base = deepcopy(final)
    for i in reversed(indices[source]):
        del base[source]["activities"][i]
    expanded = oracle(base)
    assert expanded > official
    assert set(units) <= expanded | {t+1 for t in expanded}
    result.update(
        observed_entries=templates, base=sorted(expanded), regained=len(expanded-official),
        source_is_min_activity=len(base[source]["activities"]) == min(len(s["activities"]) for s in base),
        leave_one_out_extra=[len(oracle(final[:i]+final[i+1:])-official) for i in range(5)])
    for i in range(5):
        if i != source:
            assert oracle(replay_blocks(base, i, templates, [interval(a)[0] for a in templates])) == official
    result["all_four_relocations_preserve_reference"] = True
    alternate = alternative(expanded, official, len(templates))
    if alternate is None:
        return result
    other = replay_blocks(base, source, templates, alternate)
    answer = oracle(other)
    assert len(answer) == len(official) and not answer & official
    before, after = capture(final), capture(other)
    assert [i for i,(a,b) in enumerate(zip(before, after)) if a != b] == [source]
    recipient = next(i for i in range(5) if i != source)
    turn_a = replay_blocks(base, recipient, templates, [interval(a)[0] for a in templates])
    turn_b = replay_blocks(base, recipient, templates, alternate)
    assert oracle(turn_a) == official and oracle(turn_b) == answer
    moved_a, moved_b = capture(turn_a), capture(turn_b)
    assert [i for i,(a,b) in enumerate(zip(moved_a, moved_b)) if a != b] == [recipient]
    balanced_a, balanced_b = deepcopy(base), deepcopy(base)
    observed_by_unit = {interval(a)[0]: a for a in templates}
    for index, (t1, t2) in enumerate(zip(sorted(observed_by_unit), alternate, strict=True)):
        a = observed_by_unit[t1]
        balanced_a = replay_blocks(balanced_a, index % 5, [a], [t1])
        balanced_b = replay_blocks(balanced_b, index % 5, [a], [t2])
    assert oracle(balanced_a) == official and oracle(balanced_b) == answer
    result["distributed_placement_participants"] = min(len(templates), 5)
    result.update(counterfactual_blocks=alternate, counterfactual_answers=sorted(answer), relocation_recipient=recipient+1,
                  prompt_hashes={name: [hashlib.sha256(b).hexdigest() for b in values] for name, values in
                                zip(("released", "alternative", "relocated_released", "relocated_alternative"),
                                    (before, after, moved_a, moved_b))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    assert subprocess.check_output(["git", "-C", repository, "rev-parse", "HEAD"], text=True).strip() == PIN
    capture = prompt_capture(repository)
    paths = sorted((repository / "Scheduling/meeting/minutes_10/5days").glob("*/*/task_plans"))
    assert len(paths) == 200
    rows = [analyze(p.parent, capture) for p in paths]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["answer_count","released_scenarios","single_observed_recipient","observed_blocks",
                         "recipient_position_counts_1_to_5","disjoint_matched_alternatives"])
        for label in ["2","3","4","5","All"]:
            split = [r for r in rows if label == "All" or r["scenario"].split("/")[0] == label]
            recipients = Counter(i for r in split for i in r["recipients"])
            writer.writerow([label,len(split),sum(recipients.values()),sum(r["observed_blocks"] for r in split),
                             [recipients[i] for i in range(1,6)],sum("counterfactual_answers" in r for r in split)])

if __name__ == "__main__":
    main()
