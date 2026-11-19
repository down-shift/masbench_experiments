"""Build reproducible, self-contained model inputs from the pinned MASBench checkout."""

from collections import defaultdict
from copy import deepcopy
import importlib.util
from pathlib import Path
from random import Random
import re
import subprocess
import sys
import types

import audit
from common import AGENT_FORMAT, HISTORY, ROOT, read, require, sha, write

AUDITED_KEYS = {
    ("original", "released"): "released",
    ("original", "alternative"): "alternative",
    ("relocated", "released"): "relocated_released",
    ("relocated", "alternative"): "relocated_alternative",
}
CONDITIONS = {
    "order": {"first": ("original", "first"), "last": ("original", "last")},
    "relocation": {
        "original_first": ("original", "first"),
        "original_last": ("original", "last"),
        "relocated_first": ("relocated", "first"),
        "relocated_last": ("relocated", "last"),
    },
    "concentration": {"concentrated": ("original", "fixed"), "distributed": ("distributed", "fixed")},
}
CONTRASTS = {
    "order": [["first", "last"]],
    "relocation": [["original_first", "original_last"], ["relocated_first", "relocated_last"]],
    "concentration": [["concentrated", "distributed"]],
}


def stratified(rows, size, seed):
    require(1 <= size <= len(rows), f"Requested {size} pairs from a pool of {len(rows)}.")
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: tuple(map(int, row["scenario"].split("/")))):
        groups[len(row["official"])].append(row)
    slots = {k: len(v) * size // len(rows) for k, v in groups.items()}
    remainder = sorted(groups, key=lambda k: (-(len(groups[k]) * size % len(rows)), k))
    for k in remainder[:size - sum(slots.values())]:
        slots[k] += 1
    rng = Random(seed)
    return [row for k in sorted(groups) for row in rng.sample(groups[k], slots[k])]


def upstream(repo):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == audit.PIN, f"MASBench must be checked out at {audit.PIN}.")
    require(not git("status", "--porcelain", "--", "Scheduling"), "Scheduling must be a clean checkout.")
    path = repo / "Scheduling/simulate_meeting_qwen.py"
    spec = importlib.util.spec_from_file_location("masbench_qwen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    spec.loader.exec_module(module)
    return module, sha(path.read_bytes())


def prompt(module, calendar, task, rounds, round_id, history):
    calls = []
    module.call_llm = lambda text, **kwargs: calls.append(text) or ""
    module.Agent(calendar["member_name"], calendar["member_id"], calendar).speak(
        task, history, {}, "", style="free", max_rounds=rounds, current_round=round_id, use_memory=False)
    require(len(calls) == 1, "Unexpected upstream prompt path.")
    return calls[0]


def make_pair(repo, module, entry, mode, rounds, tagged):
    scenario = entry["scenario"]
    require(re.fullmatch(r"[2-5]/[1-9]\d*", scenario), f"Invalid scenario path {scenario}.")
    directory = repo / "Scheduling/meeting/minutes_10/5days" / scenario
    released = sorted((audit.read(p) for p in directory.glob("Member*_final_schedule.json")),
                      key=lambda c: c["member_id"])
    require([c["member_id"] for c in released] == list(range(1, 6)), f"Unexpected roster in {scenario}.")
    official = {s["start_time"] for s in audit.read(directory / "task_plans")["meeting"]["solutions"]}
    expected = {"released": sorted(official), "alternative": entry["counterfactual_answers"]}
    require(expected["released"] == entry["official"] and official.isdisjoint(expected["alternative"]),
            f"Bad reference answers in {scenario}.")
    source, target = entry["recipients"][0] - 1, entry["relocation_recipient"] - 1
    templates = entry["observed_entries"]
    units = {"released": [audit.interval(a)[0] for a in templates], "alternative": entry["counterfactual_blocks"]}
    base = deepcopy(released)
    for item in templates:
        base[source]["activities"].remove(item)
    require(sorted(audit.oracle(base)) == entry["base"], f"Base oracle mismatch in {scenario}.")

    conditions = {}
    cells = CONDITIONS[mode]
    names = [c["member_name"] for c in released]
    leader = module.LeaderAgent()
    check_task = leader.initiate_task(names, 20, 10, 5)
    task = leader.initiate_task(names, 20, rounds * 5, 5)
    templates_by_allocation = {}
    for allocation, _ in dict.fromkeys(cells.values()):
        if allocation in templates_by_allocation:
            continue
        holder = (target if allocation == "relocated" else source) + 1
        worlds = {}
        for world in expected:
            if allocation == "original" and world == "released":
                calendars = released
            elif allocation in ("original", "relocated"):
                calendars = audit.replay_blocks(base, holder - 1, templates, units[world])
            else:
                calendars = base
                ordered = sorted(zip(units["released"], templates))
                for i, (unit, (_, activity)) in enumerate(zip(sorted(units[world]), ordered, strict=True)):
                    calendars = audit.replay_blocks(calendars, i % 5, [activity], [unit])
            require(sorted(audit.oracle(calendars)) == expected[world],
                    f"Oracle mismatch in {scenario}/{allocation}/{world}.")
            if (allocation, world) in AUDITED_KEYS:
                hashes = [sha(prompt(module, c, check_task, 10, 1, [])) for c in calendars]
                require(hashes == entry["prompt_hashes"][AUDITED_KEYS[allocation, world]],
                        f"Published prompt hashes disagree in {scenario}/{allocation}/{world}.")
            templates_for_world = [
                [prompt(module, c, task, rounds, r, [HISTORY]) + (AGENT_FORMAT if tagged else "")
                 for c in calendars] for r in range(1, rounds + 1)
            ]
            require(all(p.count(HISTORY) == 1 for row in templates_for_world for p in row), "Invalid prompt hook.")
            worlds[world] = templates_for_world
        if allocation != "distributed":
            for a, b in zip(worlds["released"], worlds["alternative"], strict=True):
                changed = [i for i, (x, y) in enumerate(zip(a, b), 1) if x != y]
                require(changed == [holder], f"Incorrect prompt difference in {scenario}/{allocation}.")
        templates_by_allocation[allocation] = worlds
    for label, (allocation, position) in cells.items():
        holder = (target if allocation == "relocated" else source) + 1
        rest = [i for i in range(1, 6) if i != holder]
        order = {"first": [holder] + rest, "last": rest + [holder], "fixed": list(range(1, 6))}[position]
        conditions[label] = {"allocation": allocation, "order": order,
                             "holder": None if allocation == "distributed" else holder}
    return {"scenario": scenario, "names": names, "task": task, "oracles": expected,
            "conditions": conditions, "templates": templates_by_allocation}


def prepare(args):
    cases = (ROOT / "data/cases.json").read_bytes()
    pool = [row for row in read(ROOT / "data/cases.json") if "counterfactual_answers" in row
            and not any(row["unexplained_busy_units_by_member"])
            and (args.design != "concentration" or row["distributed_placement_participants"] >= 2)]
    # A fixed holdout keeps smoke cases out of every main run.
    smoke = stratified(pool, 4, 6975)
    if args.split == "smoke":
        require(1 <= args.pairs <= 4, "Smoke uses up to four reserved pairs.")
        selected = smoke[:args.pairs]
    else:
        held = {row["scenario"] for row in smoke}
        selected = stratified([row for row in pool if row["scenario"] not in held], args.pairs, args.seed + 1)
    module, runner_hash = upstream(args.repo.resolve())
    plan = {
        "schema": 1, "design": args.design, "split": args.split, "sample_seed": args.seed,
        "rounds": args.rounds, "tagged_proposals": not args.native_prose,
        "repo_commit": audit.PIN, "upstream_runner_sha256": runner_hash, "audit_cases_sha256": sha(cases),
        "contrasts": CONTRASTS[args.design],
        "pairs": [make_pair(args.repo, module, e, args.design, args.rounds, not args.native_prose) for e in selected],
    }
    require(not args.out.exists(), f"Plan already exists: {args.out}. Use another path to freeze a new plan.")
    write(args.out, plan)
    print(f"Frozen {len(selected)} {args.split} pairs at {args.out}.")
