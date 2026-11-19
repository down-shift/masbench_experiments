"""Score the paired design against exact meeting-start sets."""

from collections import Counter
import math
from random import Random

from common import read, require, sha, slot, write
from runner import episode_dir


def result_metrics(result, answers, tagged):
    answer = slot(result["leader"]["text"], "FINAL")
    succeeded = answer in answers
    first_valid = None
    if tagged:
        for index, turn in enumerate(result["turns"], 1):
            proposals = [slot(line, "PROPOSE") for line in turn["text"].splitlines()]
            if any(p in answers for p in proposals):
                first_valid = index
                break
    return {
        "success": succeeded, "final_start": answer,
        "abstained": result["leader"]["text"].strip() == "NO_MEETING",
        "invalid_format": answer is None and result["leader"]["text"].strip() != "NO_MEETING",
        "first_valid_proposal_turn": first_valid,
        "valid_proposal_missed": (first_valid is not None and not succeeded) if tagged else None,
    }


def interval(differences):
    rng = Random(6975)
    n = len(differences)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(5000))
    return [draws[124], draws[4874]]


def exact_binomial_interval(count, n, tail=0.0125):
    # Clopper-Pearson inversion. Two discordant proportions use Bonferroni
    # coverage for their difference, including when both observed counts are zero.
    def cdf(k, p):
        return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k+1))

    def invert(k, target):
        lo, hi = 0.0, 1.0
        for _ in range(48):
            mid = (lo + hi) / 2
            if cdf(k, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return (invert(count-1, 1-tail) if count else 0.0,
            invert(count, tail) if count < n else 1.0)


def exact_difference(a, b, n):
    lo_a, hi_a = exact_binomial_interval(a, n)
    lo_b, hi_b = exact_binomial_interval(b, n)
    return [100 * (lo_a-hi_b), 100 * (hi_a-lo_b)]


def exact_p(a, b):
    n = a + b
    return min(1, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2**n)


def score(args):
    plan = read(args.plan)
    manifest = read(args.run / "manifest.json")
    require(sha(args.plan.read_bytes()) == manifest["plan_sha256"], "Run was generated from another plan.")
    tagged = plan["tagged_proposals"]
    rows, missing = [], []
    fingerprints = set()
    model_ids = set()
    for pair in plan["pairs"]:
        for cell in pair["conditions"]:
            for world, answers in pair["oracles"].items():
                path = episode_dir(args.run, pair["scenario"], cell, world) / "result.json"
                if not path.exists():
                    missing.append([pair["scenario"], cell, world])
                    continue
                result = read(path)
                require((result["scenario"], result["condition"], result["world"]) == (pair["scenario"], cell, world),
                        f"Result identity mismatch at {path}.")
                expected_order = pair["conditions"][cell]["order"] * plan["rounds"]
                require([turn["member"] for turn in result["turns"]] == expected_order,
                        f"Wrong turns in {path}.")
                for call in [*result["turns"], result["leader"]]:
                    if call.get("model"):
                        model_ids.add(call["model"])
                    if call.get("system_fingerprint"):
                        fingerprints.add(call["system_fingerprint"])
                rows.append({"scenario": pair["scenario"], "condition": cell, "world": world,
                             **result_metrics(result, answers, tagged)})
    require(args.partial or not missing, f"{len(missing)} episodes incomplete. Finish the run or use --partial for diagnostics.")
    by_key = {(r["scenario"], r["condition"], r["world"]): r["success"] for r in rows}
    starts = {(r["scenario"], r["condition"], r["world"]): r["final_start"] for r in rows}
    complete = {}
    conditions = list(plan["pairs"][0]["conditions"])
    for pair in plan["pairs"]:
        sid = pair["scenario"]
        keys = [(sid, c, w) for c in conditions for w in ("released", "alternative")]
        if all(k in by_key for k in keys):
            complete[sid] = {c: all(by_key[sid, c, w] for w in ("released", "alternative")) for c in conditions}
    summaries = {}
    for condition in conditions:
        cond = [row for row in rows if row["condition"] == condition]
        first = Counter(r["first_valid_proposal_turn"] for r in cond if r["first_valid_proposal_turn"] is not None)
        summaries[condition] = {
            "completed_worlds": len(cond), "successful_worlds": sum(r["success"] for r in cond),
            "both_worlds_success": sum(r[condition] for r in complete.values()),
            "same_start_in_both_worlds": sum(
                starts[sid, condition, "released"] is not None
                and starts[sid, condition, "released"] == starts[sid, condition, "alternative"]
                for sid in complete
            ),
            "worlds_with_valid_proposal": sum(first.values()) if tagged else None,
            "first_valid_proposal_turns": dict(first) if tagged else None,
            "worlds_with_valid_proposal_missed": sum(r["valid_proposal_missed"] for r in cond) if tagged else None,
            "invalid_format": sum(r["invalid_format"] for r in cond),
            "abstained": sum(r["abstained"] for r in cond),
        }
    contrasts = []
    for a, b in plan["contrasts"]:
        delta = [int(r[a]) - int(r[b]) for r in complete.values()]
        n_a, n_b = delta.count(1), delta.count(-1)
        infer = len(delta) >= 2 and not missing and not manifest["mock"] and plan["split"] == "main"
        contrasts.append({"a": a, "b": b, "pairs": len(delta), "a_only": n_a, "b_only": n_b,
                          "a_minus_b_pp": 100 * sum(delta) / len(delta) if delta else None,
                          "paired_bootstrap_95_pp": [100 * d for d in interval(delta)] if infer and len(set(delta)) > 1 else None,
                          "conservative_exact_95_pp": exact_difference(n_a, n_b, len(delta)) if infer else None,
                          "mcnemar_exact_p_unadjusted": exact_p(n_a, n_b) if infer else None})
    report = {"design": plan["design"], "mock": manifest["mock"], "split": plan["split"],
              "complete_pairs": len(complete), "planned_pairs": len(plan["pairs"]),
              "missing_episodes": missing, "model": manifest["model"], "observed_model_ids": sorted(model_ids),
              "system_fingerprints": sorted(fingerprints), "conditions": summaries, "contrasts": contrasts}
    write(args.run / "scores.json", report)
    write(args.run / "world_scores.json", rows)
    print(f"{len(complete)}/{len(plan['pairs'])} complete pairs. See {args.run / 'scores.json'}.")
    for r in contrasts:
        print(f"{r['a']} vs {r['b']}: {r['a_only']} vs {r['b_only']} pairs won exclusively.")
