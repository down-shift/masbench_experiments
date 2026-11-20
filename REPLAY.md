# Participant-report replay

The cases use synthetic private busy calendars. Each participant receives only their own busy intervals and all candidate times. There are 4 smoke and 12 main cases, disjoint within the fixed seed. One slot is feasible and each of two participants exclusively blocks a different alternative. Reports must match the requested format, with an optional final period, and the deterministic calendar oracle. The leader sees only complete, correct reports. Reversing candidate lines leaves the reports unchanged.

With the Qwen3-8B server running on Ubuntu, preserve its startup log and exact checkpoint revision. From the project directory:

```sh
uv run python replay_agents.py prepare --split smoke --out runs/replay-v2-smoke
uv run python replay_agents.py run --split smoke --out runs/replay-v2-smoke --max-calls 28
uv run python replay_agents.py score --split smoke --out runs/replay-v2-smoke
```

Proceed only if the completed smoke score shows `"smoke_pass": true` (at least three valid leader pairs). Ignore switch counts for this gate. Measure elapsed time and lower the main call cap if needed. A strict format failure excludes a case from replay and remains in the reported denominator. If the smoke protocol needs modification, record that change before main and use new directories.

```sh
uv run python replay_agents.py prepare --split main --out runs/replay-v2-main
uv run python replay_agents.py run --split main --out runs/replay-v2-main --max-calls 42
uv run python replay_agents.py run --split main --out runs/replay-v2-main --max-calls 42
uv run python replay_agents.py score --split main --out runs/replay-v2-main
```

Repeat the run command until it reports completion. The cap counts retries and cached calls are reused. Keep the model server configuration fixed throughout. The number of leader calls depends on report eligibility. Score reports missing and invalid calls explicitly. Treat incomplete or mock outputs as pipeline checks only. Leader flip rates are conditional on accurate reports and two valid leader responses; report all denominators. The runner stores prompts by hash and raw responses. The frozen script and plan reconstruct exact prompts.
