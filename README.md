# MASBench scheduling experiments

This repository contains scheduling experiments derived from MASBench, a resumable model runner, and scoring code. The paper and its anonymous reproducibility package are in `paper_aaai/`. The package `paper_aaai/supplement.zip` includes the frozen confirmation plan, saved responses, and instructions for reproducing the paper's main analysis offline.

The `plans/` and `runs/` directories are ignored by Git. The commands below describe the earlier scheduling experiments and require locally prepared plans; the supplement includes only `order-smoke.json` and `order-main.json` from those plans. Python and `uv` suffice for preparation, offline checks, and scoring. Model inference runs on a separate chat-completions server.

## Start on an M1 Mac

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run commands from the project directory. Extract the plans from `paper_aaai/supplement.zip` into the project directory first. The first command creates a Python 3.12 environment. There are no third-party Python packages.

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python experiment.py run --plan plans/order-smoke.json \
  --out runs/offline-check --mock --max-calls 96
uv run --locked python experiment.py score --plan plans/order-smoke.json \
  --run runs/offline-check
```

The offline fixture uses the same constant reply in both worlds. Its scores are only pipeline checks. You can run real model requests from the Mac if you have an API endpoint. Set `MASBENCH_BASE_URL` to your provider's OpenAI-compatible base URL ending in `/v1` or the provider-specific equivalent, and set `MASBENCH_API_KEY` in the shell if the provider requires one. Use an API model that accepts temperature 0, plain-text chat, and a configurable output limit. Use `--token-field max_completion_tokens` if required by the endpoint.

```sh
# With MASBENCH_BASE_URL and (if needed) MASBENCH_API_KEY already set:
model='YOUR_EXACT_MODEL_ID'
uv run --locked python experiment.py run --plan plans/order-smoke.json \
  --out runs/order-smoke --model "$model" --max-calls 110
uv run --locked python experiment.py score --plan plans/order-smoke.json --run runs/order-smoke
```

Inspect `scores.json` and some `episodes/*/*/*/result.json` transcripts before starting the main run. Look for valid final formats, enough valid proposals to make the comparison meaningful, and any unexpected model/version changes. If smoke results motivate a protocol change, freeze it before the main run and exclude the smoke cases from the analysis.

## Run Qwen3-4B on Ubuntu with an RTX 5080

Use the existing CUDA-enabled vLLM environment and cached `Qwen/Qwen3-4B` checkpoint. The launcher pins the model and tokenizer to the same revision used in `when-sampling`, loads the model in unquantized BF16, and sets a 16,384-token context for the scheduling prompts. Reuse a running server if it matches these settings. If the existing server is using the `when-sampling` script's 4,096-token context, stop it and launch with the command below.

```sh
# Point this at vLLM in your existing environment. Omit VLLM_BIN if vllm is on PATH.
VLLM_BIN=/path/to/your/vllm-env/bin/vllm sh scripts/serve-qwen3-4b.sh
```

In another terminal, run the smoke plan with thinking disabled for every speaker and the leader. Use a dedicated run directory so earlier model responses cannot be mixed with these results.

```sh
uv sync --locked
unset MASBENCH_API_KEY
uv run --locked python experiment.py run --qwen3-4b --plan plans/order-smoke.json \
  --base-url http://127.0.0.1:8000/v1 --out runs/order-smoke-qwen3-4b-bf16 \
  --workers 2 --max-calls 110 --timeout 180
uv run --locked python experiment.py score --plan plans/order-smoke.json \
  --run runs/order-smoke-qwen3-4b-bf16
```

The preset sends `chat_template_kwargs.enable_thinking=false` and saves it in `manifest.json`. Precision and checkpoint revision are set by the server. Keep its startup log and vLLM version with the run. Check prompt token counts and completion reasons in the smoke outputs before starting the main run. For Ollama and other OpenAI-compatible servers, use the generic `--model` and `--base-url` options above.

## Main comparison

Run the same configuration you settled on during smoke. The order plan has 128 episodes and needs 768 successful model calls. `--max-calls` is a cap **per invocation** and includes retries. The same invocation can be repeated to resume using the saved calls. A completed run resumes without sending any further requests. The command below uses the Qwen3-4B preset.

```sh
plan=plans/order-main.json
run=runs/order-main-qwen3-4b-bf16
uv run --locked python experiment.py run --qwen3-4b --plan "$plan" --out "$run" \
  --base-url http://127.0.0.1:8000/v1 --workers 2 --max-calls 850 --timeout 180
uv run --locked python experiment.py score --plan "$plan" --run "$run"
```

| Plan | Comparison | Pairs | Episodes | Successful calls |
|---|---|---:|---:|---:|
| `order-smoke.json` | Holder first vs. last, reserved checks | 4 | 16 | 96 |
| `order-main.json` | Holder first vs. last | 32 | 128 | 768 |
| `relocation-main.json` | First vs. last separately for original and alternate holders | 32 | 256 | 1,536 |
| `concentration-main.json` | Original concentrated vs. round-robin distributed blocks, fixed member-ID order | 32 | 128 | 768 |

The three main plans use the same 32 scenarios. The relocation plan already includes the original-holder conditions. If running it after the order experiment, budget for those calls again or treat the relocated contrast as a separate analysis. The code reports the two within-placement contrasts separately. The concentration comparison changes local calendar contents as part of the intervention.

## Protocol and analysis

- Each released/counterfactual pair has disjoint valid meeting starts. Each episode gets a fresh transcript, five speakers per round, and a single transcript-only decision call at the end. First and last change only the holder's place in the order. The other agents keep their relative order.
- Agent inputs use the pinned MASBench Qwen free/no-memory prompt builder with an explicit five-turn budget for one round. By default an extra line requests `PROPOSE Day X HH:MM-HH:MM` for active proposals. The leader must output exactly `FINAL Day X HH:MM-HH:MM` or `NO_MEETING`. A malformed final reply counts as a model failure. A failed provider request leaves an episode incomplete. The oracle is used in preparation and offline scoring only. See the raw transcripts if the strict proposal format misses a valid proposal in prose.
- Primary outcome: one pair succeeds if **both** worlds get valid final meetings. The report gives the paired difference in percentage points and the two discordant counts. For complete live main runs it gives a paired bootstrap interval (unless the bootstrap would be degenerate), a conservative exact interval constructed from the two discordant proportions, and an unadjusted exact McNemar p-value. At 32 pairs, give the raw counts and interval with any reported effect. The two contrasts in the relocation plan are separate and unadjusted.
- Secondary outcomes include single-world success, exact reuse of the same final start across worlds, the turn of the first tagged valid proposal, and valid proposals that the leader missed. Invalid-format replies and abstentions have separate counts.
- Completed requests are saved separately with the exact request prompt hash, response text, provider-reported model and fingerprint when available, token usage, and latency. The manifest binds the plan, model, endpoint, generation parameters, and runner code. It contains no API key. It rejects changed settings at an existing run path. Use a different directory for a new configuration. Endpoint retries can repeat an attempted request after a connection failure.
- For a partial run, use `score --partial` to see completion and descriptive metrics. Inferential statistics are withheld. Model tags and server settings can drift independently of the runner. Preserve the frozen plan, manifest, responses, and model/server versions before reporting results.

## Rebuild a plan

The plans in the supplement run without Git or the upstream checkout. To rebuild, check out the exact audited revision and run:

```sh
git clone https://github.com/BUPT-GAMMA/MASBench.git
git -C MASBench checkout c0301a650db95958924bb080192c09fef39389a5
uv run --locked python experiment.py prepare --repo MASBench \
  --design order --split main --pairs 32 --rounds 1 --seed 6975 \
  --out plans/rebuilt-order.json
cmp plans/order-main.json plans/rebuilt-order.json
```

Preparation reconstructs both worlds, checks exact answer sets for each allocation, verifies concentrated prompt hashes against the released audit, and compiles the agent templates. It reserves four smoke scenarios before main sampling, and stratifies by the number of valid starts. Use `--native-prose` to use the native agent wording without the extra proposal tags. Final scoring still uses the fixed single-line leader reply. With multiple rounds the driver keeps the full public transcript; it does not stop early.
