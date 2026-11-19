"""MASBench matched-pair experiments. Run with uv run python experiment.py."""

import argparse
import json
import os
from pathlib import Path
import subprocess

from prepare import CONDITIONS, prepare
from runner import run
from score import score


def positive(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("expected an integer of at least one")
    return value


def nonnegative(raw):
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return value


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="freeze a self-contained plan offline")
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--design", default="order", choices=CONDITIONS)
    p.add_argument("--split", default="main", choices=["smoke", "main"])
    p.add_argument("--pairs", type=positive, default=32)
    p.add_argument("--rounds", type=positive, default=1)
    p.add_argument("--seed", type=int, default=6975)
    p.add_argument("--native-prose", action="store_true", help="omit the PROPOSE tag instruction and proposal diagnostics")
    p.set_defaults(action=prepare)

    p = sub.add_parser("run", help="run or resume a frozen plan")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    model = p.add_mutually_exclusive_group()
    model.add_argument("--model")
    model.add_argument("--qwen3-4b", action="store_true", help="use Qwen/Qwen3-4B in vLLM with thinking disabled")
    p.add_argument("--base-url", default=os.getenv("MASBENCH_BASE_URL"),
                   help="OpenAI-compatible base URL; defaults to local Ollama or local vLLM with --qwen3-4b")
    p.add_argument("--max-calls", required=True, type=positive, help="per-invocation attempt cap, including retries")
    p.add_argument("--max-tokens", type=positive, default=256)
    p.add_argument("--token-field", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--timeout", type=positive, default=90)
    p.add_argument("--retries", type=nonnegative, default=2)
    p.add_argument("--workers", type=positive, default=1)
    p.add_argument("--extra", type=json.loads, default={}, help="JSON provider-specific request parameters")
    p.add_argument("--mock", action="store_true", help="exercise the pipeline using constant dummy replies, offline")
    p.set_defaults(action=run)

    p = sub.add_parser("score", help="score completed episodes with the audit oracles")
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--partial", action="store_true", help="score for diagnostics with incomplete episodes")
    p.set_defaults(action=score)

    args = cli.parse_args()
    try:
        if args.command == "run":
            if args.qwen3_4b:
                args.model = "Qwen/Qwen3-4B"
                if args.base_url is None:
                    args.base_url = "http://127.0.0.1:8000/v1"
                if not isinstance(args.extra, dict):
                    raise ValueError("--extra must be a JSON object.")
                template = args.extra.get("chat_template_kwargs", {})
                if not isinstance(template, dict) or template.get("enable_thinking", False) is not False:
                    raise ValueError("--qwen3-4b requires chat_template_kwargs.enable_thinking to be false.")
                args.extra = {**args.extra, "chat_template_kwargs": {**template, "enable_thinking": False}}
            elif args.base_url is None:
                args.base_url = "http://127.0.0.1:11434/v1"
        args.action(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        cli.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
