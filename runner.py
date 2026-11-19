"""Run frozen scheduling episodes through a chat-completions endpoint."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import math
import os
from pathlib import Path
from random import Random
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from common import HISTORY, LEADER_FORMAT, ROOT, read, require, sha, write


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class Backend:
    def __init__(self, args):
        self.mock = args.mock
        self.config = {
            "model": "offline-fixture" if args.mock else args.model,
            "endpoint": None if args.mock else args.base_url.rstrip("/"),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "token_field": args.token_field,
            "provider_settings": args.extra,
        }
        self.key = None if args.mock else os.environ.get("MASBENCH_API_KEY")
        if not args.mock:
            require(args.model, "Set --model for a real run.")
            url = urlsplit(self.config["endpoint"])
            require(url.scheme in ("https", "http") and url.hostname and not (
                url.username or url.password or url.query or url.fragment), "Use a base URL without credentials or query.")
            require(url.scheme == "https" or url.hostname in {"localhost", "127.0.0.1", "::1"} or not self.key,
                    "Use HTTPS when sending an API key to a remote server.")
        self.timeout, self.retries, self.limit = args.timeout, args.retries, args.max_calls
        self.count = 0
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.count >= self.limit:
                raise RuntimeError("Call limit reached. Resume the same run with a new invocation budget.")
            self.count += 1

    def complete(self, prompt, role):
        if self.mock:
            self.reserve()
            text = "PROPOSE Day 1 08:00-08:20" if role == "agent" else "FINAL Day 1 08:00-08:20"
            return {"text": text, "model": "offline-fixture", "finish_reason": "stop", "usage": {}}
        payload = {
            "model": self.config["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config["temperature"], self.config["token_field"]: self.config["max_tokens"],
            **self.config["provider_settings"],
        }
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.config["endpoint"] + "/chat/completions",
                          data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        for attempt in range(self.retries + 1):
            self.reserve()
            try:
                with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                    body = json.load(response)
                choice = body["choices"][0]
                message = choice["message"].get("content")
                if message is None:
                    message = ""
                require(isinstance(message, str), "The server returned non-text message content.")
                return {"text": message, "model": body.get("model"), "id": body.get("id"),
                        "system_fingerprint": body.get("system_fingerprint"),
                        "finish_reason": choice.get("finish_reason"), "usage": body.get("usage") or {}}
            except HTTPError as error:
                # Avoid writing provider messages containing identifiers or request data to the logs.
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Endpoint returned HTTP {error.code}.") from None
            except (URLError, TimeoutError):
                if attempt == self.retries:
                    raise RuntimeError("Endpoint connection or timeout failure.") from None
            except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
                raise RuntimeError("Unexpected chat completion response.") from error
            time.sleep(min(2 ** attempt, 8))
        raise AssertionError("Retry loop must return or raise.")


def episode_dir(folder, scenario, condition, world):
    require(re.fullmatch(r"[2-5]/[1-9]\d*", scenario), "Malformed scenario ID.")
    require(re.fullmatch(r"[a-z_]+", condition) and world in {"released", "alternative"}, "Malformed episode ID.")
    return folder / "episodes" / scenario.replace("/", "-") / condition / world


def cached_call(path, backend, prompt, role):
    prompt_hash = sha(prompt)
    if path.exists():
        record = read(path)
        require(record["prompt_sha256"] == prompt_hash and record["role"] == role,
                f"Cached request differs from the frozen input at {path}.")
        return record
    started = time.monotonic()
    result = backend.complete(prompt, role)
    record = {**result, "prompt_sha256": prompt_hash, "role": role,
              "latency_seconds": round(time.monotonic() - started, 4)}
    write(path, record)
    return record


def episode(folder, backend, pair, condition, world):
    directory = episode_dir(folder, pair["scenario"], condition, world)
    cell = pair["conditions"][condition]
    templates = pair["templates"][cell["allocation"]][world]
    conversation = []
    turns = []
    for round_id, prompts in enumerate(templates, 1):
        for member in cell["order"]:
            template = prompts[member - 1]
            prompt = template.replace(HISTORY, "".join(conversation), 1)
            record = cached_call(directory / f"r{round_id:02}_m{member}.json", backend, prompt, "agent")
            # This matches MASBench's within-round public history formatting.
            conversation.append(f'{pair["names"][member - 1]}: {record["text"]}\n')
            turns.append({"member": member, "round": round_id, **record})
    prompt = pair["task"] + LEADER_FORMAT + "\nDiscussion transcript:\n" + "".join(conversation)
    leader = cached_call(directory / "leader.json", backend, prompt, "leader")
    write(directory / "result.json", {"scenario": pair["scenario"], "condition": condition, "world": world,
                                     "turns": turns, "leader": leader})


def jobs(plan):
    work = [(p, cell, world) for p in plan["pairs"] for cell in p["conditions"]
            for world in ("released", "alternative")]
    Random(plan["sample_seed"] + 2).shuffle(work)
    return work


def implementation_hash():
    return sha("".join(sha((ROOT / name).read_bytes()) for name in ("common.py", "runner.py")))


def run(args):
    require(math.isfinite(args.temperature) and 0 <= args.temperature <= 2, "Invalid temperature.")
    protected = {"model", "messages", "temperature", "max_tokens", "max_completion_tokens", "stream"}
    require(isinstance(args.extra, dict) and not protected.intersection(args.extra),
            "--extra must be a JSON object and cannot override core request settings.")
    require(not any(re.search(r"key|token|auth|secret|password", key, re.I) for key in args.extra
                    if key not in {"chat_template_kwargs", "min_tokens", "top_k"}),
            "Put credentials in MASBENCH_API_KEY, not provider settings.")
    plan = read(args.plan)
    require(plan["schema"] == 1, "Unsupported plan schema.")
    backend = Backend(args)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".lock").open("w") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is already running this directory.") from None
        manifest = {"plan_sha256": sha(args.plan.read_bytes()), "runner_sha256": implementation_hash(),
                    "mock": args.mock, **backend.config}
        path = args.out / "manifest.json"
        if path.exists():
            require(read(path) == manifest, "Run settings changed. Use a new output directory.")
        else:
            write(path, manifest)
        work = jobs(plan)
        full_calls = len(work) * (5 * plan["rounds"] + 1)
        print(f"{len(work)} episodes; {full_calls} successful calls for a fresh complete run. "
              f"Invocation cap: {args.max_calls} attempts.", flush=True)
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(episode, args.out, backend, p, c, w): (p["scenario"], c, w) for p, c, w in work}
            for i, future in enumerate(as_completed(pending), 1):
                try:
                    future.result()
                except (OSError, RuntimeError, ValueError) as error:
                    failures.append({"episode": pending[future], "error": str(error)})
                if i % 16 == 0 or i == len(work):
                    print(f"Processed {i}/{len(work)} episodes, {len(failures)} incomplete.", flush=True)
        write(args.out / "last_invocation.json", {"attempts": backend.count, "failures": failures})
        if failures:
            raise RuntimeError(f"{len(failures)} incomplete episodes. See last_invocation.json and resume with the same command.")
