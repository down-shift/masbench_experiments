"""Protocol, checkpoint, and scoring checks against a local fake HTTP server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

from common import HISTORY, ROOT, format_slot, read, slot, write
from runner import jobs
from score import exact_difference


class ExperimentTest(unittest.TestCase):
    def test_scoring_start_grid_duration_and_no_example_leak(self):
        self.assertEqual(slot("FINAL " + format_slot(273), "FINAL"), 273)
        for text in ("FINAL Day 1 08:10-08:40", "FINAL Day 1 08:05-08:25",
                     "FINAL Day 1 08:60-09:20", "I reject FINAL Day 1 08:00-08:20",
                     "FINAL Day 1 24:00-24:20", "FINAL Day 1 08:00-08:20\nExtra"):
            self.assertIsNone(slot(text, "FINAL"))

    def test_no_discordant_pairs_has_uncertainty(self):
        lo, hi = exact_difference(0, 0, 32)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_http_resume_passes_only_private_input_and_public_history(self):
        pair = {
            "scenario": "2/19", "names": ["Agent" + str(i) for i in range(1, 6)],
            "task": "Find a meeting for everyone.",
            "oracles": {"released": [273], "alternative": [70]},
            "conditions": {"first": {"allocation": "original", "order": [3, 1, 2, 4, 5]},
                           "last": {"allocation": "original", "order": [1, 2, 4, 5, 3]}},
            "templates": {"original": {
                w: [[f"PrivateCalendar{i} " + HISTORY for i in range(1, 6)]]
                for w in ("released", "alternative")
            }},
        }
        plan = {"schema": 1, "pairs": [pair], "sample_seed": 6975, "rounds": 1,
                "contrasts": [["first", "last"]], "tagged_proposals": True, "design": "order", "split": "main"}
        scripted = []
        for row, cell, world in jobs(plan):
            valid = row["oracles"][world][0]
            for member in row["conditions"][cell]["order"]:
                scripted.append((member, "PROPOSE " + format_slot(valid)))
            scripted.append((None, "FINAL " + format_slot(valid if cell == "first" else 48)))
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                _, response = scripted[len(received)]
                received.append((self.path, request))
                payload = json.dumps({"model": "fixture", "choices": [
                    {"message": {"content": response}, "finish_reason": "stop"}
                ]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                tmp = Path(tmp)
                plan_path, results = tmp / "plan.json", tmp / "run"
                write(plan_path, plan)
                cli = [sys.executable, str(ROOT / "experiment.py")]
                run_cmd = cli + ["run", "--plan", str(plan_path), "--out", str(results), "--qwen3-4b",
                                 "--base-url", f"http://127.0.0.1:{server.server_port}/v1"]
                interrupted = subprocess.run(run_cmd + ["--max-calls", "2"], capture_output=True, timeout=30)
                self.assertEqual(interrupted.returncode, 1)
                self.assertEqual(len(received), 2)
                self.assertEqual(len(list(results.glob("episodes/*/*/*/r*.json"))), 2)
                subprocess.run(run_cmd + ["--max-calls", "22"], check=True, capture_output=True, timeout=30)
                subprocess.run(run_cmd + ["--max-calls", "1"], check=True, capture_output=True, timeout=30)
                self.assertEqual(len(received), 24)
                subprocess.run(cli + ["score", "--plan", str(plan_path), "--run", str(results)],
                               check=True, capture_output=True, timeout=30)
                summary = read(results / "scores.json")
                self.assertEqual(summary["conditions"]["first"]["both_worlds_success"], 1)
                self.assertEqual(summary["conditions"]["last"]["both_worlds_success"], 0)
                self.assertEqual(summary["conditions"]["last"]["worlds_with_valid_proposal_missed"], 2)
                self.assertEqual(summary["contrasts"][0]["a_only"], 1)
                manifest = read(results / "manifest.json")
                self.assertEqual(manifest["model"], "Qwen/Qwen3-4B")
                self.assertEqual(manifest["provider_settings"], {"chat_template_kwargs": {"enable_thinking": False}})
                for (path, request), (member, _) in zip(received, scripted, strict=True):
                    self.assertEqual(path, "/v1/chat/completions")
                    self.assertEqual(request["model"], "Qwen/Qwen3-4B")
                    self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False})
                    text = request["messages"][0]["content"]
                    self.assertNotIn(HISTORY, text)
                    self.assertNotIn("oracles", text)
                    expected = [f"PrivateCalendar{member}"] if member else []
                    self.assertEqual([f"PrivateCalendar{i}" for i in range(1, 6)
                                      if f"PrivateCalendar{i}" in text], expected)
                    if member is None:
                        self.assertIn("Discussion transcript:", text)
                        self.assertEqual(text.count("PROPOSE"), 5)
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
