"""Manifest and local HTTP boundary checks; no external network or real model."""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent_reliability.dataset import TASK_TYPES, load_manifest, make_manifest
from agent_reliability.io import write_json
from agent_reliability.llm import HTTPTransport, InfrastructureError


class InterfaceTests(unittest.TestCase):
    def test_manifest_filters_freezes_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, kind in enumerate(TASK_TYPES):
                folder = root / "data" / "json" / "train" / f"task{i}"
                write_json(folder / "traj_data.json", {"task_type": kind, "scene": {"floor_plan": i}})
                write_json(folder / "game.tw-pddl", {"solvable": True})
            excluded = root / "data" / "json" / "train" / "bad"
            write_json(excluded / "traj_data.json", {"task_type": TASK_TYPES[0], "scene": {"floor_plan": 100}})
            path = root / "manifest.json"
            manifest = make_manifest(root / "data", path, "train", 1, 42)
            self.assertEqual(len(manifest["tasks"]), 6)
            self.assertEqual(manifest["excluded"]["missing_game"], 1)
            self.assertEqual(load_manifest(path), manifest)
            with self.assertRaises(FileExistsError):
                make_manifest(root / "data", path, "train", 1, 42)
            manifest["tasks"][0]["game_sha256"] = "changed"
            write_json(path, manifest)
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_http_openai_compatible_wire_contract(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"data": [{"id": "offline_fake"}]}).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, payload))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode())

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            transport = HTTPTransport(f"http://127.0.0.1:{server.server_port}/v1", "offline", 2)
            self.assertEqual(transport.health("offline_fake")["served_model"], "offline_fake")
            with self.assertRaises(InfrastructureError):
                transport.health("wrong_model")
            transport.complete({"model": "offline_fake", "messages": [],
                                "chat_template_kwargs": {"enable_thinking": False}, "top_k": 20})
            self.assertEqual(received[0][0], "/v1/chat/completions")
            self.assertFalse(received[0][1]["chat_template_kwargs"]["enable_thinking"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
