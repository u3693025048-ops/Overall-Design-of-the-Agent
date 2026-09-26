"""
Minimal stdlib HTTP server for async task status polling.

Usage:
  python -m multi_agent.asynctask.http_server --port 8765
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse

from multi_agent.asynctask.task_queue import AsyncTaskQueue


class TaskStatusHandler(BaseHTTPRequestHandler):
    queue: AsyncTaskQueue = AsyncTaskQueue()

    def log_message(self, format: str, *args) -> None:
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if path == "/api/tasks":
            tasks = [t.to_dict() for t in self.queue.list_tasks(limit=50)]
            self._send_json(200, {"tasks": tasks})
            return
        if path.startswith("/api/tasks/"):
            task_id = path.split("/")[-1]
            rec = self.queue.get_status(task_id)
            if rec is None:
                self._send_json(404, {"error": "task not found", "task_id": task_id})
                return
            self._send_json(200, rec.to_dict())
            return
        self._send_json(404, {"error": "not found"})


def serve(port: int = 8765, host: str = "127.0.0.1") -> None:
    server = HTTPServer((host, port), TaskStatusHandler)
    print(f"[TaskHTTP] listening on http://{host}:{port}")
    print(f"  GET /health")
    print(f"  GET /api/tasks")
    print(f"  GET /api/tasks/{{task_id}}")
    server.serve_forever()


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="GSMultiAgent async task status HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
