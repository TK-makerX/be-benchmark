from __future__ import annotations

import html
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from be_benchmark_shell.registry import RegistryError
from be_benchmark_shell.service import evaluate_request


FORM = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Prediction interface scaffold</title>
<style>body{{font-family:Arial,sans-serif;max-width:780px;margin:40px auto;padding:0 20px;color:#24313d}}label{{display:block;margin-top:14px}}input,select,button{{font:inherit;padding:8px;width:100%;box-sizing:border-box}}button{{margin-top:18px;background:#315f78;color:white;border:0}}pre{{background:#f4f6f7;padding:16px;white-space:pre-wrap}}</style></head>
<body><h1>Prediction interface scaffold</h1><p>This interface validates a request. No numerical backend is bundled.</p>
<form method="post"><label>Sequence<input name="sequence" required></label>
<label>Editor family<input name="editor_family" required></label>
<label>Task<input name="task" required></label>
<button type="submit">Validate request</button></form>{result}</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            encoded = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        self._send(FORM.format(result=""))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        fields = parse_qs(self.rfile.read(length).decode("utf-8"))
        try:
            payload = evaluate_request(
                fields.get("sequence", [""])[0],
                fields.get("editor_family", [""])[0],
                fields.get("task", [""])[0],
            )
            status = 503 if payload.get("status") == "backend_not_configured" else 200
        except RegistryError as exc:
            payload = {"status": "invalid_registry", "error": str(exc)}
            status = 503
        except (ValueError, KeyError, TypeError) as exc:
            payload = {"status": "invalid_request", "error": str(exc)}
            status = 400
        except Exception:
            payload = {"status": "invalid_request", "error": "request could not be processed"}
            status = 400
        result = "<pre>" + html.escape(json.dumps(payload, indent=2)) + "</pre>"
        self._send(FORM.format(result=result), status)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    host = os.environ.get("BE_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("BE_WEB_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Interface scaffold running on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
