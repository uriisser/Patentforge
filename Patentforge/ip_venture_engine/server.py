"""
Minimal localhost web server for IP-Native Venture Engine.
Uses only the Python standard library.

Run with:
    python -m ip_venture_engine.server
Then open: http://localhost:8000
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from .domains import list_domains, get_domain_by_id
from .engine import load_patent_texts, run_engine

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PORT = 8000


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class VentureHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # noqa: A002
        print(f"  {self.address_string()} - {format % args}")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html; charset=utf-8")
        elif path == "/api/domains":
            self._json_response(list_domains())
        elif path == "/api/patents":
            patents = load_patent_texts()
            self._json_response([fname for fname, _ in patents])
        else:
            self._not_found()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                context = json.loads(body)
            except json.JSONDecodeError:
                self._error(400, "Invalid JSON body")
                return

            # Basic validation
            required = ("domain_id", "domain_label",
                        "patent_start_year", "patent_end_year", "current_year")
            missing = [k for k in required if k not in context]
            if missing:
                self._error(400, f"Missing fields: {missing}")
                return

            patents = load_patent_texts()
            results = run_engine(patents, context)
            self._json_response(results)
        else:
            self._not_found()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _serve_file(self, filename: str, content_type: str):
        fpath = os.path.join(STATIC_DIR, filename)
        try:
            with open(fpath, "rb") as fh:
                data = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self._not_found()

    def _json_response(self, payload, status: int = 200):
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str):
        self._json_response({"error": message}, status=status)

    def _not_found(self):
        self._error(404, "Not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(port: int = PORT):
    server = HTTPServer(("localhost", port), VentureHandler)
    print(f"\n  IP-Native Venture Engine -- Web UI")
    print(f"  Running at:  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    run_server()
