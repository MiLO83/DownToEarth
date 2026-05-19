"""
live_server.py — WebSocket bridge: refine loop → WebXR viewer.

Architecture:
  - Pipeline runs in main thread, calls `broadcast()` after every iteration
  - WebSocket server runs in a background asyncio loop
  - Connected viewers receive every snapshot pushed via broadcast
  - Snapshots are voxel state JSON (compact: list of [ix, iy, iz, cls, conf]
    tuples + metadata)
  - Static HTTP also serves the viewer's HTML/JS so you only need one URL

Run as a standalone server (it'll wait for pipeline to push state):
    python -m pipeline.live_server

Or call broadcast() from your script after each iteration.
"""
from __future__ import annotations
import asyncio
import json
import pathlib
import threading
from typing import Any
import http.server
import socketserver

import websockets


_VIEWER_ROOT = pathlib.Path(__file__).resolve().parent.parent / "viewer" / "public"
_RUNS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "runs"


class LiveServer:
    """One-server-instance-per-pipeline-run. Keeps an asyncio loop in a
    background thread and a static HTTP server on a separate port."""

    def __init__(self, ws_port: int = 8765, http_port: int = 5174):
        self.ws_port = ws_port
        self.http_port = http_port
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._latest_snapshot: dict | None = None
        self._http_httpd: socketserver.TCPServer | None = None
        self._http_thread: threading.Thread | None = None

    # ─── Public ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start both servers in background threads."""
        self._start_ws()
        self._start_http()
        print(f"[live] WebSocket on ws://localhost:{self.ws_port}")
        print(f"[live] HTTP viewer on http://localhost:{self.http_port}")
        print(f"[live] open the viewer in a browser to watch live")

    def broadcast(self, payload: dict) -> None:
        """Send a JSON snapshot to all connected viewers. Safe to call from
        any thread."""
        if not self._loop:
            return
        self._latest_snapshot = payload
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    def shutdown(self) -> None:
        if self._http_httpd:
            self._http_httpd.shutdown()
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ─── WS internals ────────────────────────────────────────────────────

    def _start_ws(self) -> None:
        def runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._ws_main())
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()

    async def _ws_main(self) -> None:
        async with websockets.serve(self._handle_client, "0.0.0.0", self.ws_port):
            await asyncio.Future()  # run forever

    async def _handle_client(self, ws) -> None:
        self._clients.add(ws)
        try:
            # On connect, send the latest snapshot so the viewer hydrates
            if self._latest_snapshot is not None:
                await ws.send(json.dumps(self._latest_snapshot, separators=(",", ":")))
            async for _msg in ws:
                pass    # viewer is read-only for now
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _broadcast_async(self, payload: dict) -> None:
        if not self._clients:
            return
        body = json.dumps(payload, separators=(",", ":"))
        dead = []
        for ws in self._clients:
            try:
                await ws.send(body)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # ─── HTTP internals ──────────────────────────────────────────────────

    def _start_http(self) -> None:
        root = _VIEWER_ROOT
        ws_port = self.ws_port

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root), **kwargs)
            def end_headers(self):
                # Inject the WS port into a global JS var via a tiny shim
                if self.path == "/":
                    pass  # we let index.html include the runtime config
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                super().end_headers()
            def log_message(self, *args, **kwargs): pass    # silence default logs

        runs_root = _RUNS_ROOT

        # Inject ws port via a config endpoint + serve /runs/<scene>/ from runs/
        class ConfigHandler(Handler):
            def do_GET(self):
                if self.path == "/__config__":
                    body = json.dumps({"ws_port": ws_port}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.startswith("/runs/"):
                    rel = self.path[len("/runs/"):].split("?")[0]
                    fp = (runs_root / rel).resolve()
                    if fp.is_file() and str(fp).startswith(str(runs_root.resolve())):
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json"
                                         if fp.suffix == ".json" else "application/octet-stream")
                        self.send_header("Content-Length", str(fp.stat().st_size))
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        with open(fp, "rb") as f:
                            self.wfile.write(f.read())
                        return
                super().do_GET()

        self._http_httpd = socketserver.TCPServer(("0.0.0.0", self.http_port), ConfigHandler)
        self._http_thread = threading.Thread(target=self._http_httpd.serve_forever, daemon=True)
        self._http_thread.start()


if __name__ == "__main__":
    srv = LiveServer()
    srv.start()
    # Keep the main thread alive so the daemon threads keep running.
    try:
        import time
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        srv.shutdown()
