"""Dependency-free localhost web server for the Meeple Bots GUI."""

from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

_MAX_REQUEST_BYTES = 64 * 1024


class GuiApplication(Protocol):
    """Game-specific operations required by the generic HTTP server."""

    def start(self, payload: dict[str, Any]) -> dict[str, object]: ...

    def move(self, payload: dict[str, Any]) -> dict[str, object]: ...

    def snapshot(self) -> dict[str, object]: ...

    def cancel(self) -> None: ...


def make_handler(application: GuiApplication, page: str):
    class GuiHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self._send_bytes(page.encode(), "text/html; charset=utf-8")
                return
            if self.path == "/api/state":
                self._send_json(application.snapshot())
                return
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
                if self.path == "/api/start":
                    result = application.start(payload)
                elif self.path == "/api/move":
                    result = application.move(payload)
                else:
                    self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
                    return
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(result)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("Content-Length is required")
            length = int(raw_length)
            if not 0 <= length <= _MAX_REQUEST_BYTES:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _send_json(
            self,
            payload: dict[str, object],
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self._send_bytes(
                json.dumps(payload).encode(),
                "application/json; charset=utf-8",
                status,
            )

        def _send_bytes(
            self,
            body: bytes,
            content_type: str,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return GuiHandler


def run_gui(
    *,
    game: str = "tic-tac-toe",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Select and serve one supported game's GUI."""

    if game == "connect-four":
        from ..games.connect_four.gui import PAGE, ConnectFourApplication

        application = ConnectFourApplication()
    elif game == "tic-tac-toe":
        from ..games.tic_tac_toe.gui import PAGE, TicTacToeApplication

        application = TicTacToeApplication()
    else:
        raise ValueError(
            f"graphical interface is not available for {game}; "
            "available games: connect-four, tic-tac-toe"
        )

    serve_gui(
        application,
        PAGE,
        host=host,
        port=port,
        open_browser=open_browser,
    )


def serve_gui(
    application: GuiApplication,
    page: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """Serve one game-specific GUI until interrupted."""

    if not 0 <= port <= 65_535:
        raise ValueError("port must be between 0 and 65535")
    server = ThreadingHTTPServer((host, port), make_handler(application, page))
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"Meeple Bots GUI: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Meeple Bots GUI")
    finally:
        application.cancel()
        server.server_close()
