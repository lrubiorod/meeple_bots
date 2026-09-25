"""The dependency-free GUI handler keeps its route and error contract."""

from http import HTTPStatus
from io import BytesIO
import json
import unittest

from meeple_bots.gui.server import make_handler


class GuiHttpTests(unittest.TestCase):
    def test_routes_payloads_and_bad_requests(self):
        class Application:
            def __init__(self):
                self.state = {"status": "idle", "session_id": "current", "turn": 0}

            def snapshot(self):
                return self.state.copy()

            def start(self, payload):
                self.state = {"status": "waiting_human", "session_id": "current",
                              "turn": 0, "seed": payload["seed"]}
                return self.snapshot()

            def move(self, payload):
                if payload["session_id"] != self.state["session_id"]:
                    raise ValueError("this decision is no longer active")
                self.state["turn"] += 1
                return self.snapshot()

        application = Application()
        handler_type = make_handler(application, "<p>GUI</p>")

        def request(method, path, payload=None):
            # BaseHTTPRequestHandler's I/O contract without opening a socket.
            handler = handler_type.__new__(handler_type)
            handler.path = path
            raw = b"" if payload is None else json.dumps(payload).encode()
            handler.headers = {"Content-Length": str(len(raw))}
            handler.rfile = BytesIO(raw)
            handler.wfile = BytesIO()
            response = {}
            handler.send_response = lambda status: response.setdefault("status", status)
            handler.send_header = lambda name, value: response.setdefault(name, value)
            handler.end_headers = lambda: None
            getattr(handler, method)()
            response["body"] = handler.wfile.getvalue()
            return response

        page = request("do_GET", "/")
        self.assertEqual((page["status"], page["Content-Type"], page["body"]),
                         (HTTPStatus.OK, "text/html; charset=utf-8", b"<p>GUI</p>"))
        self.assertEqual(json.loads(request("do_GET", "/api/state")["body"]),
                         application.snapshot())
        started = request("do_POST", "/api/start", {"seed": 42})
        self.assertEqual(started["status"], HTTPStatus.OK)
        self.assertEqual(json.loads(started["body"]),
                         {"status": "waiting_human", "session_id": "current", "turn": 0,
                          "seed": 42})
        moved = request("do_POST", "/api/move", {"session_id": "current"})
        self.assertEqual(json.loads(moved["body"])["turn"], 1)
        stale = request("do_POST", "/api/move", {"session_id": "stale"})
        self.assertEqual(stale["status"], HTTPStatus.BAD_REQUEST)
        self.assertEqual(json.loads(stale["body"]),
                         {"error": "this decision is no longer active"})
        missing = request("do_GET", "/missing")
        self.assertEqual(missing["status"], HTTPStatus.NOT_FOUND)
        self.assertEqual(json.loads(missing["body"]), {"error": "not found"})
