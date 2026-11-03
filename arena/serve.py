"""Serve live tables: people and bots in any mix of seats.

    uv run python -m arena.serve --port 8790        # then open http://127.0.0.1:8790/

The lobby creates a table: each seat is a person (an invite link) or one of the listed bots.
People play in the browser; a program can take a person's seat with the same WebSocket JSON
(see docs/table.md). Only bots on the list can sit, so no request picks arbitrary files.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
import time
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from .bots import ROOT, SUB_PREFIX, list_submissions, load_manifest
from .run import load_anchors
from .table import Seat, Table

STATIC = Path(__file__).resolve().parent / "static"
MAX_TABLES = 24
IDLE_S = 12 * 3600
TABLES: dict[str, Table] = {}


def available_bots() -> list[dict[str, str]]:
    """Bots that can take a seat: the anchors whose files are here, then the submissions."""
    bots = [{"spec": a["spec"], "name": a["name"], "author": a.get("author", ""), "group": "anchor"} for a in load_anchors()]
    for name in list_submissions():
        m = load_manifest(name)
        if m.needs and not (ROOT / m.needs).exists():
            continue
        bots.append({"spec": SUB_PREFIX + name, "name": name, "author": m.author, "group": "submission"})
    return bots


def _prune() -> None:
    now = time.time()
    for tid, t in list(TABLES.items()):
        if not t.clients and now - t.created > IDLE_S:
            TABLES.pop(tid)


async def bots(_: Request) -> Response:
    return JSONResponse(available_bots())


async def create(request: Request) -> Response:
    body = await request.json()
    _prune()
    if len(TABLES) >= MAX_TABLES:
        return JSONResponse({"error": "too many open tables"}, status_code=429)
    allowed = {b["spec"]: b["name"] for b in available_bots()}
    seats = []
    for k, s in enumerate((body.get("seats") or [])[:4]):
        if s.get("kind") == "bot":
            if s.get("spec") not in allowed:
                return JSONResponse({"error": f"unknown bot {s.get('spec')!r}"}, status_code=400)
            seats.append(Seat("bot", allowed[s["spec"]], s["spec"]))
        else:
            name = str(s.get("name") or f"Player {k + 1}").strip()[:24] or f"Player {k + 1}"
            seats.append(Seat("human", name))
    if len(seats) != 4:
        return JSONResponse({"error": "a table needs 4 seats"}, status_code=400)
    tid = secrets.token_urlsafe(6)
    table = Table(tid, seats, random.randrange(1, 2**31))
    TABLES[tid] = table
    table.kick()  # an all-bot table starts playing at once
    return JSONResponse({
        "id": tid,
        "watch": f"/t/{tid}",
        "seats": [{"name": s.name, "kind": s.kind, "link": f"/t/{tid}#{s.token}" if s.kind == "human" else None} for s in seats],
    })


async def tables(_: Request) -> Response:
    _prune()
    rows = [{"id": t.id, "names": [s.name for s in t.seats], "round": t.bola.round_no, "over": t.bola.is_over,
             "watchers": len(t.clients)} for t in sorted(TABLES.values(), key=lambda t: -t.created)]
    return JSONResponse(rows)


async def page(_: Request) -> Response:
    return FileResponse(STATIC / "index.html")


async def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "tables": len(TABLES)})


async def socket(ws: WebSocket) -> None:
    table = TABLES.get(ws.path_params["tid"])
    await ws.accept()
    if table is None:
        await ws.send_json({"type": "error", "error": "no such table"})
        await ws.close()
        return
    # The first message says who is connecting: {"type": "hello", "token": <invite secret or "">}.
    # The secret is never in the URL, so it stays out of access logs.
    try:
        hello = json.loads(await ws.receive_text())
    except (WebSocketDisconnect, ValueError):
        return
    token = str(hello.get("token") or "") if isinstance(hello, dict) else ""
    seat = table.seat_of(token)
    if token and seat is None:
        await ws.send_json({"type": "error", "error": "that invite link is not valid"})
    me = (ws, seat)
    table.clients.add(me)
    if seat is not None:
        table.seats[seat].clients += 1
    await table.broadcast()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if seat is None:
                continue  # spectators only watch
            if msg.get("type") == "act":
                error = await table.play_human(seat, int(msg.get("action", -1)))
                if error:
                    await ws.send_json({"type": "error", "error": error})
            elif msg.get("type") == "new_game" and table.bola.is_over:
                async with table.lock:
                    table.new_game(random.randrange(1, 2**31))
                await table.broadcast()
                table.kick()
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        table.clients.discard(me)
        if seat is not None:
            table.seats[seat].clients -= 1
        await table.broadcast()


class GatewayGuard:
    """Behind the home server's gateway: every request but /health must carry the private
    proxy key the reverse proxy adds, and a gateway identity allowed to use the table
    (``owners``, or the ``service:<id>`` group). Off when no key file is configured, as on a
    laptop."""

    def __init__(self, app, key: str, service: str) -> None:
        self.app, self.key, self.groups = app, key.encode(), {"owners", f"service:{service}"}

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope["path"] != "/health":
            headers = dict(scope["headers"])
            groups = set(headers.get(b"remote-groups", b"").decode().split(","))
            ok = secrets.compare_digest(headers.get(b"x-home-estimation-key", b""), self.key)
            if not (ok and headers.get(b"remote-user") and groups & self.groups):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                else:
                    await JSONResponse({"error": "forbidden"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)


def app() -> Starlette:
    return Starlette(routes=[
        Route("/", page),
        Route("/t/{tid}", page),
        Route("/health", health),
        Route("/api/bots", bots),
        Route("/api/tables", tables, methods=["GET"]),
        Route("/api/tables", create, methods=["POST"]),
        WebSocketRoute("/ws/{tid}", socket),
        Mount("/static", StaticFiles(directory=STATIC)),
    ])


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m arena.serve", description=__doc__.split("\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--proxy-key-file", type=Path, help="require this key from the reverse proxy")
    parser.add_argument("--service", default="estimation", help="gateway service id for the group check")
    args = parser.parse_args(argv)
    asgi = app()
    if args.proxy_key_file:
        asgi = GatewayGuard(asgi, args.proxy_key_file.read_text().strip(), args.service)
    uvicorn.run(asgi, host=args.host, port=args.port, proxy_headers=True, access_log=False, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
