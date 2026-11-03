"""The live table: a person's seat over the WebSocket, bots in the others, hands kept private."""

import pytest
from starlette.testclient import TestClient

from arena import serve, table


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    for name in ("BOT_PAUSE_S", "TRICK_PAUSE_S", "ROUND_PAUSE_S"):
        monkeypatch.setattr(table, name, 0.0)
    serve.TABLES.clear()


def make(client, seats):
    res = client.post("/api/tables", json={"seats": seats})
    assert res.status_code == 200, res.text
    return res.json()


def test_unlisted_bot_is_refused():
    with TestClient(serve.app()) as client:
        res = client.post("/api/tables", json={"seats": [{"kind": "bot", "spec": "rl:checkpoint=/etc/passwd"}] * 4})
        assert res.status_code == 400


def test_person_plays_a_round_against_bots():
    with TestClient(serve.app()) as client:
        made = make(client, [{"kind": "human", "name": "Ann"}] + [{"kind": "bot", "spec": "heuristic"}] * 3)
        token = made["seats"][0]["link"].split("#", 1)[1]
        with client.websocket_connect(f"/ws/{made['id']}") as ws:
            ws.send_json({"type": "hello", "token": token})
            moves = 0
            while True:
                msg = ws.receive_json()
                assert msg["type"] == "state", msg
                s = msg["state"]
                assert s["me"]["seat"] == 0
                if s["round"] >= 2:
                    break
                if s["turn"] == 0 and s["me"]["legal"]:
                    ws.send_json({"type": "act", "action": s["me"]["legal"][0]["id"]})
                    moves += 1
            assert moves >= 1
            assert s["last_round"]["round"] == 1


def test_spectator_sees_no_hand_and_cannot_move():
    with TestClient(serve.app()) as client:
        made = make(client, [{"kind": "human", "name": "Ann"}] + [{"kind": "bot", "spec": "heuristic"}] * 3)
        with client.websocket_connect(f"/ws/{made['id']}") as ws:
            ws.send_json({"type": "hello", "token": ""})
            s = ws.receive_json()["state"]
            assert s["me"] is None and "hand" not in s
            ws.send_json({"type": "act", "action": 0})
        assert serve.TABLES[made["id"]].events == []  # the Dash decision is still seat 0's


def test_gateway_guard_needs_key_and_identity():
    guarded = serve.GatewayGuard(serve.app(), "k" * 64, "estimation")
    with TestClient(guarded) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/bots").status_code == 403
        key = {"X-Home-Estimation-Key": "k" * 64}
        assert client.get("/api/bots", headers={**key, "Remote-User": "x", "Remote-Groups": "household"}).status_code == 403
        assert client.get("/api/bots", headers={**key, "Remote-User": "egouda", "Remote-Groups": "owners"}).status_code == 200
        assert client.get("/api/bots", headers={**key, "Remote-User": "friend", "Remote-Groups": "household,service:estimation"}).status_code == 200
