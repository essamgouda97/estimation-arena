import pytest

from arena.bots import load_manifest, make_submission, parse_sub
from estimation.engine import RulesConfig


def test_parse_sub_options():
    assert parse_sub("sub:table-pressure") == ("table-pressure", {})
    assert parse_sub("sub:table-pressure:pressure=0.5:shift=-1") == (
        "table-pressure",
        {"pressure": "0.5", "shift": "-1"},
    )
    with pytest.raises(ValueError):
        parse_sub("sub:table-pressure:pressure")


def test_python_bot_takes_options_and_rejects_unknown_ones():
    manifest = load_manifest("table-pressure")
    bot = make_submission(manifest, 0, RulesConfig(), {"pressure": "0.5"})
    assert bot.pressure == 0.5
    with pytest.raises(ValueError, match="no option"):
        make_submission(manifest, 0, RulesConfig(), {"bogus": "1"})


def test_wire_bot_rejects_options():
    with pytest.raises(ValueError, match="no options"):
        make_submission(load_manifest("lowcard-wire"), 0, RulesConfig(), {"x": "1"})
