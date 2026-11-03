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


def test_sure_trumps_counts_only_trumps_above_every_outstanding_trump():
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("tp", "bots/table-pressure/bot.py")
    tp = module_from_spec(spec)
    spec.loader.exec_module(tp)
    spades = 3 * 13  # card id of the Spades two; the Ace is +12
    hand = (1 << (spades + 12)) | (1 << (spades + 11)) | (1 << (spades + 2))  # A, K, 4 of Spades
    out = 1 << (spades + 9)  # the Jack is still out
    assert tp.sure_trumps(hand, 3, out) == 2  # A and K are sure, the 4 is not
    assert tp.sure_trumps(hand, 3, 0) == 3  # no trump out: all three are sure
    assert tp.sure_trumps(hand, 4, out) == 0  # Sans has no trumps
