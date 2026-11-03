#!/usr/bin/env python3
"""Wire bot that plays the highest legal card; otherwise the same choices as lowcard-wire."""
import json
import sys

PASS = 53
NO_DASH = 0


def choose(obs: dict, legal: list[int]) -> int:
    cards = [a for a in legal if a >= 68]
    if cards:
        return max(cards)
    if PASS in legal:
        return PASS
    if NO_DASH in legal:
        return NO_DASH
    return min(legal)


def main() -> None:
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] == "act":
            sys.stdout.write(f"{choose(msg['obs'], msg['legal'])}\n")
            sys.stdout.flush()
        elif msg["type"] == "bye":
            return


if __name__ == "__main__":
    main()
