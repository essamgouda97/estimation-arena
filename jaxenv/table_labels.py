"""Short names for action ids, with no dependency on JAX or the server."""

TRUMPS = ("♣", "♦", "♥", "♠", "NT")
RANKS = "23456789TJQKA"


def label(a: int) -> str:
    if a == 0:
        return "No Dash"
    if a == 1:
        return "Dash Call"
    if a <= 51:
        return f"{(a - 2) // 5 + 4}{TRUMPS[(a - 2) % 5]}"
    if a == 52:
        return "With"
    if a == 53:
        return "Pass"
    if a < 68:
        return f"Estimate {a - 54}"
    card = a - 68
    return RANKS[card % 13].replace("T", "10") + TRUMPS[card // 13]
