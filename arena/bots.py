"""Submissions: ``bots/<name>/manifest.json`` describes one bot.

    {
      "name": "lowcard-wire",            # folder name; letters, digits, - and _
      "author": "Essam",
      "kind": "wire",                    # "wire": a command speaking the protocol; "python": a factory
      "command": ["python3", "bot.py"],  # wire: run inside the bot's folder, one process per game
      "entry": "bot.py:make",            # python: file and function make(seed, rules) -> Bot
      "description": "one line",
      "timeout_s": 5                     # optional hard limit per decision (wire only)
    }

A lineup names a submission as ``sub:<name>``, or ``sub:<name>:key=value:...`` to pass string
options to a python bot's factory as ``make(seed, rules, key=value)`` (for parameter sweeps);
anything else is an engine bot spec such as ``heuristic`` or ``pimc:samples=8`` (see
``estimation.bots.registry``).
"""

from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from estimation.bots import Bot, make_bot
from estimation.engine import RulesConfig

from .protocol import DEFAULT_TIMEOUT_S, WireBot

ROOT = Path(__file__).resolve().parents[1]
BOTS_DIR = ROOT / "bots"
SUB_PREFIX = "sub:"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


@dataclass(frozen=True, slots=True)
class Manifest:
    name: str
    author: str
    kind: str
    description: str
    command: tuple[str, ...]
    entry: str
    timeout_s: float
    folder: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "author": self.author,
            "kind": self.kind,
            "description": self.description,
            "command": list(self.command),
            "entry": self.entry,
            "timeout_s": self.timeout_s,
        }


def list_submissions() -> list[str]:
    return sorted(p.parent.name for p in BOTS_DIR.glob("*/manifest.json"))


def load_manifest(name: str) -> Manifest:
    if not NAME_RE.match(name):
        raise ValueError(f"bad bot name {name!r}")
    folder = BOTS_DIR / name
    path = folder / "manifest.json"
    if not path.is_file():
        raise ValueError(f"no submission named {name!r}: {path} is missing")
    data = json.loads(path.read_text())
    kind = data.get("kind")
    if kind not in ("wire", "python"):
        raise ValueError(f"{path}: kind must be 'wire' or 'python', got {kind!r}")
    if data.get("name", name) != name:
        raise ValueError(f"{path}: name {data['name']!r} does not match folder {name!r}")
    command = tuple(str(c) for c in data.get("command", ()))
    entry = str(data.get("entry", ""))
    if kind == "wire" and not command:
        raise ValueError(f"{path}: a wire bot needs a command")
    if kind == "python" and ":" not in entry:
        raise ValueError(f"{path}: a python bot needs entry 'file.py:function'")
    return Manifest(
        name=name,
        author=str(data.get("author", "")),
        kind=kind,
        description=str(data.get("description", "")),
        command=command,
        entry=entry,
        timeout_s=float(data.get("timeout_s", DEFAULT_TIMEOUT_S)),
        folder=folder,
    )


def parse_sub(spec: str) -> tuple[str, dict[str, str]]:
    """``sub:name:key=value:...`` as the name and its options."""
    name, *pairs = spec[len(SUB_PREFIX) :].split(":")
    options: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"bot option {pair!r} must look like key=value")
        options[key] = value
    return name, options


def make_submission(
    manifest: Manifest, seed: int, rules: RulesConfig, options: dict[str, str] | None = None
) -> Bot:
    if manifest.kind == "wire":
        if options:
            raise ValueError(f"{manifest.name} is a wire bot; it takes no options")
        return WireBot(manifest.command, manifest.folder, seed, rules, manifest.timeout_s)
    file, function = manifest.entry.split(":", 1)
    path = manifest.folder / file
    spec = importlib.util.spec_from_file_location(f"arena_bot_{manifest.name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, function)
    return factory(seed, rules, **(options or {}))


def maker(spec: str, seed: int, rules: RulesConfig) -> Bot:
    """The arena's ``BotMaker``: a submission for ``sub:<name>``, else an engine bot."""
    if spec.startswith(SUB_PREFIX):
        name, options = parse_sub(spec)
        return make_submission(load_manifest(name), seed, rules, options)
    return make_bot(spec, seed, rules)
