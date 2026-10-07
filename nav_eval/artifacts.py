"""Integrity helpers shared by collection and evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path

from nav_eval.storage import file_digest

ARTIFACTS = ("run.json", "events.jsonl", "episodes.jsonl", "evidence/episodes.jsonl")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def artifact_digests(root):
    return {name: file_digest(Path(root) / name) for name in ARTIFACTS}
