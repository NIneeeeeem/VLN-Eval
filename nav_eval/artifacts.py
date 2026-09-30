"""Integrity helpers shared by collection and evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path

ARTIFACTS = ("run.json", "events.jsonl", "episodes.jsonl", "evidence/episodes.jsonl")


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def artifact_digests(root):
    return {name: digest_bytes((Path(root) / name).read_bytes()) for name in ARTIFACTS}
