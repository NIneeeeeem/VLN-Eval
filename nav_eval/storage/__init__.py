"""Episode commits are the source of truth; JSONL reports are derived views."""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from nav_eval.plugins import digest

ROLLOUT_SCHEMA = "nav-eval-rollout/0.4"


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("wb") as handle:
        handle.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def run_lock(root):
    import fcntl
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("run already has an active owner") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class AttemptStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "attempts").mkdir(exist_ok=True)

    def attempts(self, episode_id):
        parent = self.root / "attempts" / digest(episode_id)
        result = []
        for path in sorted(parent.glob("*.json")):
            attempt = read_json(path)
            if attempt.get("sha256") != digest(attempt["data"]):
                raise ValueError(f"attempt integrity failure: {path}")
            data = attempt["data"]
            if data["record"]["episode_id"] != episode_id:
                raise ValueError("attempt belongs to a different episode")
            result.append(data)
        return result

    def commit(self, record, evidence, events):
        parent = self.root / "attempts" / digest(record["episode_id"])
        parent.mkdir(exist_ok=True)
        index = len(self.attempts(record["episode_id"]))
        data = {"record": record, "evidence": evidence, "events": events}
        path = parent / f"{index:06d}-{record['attempt_id']}.json"
        if path.exists():
            raise ValueError("attempt already committed")
        write_json(path, {"data": data, "sha256": digest(data)})

    def canonical(self, episode_id):
        attempts = self.attempts(episode_id)
        return next((a for a in attempts if a["record"]["status"] in {"completed", "policy_error"}),
                    attempts[-1] if attempts else None)

    def export(self, episode_ids):
        records, evidence, events = [], [], []
        for episode_id in episode_ids:
            attempt = self.canonical(episode_id)
            if attempt is None:
                record = {"episode_id": episode_id, "attempt_id": "pending", "status": "pending"}
                blob = None
            else:
                record, blob = attempt["record"], attempt["evidence"]
                events.extend(attempt["events"])
            records.append(record)
            evidence.append({"episode_id": episode_id, "attempt_id": record["attempt_id"], "evidence": blob})
        for name, rows in (("episodes.jsonl", records), ("evidence/episodes.jsonl", evidence), ("events.jsonl", events)):
            atomic_write(self.root / name, "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
        return records


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()
