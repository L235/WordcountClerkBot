"""Batched, revision-pinned reads and atomic local publication checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path


def _batches(items, size=50):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def read_revisions(site, titles: list[str]) -> dict[str, int]:
    """Read current revision IDs; 0 denotes an explicitly missing page."""
    revisions = {}
    for batch in _batches(list(dict.fromkeys(titles))):
        data = site.simple_request(
            action="query", formatversion=2, prop="revisions",
            rvprop="ids", titles="|".join(batch),
        ).submit()["query"]
        aliases = {row["from"]: row["to"] for row in data.get("normalized", [])}
        pages = {row["title"]: row for row in data["pages"]}
        for title in batch:
            page = pages[aliases.get(title, title)]
            if "invalid" in page:
                raise RuntimeError(f"Invalid page title: {title}")
            revisions[title] = 0 if "missing" in page else page["revisions"][0]["revid"]
    return revisions


def read_contents(site, revisions: dict[str, int]) -> dict[str, str]:
    """Fetch the precise revisions checked, not newer edits made during a pass."""
    texts = {title: "" for title, revision in revisions.items() if revision == 0}
    ids = list(dict.fromkeys(rev for rev in revisions.values() if rev))
    by_id = {}
    for batch in _batches(ids):
        data = site.simple_request(
            action="query", formatversion=2, prop="revisions", rvslots="main",
            rvprop="ids|content", revids="|".join(map(str, batch)),
        ).submit()["query"]
        for page in data.get("pages", []):
            for revision in page.get("revisions", []):
                content = revision["slots"]["main"]["content"]
                if not isinstance(content, str):
                    raise RuntimeError("Revision content unavailable")
                by_id[revision["revid"]] = content
    for title, revision in revisions.items():
        if revision:
            # Missing/suppressed revisions must fail, never become empty input.
            texts[title] = by_id[revision]
    return texts


def load_checkpoint(path: Path) -> dict:
    try:
        state = json.loads(path.read_text())
        if (not isinstance(state, dict)
                or not isinstance(state.get("sources"), dict)
                or not isinstance(state.get("outputs"), dict)
                or not isinstance(state.get("cases"), list)
                or not all(isinstance(case, str) for case in state["cases"])
                or not isinstance(state.get("refreshed_at"), (int, float))):
            return {}
        return state
    except (OSError, ValueError):
        return {}


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as file:
        json.dump(state, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
