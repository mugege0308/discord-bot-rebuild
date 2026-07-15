from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorktreeRecord:
    path: Path
    head: str | None
    branch: str | None  # full ref, e.g. "refs/heads/foo", or None if detached/unknown
    detached: bool
    bare: bool
    locked: bool
    prunable: bool


def parse_worktree_list_porcelain_z(raw: bytes) -> list[WorktreeRecord] | None:
    """Parse the NUL-delimited output of `git worktree list --porcelain -z`.

    -z is used (and this parser requires it) specifically so a worktree path
    containing an embedded newline, drive letter, backslash, or non-ASCII
    text can never be misread: with -z, only NUL bytes ever separate
    fields/records, and NUL can never legally appear inside a path, so no
    path content can be confused with a delimiter. Decoding uses UTF-8 with
    surrogateescape, so arbitrary filesystem bytes always round-trip without
    raising - this parser never depends on core.quotePath or any other git
    quoting configuration, since -z output is never quoted in the first
    place.

    Returns None - never raises, never returns a partial list - if the
    input doesn't match the expected structure (most importantly: a field
    line appearing before any "worktree " header, or the input not ending
    in NUL, which -z output always does), so callers fail safely instead of
    silently acting on a misparsed or truncated record. Unknown field lines
    within an otherwise well-formed record are ignored rather than treated
    as fatal, so a future git version adding a new porcelain field doesn't
    break this parser.

    Note this only catches truncation that lands mid-field (the byte
    stream stops before its trailing NUL was ever written); truncation that
    happens to land exactly on a NUL boundary can look structurally
    complete to this parser even though trailing records are missing.
    Callers MUST check the underlying git-run result's stdout_truncated
    flag themselves before ever calling this function - that flag is the
    only fully reliable truncation signal, since it is set by
    dev_git_runner based on the actual number of bytes git produced, not on
    guessing from the (possibly coincidentally well-formed) output shape.
    """
    if raw == b"":
        return []

    if not raw.endswith(b"\0"):
        # -z output is always NUL-terminated; anything else is incomplete
        # (e.g. truncated mid-field) and must never be parsed as if it were
        # complete - a partial trailing record could otherwise be silently
        # accepted as valid data.
        return None

    # -z terminates every field (and the whole output) with NUL rather than
    # separating them; splitting on NUL and dropping the final empty
    # trailing token (from that trailing terminator) reconstructs the same
    # token stream the newline-delimited format expresses as lines, with an
    # empty token marking a record boundary exactly like a blank line does.
    tokens = raw.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens = tokens[:-1]

    entries: list[WorktreeRecord] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            entries.append(
                WorktreeRecord(
                    path=Path(current["path"]),
                    head=current["head"],
                    branch=current["branch"],
                    detached=current["detached"],
                    bare=current["bare"],
                    locked=current["locked"],
                    prunable=current["prunable"],
                )
            )
            current = None

    for token in tokens:
        if token == b"":
            flush()
            continue

        line = token.decode("utf-8", errors="surrogateescape")

        if line.startswith("worktree "):
            flush()
            current = {
                "path": line[len("worktree "):],
                "branch": None,
                "head": None,
                "detached": False,
                "bare": False,
                "locked": False,
                "prunable": False,
            }
            continue
        if current is None:
            # A field appeared before any "worktree " header - the input
            # isn't well-formed porcelain -z output at all.
            return None
        if line.startswith("HEAD "):
            current["head"] = line[len("HEAD "):]
        elif line.startswith("branch "):
            current["branch"] = line[len("branch "):]
        elif line == "detached":
            current["detached"] = True
        elif line == "bare":
            current["bare"] = True
        elif line == "locked" or line.startswith("locked "):
            current["locked"] = True
        elif line == "prunable" or line.startswith("prunable "):
            current["prunable"] = True
        # else: unknown field within a well-formed record - ignore.

    flush()
    return entries
