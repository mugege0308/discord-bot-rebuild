"""Build the generic Danbooru tag SQLite snapshot.

The importer is intentionally separate from runtime search.  It supports the
official JSON API as well as a simple CSV/JSON snapshot so an API mirror or a
previously downloaded export can be used when Danbooru rate limiting or
Cloudflare prevents a direct sync.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import requests

from services.danbooru_tag_store import (
    DANBOORU_DB_SCHEMA_VERSION,
    DEFAULT_DANBOORU_DB_PATH,
    create_schema,
    normalize_tag_name,
    rows_from_documents,
)


logger = logging.getLogger(__name__)
DEFAULT_API_BASE_URL = "https://danbooru.donmai.us"
DEFAULT_PAGE_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 30
_CATEGORY_VALUES = {"general", "artist", "copyright", "character", "meta"}


class DanbooruImportError(RuntimeError):
    pass


def _request_json(
    session: requests.Session,
    url: str,
    *,
    params: Mapping[str, object],
    timeout_seconds: float,
) -> list[dict[str, object]]:
    response = session.get(url, params=params, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise DanbooruImportError(f"Danbooru API response is not a list: {url}")
    return [value for value in payload if isinstance(value, dict)]


def fetch_api_collection(
    session: requests.Session,
    endpoint: str,
    *,
    base_url: str = DEFAULT_API_BASE_URL,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_pages: int | None = None,
) -> tuple[list[dict[str, object]], float]:
    """Fetch a paginated Danbooru collection and return rows plus elapsed ms."""

    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    page = 1
    while True:
        params: dict[str, object] = {
            "limit": min(max(int(page_size), 1), 1000),
            "page": page,
        }
        if endpoint == "tags":
            # Keep deprecated rows in the snapshot so the runtime can make
            # the non-deprecated policy decision locally.
            params["search[order]"] = "id"
        else:
            params["search[status]"] = "active"
        batch = _request_json(
            session,
            f"{base_url.rstrip('/')}/{endpoint}.json",
            params=params,
            timeout_seconds=timeout_seconds,
        )
        rows.extend(batch)
        if len(batch) < params["limit"] or (max_pages is not None and page >= max_pages):
            break
        page += 1
    return rows, (time.perf_counter() - started) * 1000


def fetch_official_snapshot(
    *,
    base_url: str = DEFAULT_API_BASE_URL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_pages: int | None = None,
    session: requests.Session | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], float]:
    own_session = session is None
    client = session or requests.Session()
    try:
        tags, tags_ms = fetch_api_collection(
            client,
            "tags",
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_pages=max_pages,
        )
        aliases, aliases_ms = fetch_api_collection(
            client,
            "tag_aliases",
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_pages=max_pages,
        )
        implications, implications_ms = fetch_api_collection(
            client,
            "tag_implications",
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_pages=max_pages,
        )
        return tags, aliases, implications, tags_ms + aliases_ms + implications_ms
    finally:
        if own_session:
            client.close()


def parse_snapshot_file(path: Path | str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Read JSON or the common one-row-per-line Danbooru tag CSV snapshot."""

    source = Path(path)
    # Public exports occasionally contain a legacy non-UTF8 alias byte.  The
    # canonical tag name/count fields are ASCII; replacement keeps the row
    # importable without allowing a malformed alias to abort the snapshot.
    raw = source.read_text(encoding="utf-8-sig", errors="replace")
    if source.suffix.casefold() == ".json":
        payload = json.loads(raw)
        if isinstance(payload, dict):
            tag_rows = payload.get("tags", payload.get("data", []))
            alias_rows = payload.get("aliases", [])
            implication_rows = payload.get("implications", [])
        else:
            tag_rows, alias_rows, implication_rows = payload, [], []
        return (
            [value for value in tag_rows if isinstance(value, dict)],
            [value for value in alias_rows if isinstance(value, dict)],
            [value for value in implication_rows if isinstance(value, dict)],
        )

    tags: list[dict[str, object]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # k39ink/tag-rag's snapshot is an outer-quoted CSV row ending in ';'.
        if line.startswith('"') and line.endswith('";'):
            line = line[1:-2].replace('""', '"')
        values = next(csv.reader([line]))
        if not values:
            continue
        name = values[0].strip()
        if not name or name.casefold() in {"tag", "name"}:
            continue
        try:
            category = int(values[1]) if len(values) > 1 else 0
        except ValueError:
            category = values[1] if len(values) > 1 else "general"
        try:
            post_count = int(values[2]) if len(values) > 2 else 0
        except ValueError:
            post_count = 0
        alias_text = values[3].strip().strip('"') if len(values) > 3 else ""
        aliases = [item.strip() for item in alias_text.split(",") if item.strip()]
        tags.append(
            {
                "name": name,
                "category": category,
                "post_count": post_count,
                "aliases": aliases,
            }
        )
    alias_rows: list[dict[str, object]] = []
    for row in tags:
        for alias in row.pop("aliases", []):
            alias_rows.append(
                {
                    "antecedent_name": alias,
                    "consequent_name": row["name"],
                    "status": "active",
                }
            )
    return tags, alias_rows, []


def _insert_snapshot(
    connection: sqlite3.Connection,
    tags: Sequence[Mapping[str, object]],
    aliases: Sequence[Mapping[str, str]],
    implications: Sequence[Mapping[str, str]],
    *,
    snapshot_version: str,
    source: str,
    api_elapsed_ms: float | None,
    source_id_kind: str,
) -> None:
    create_schema(connection)
    connection.execute("DELETE FROM tags_fts")
    connection.execute("DELETE FROM tag_aliases")
    connection.execute("DELETE FROM tag_implications")
    connection.execute("DELETE FROM tags")
    connection.execute("DELETE FROM metadata")
    connection.executemany(
        """
        INSERT INTO tags(tag_id, name, normalized_name, prompt_tag, category, post_count, is_deprecated)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                int(row["tag_id"]),
                str(row["name"]),
                str(row["normalized_name"]),
                str(row["prompt_tag"]),
                str(row["category"]),
                int(row["post_count"]),
                int(bool(row["is_deprecated"])),
            )
            for row in tags
        ],
    )
    connection.executemany(
        """
        INSERT OR IGNORE INTO tag_aliases(
            antecedent_name, normalized_antecedent, consequent_name,
            normalized_consequent, status
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                str(row["antecedent_name"]),
                str(row["normalized_antecedent"]),
                str(row["consequent_name"]),
                str(row["normalized_consequent"]),
                str(row.get("status", "active")),
            )
            for row in aliases
        ],
    )
    connection.executemany(
        "INSERT OR IGNORE INTO tag_implications(antecedent_name, consequent_name, status) VALUES (?, ?, ?)",
        [
            (
                str(row["antecedent_name"]),
                str(row["consequent_name"]),
                str(row.get("status", "active")),
            )
            for row in implications
        ],
    )
    alias_map: dict[int, list[str]] = {}
    tag_id_by_name = {str(row["normalized_name"]): int(row["tag_id"]) for row in tags}
    for alias in aliases:
        target_id = tag_id_by_name.get(str(alias["normalized_consequent"]))
        if target_id is not None and str(alias.get("status", "active")) == "active":
            alias_map.setdefault(target_id, []).append(str(alias["antecedent_name"]))
    connection.executemany(
        "INSERT INTO tags_fts(tag_id, name, prompt_tag, aliases) VALUES (?, ?, ?, ?)",
        [
            (
                int(row["tag_id"]),
                str(row["name"]).replace("_", " "),
                str(row["prompt_tag"]),
                " ".join(alias_map.get(int(row["tag_id"]), [])),
            )
            for row in tags
        ],
    )
    metadata = {
        "schema_version": DANBOORU_DB_SCHEMA_VERSION,
        "snapshot_version": snapshot_version,
        "source": source,
        "source_id_kind": source_id_kind,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "api_elapsed_ms": "" if api_elapsed_ms is None else f"{api_elapsed_ms:.1f}",
        "tag_count": str(len(tags)),
        "general_active_count": str(
            sum(1 for row in tags if str(row["category"]) == "general" and not row["is_deprecated"])
        ),
        "alias_count": str(len(aliases)),
        "implication_count": str(len(implications)),
    }
    connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items())
    connection.commit()


def write_snapshot(
    output_path: Path | str,
    tag_documents: Iterable[Mapping[str, object]],
    alias_documents: Iterable[Mapping[str, object]] = (),
    implication_documents: Iterable[Mapping[str, object]] = (),
    *,
    snapshot_version: str,
    source: str,
    api_elapsed_ms: float | None = None,
    source_id_kind: str = "snapshot_surrogate_id",
) -> dict[str, object]:
    tags, aliases, implications = rows_from_documents(
        tag_documents,
        alias_documents,
        implication_documents,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{target.stem}.", suffix=".sqlite3", dir=target.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            _insert_snapshot(
                connection,
                tags,
                aliases,
                implications,
                snapshot_version=snapshot_version,
                source=source,
                api_elapsed_ms=api_elapsed_ms,
                source_id_kind=source_id_kind,
            )
        finally:
            connection.close()
        os.replace(temporary_path, target)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "path": str(target),
        "tag_count": len(tags),
        "general_active_count": sum(
            1 for row in tags if str(row["category"]) == "general" and not row["is_deprecated"]
        ),
        "alias_count": len(aliases),
        "implication_count": len(implications),
        "size_bytes": target.stat().st_size,
        "api_elapsed_ms": api_elapsed_ms,
    }


def import_snapshot_file(
    input_path: Path | str,
    output_path: Path | str = DEFAULT_DANBOORU_DB_PATH,
    *,
    snapshot_version: str = "snapshot",
    source: str | None = None,
    api_elapsed_ms: float | None = None,
) -> dict[str, object]:
    tags, aliases, implications = parse_snapshot_file(input_path)
    return write_snapshot(
        output_path,
        tags,
        aliases,
        implications,
        snapshot_version=snapshot_version,
        source=source or str(input_path),
        api_elapsed_ms=api_elapsed_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_DANBOORU_DB_PATH)
    parser.add_argument("--snapshot-version", default="manual")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()
    if args.input:
        summary = import_snapshot_file(args.input, args.output, snapshot_version=args.snapshot_version)
    elif args.api:
        tags, aliases, implications, elapsed_ms = fetch_official_snapshot(
            base_url=args.base_url,
            max_pages=args.max_pages,
        )
        summary = write_snapshot(
            args.output,
            tags,
            aliases,
            implications,
            snapshot_version=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            source=args.base_url,
            api_elapsed_ms=elapsed_ms,
            source_id_kind="official_api_id",
        )
    else:
        parser.error("--inputまたは--apiが必要です")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
