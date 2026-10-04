"""Read-only SQLite storage and search for the generic Danbooru tag MVP.

The small prompt-tag dictionary remains the authoritative source for the
composition categories.  This module intentionally knows nothing about the
LLM or the prompt designer; it only returns real, active Danbooru rows.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DANBOORU_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "danbooru"
DEFAULT_DANBOORU_DB_PATH = DANBOORU_DATA_DIR / "tags.sqlite3"
DANBOORU_DB_SCHEMA_VERSION = "1"
GENERAL_CATEGORY = "general"


_CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "meta",
}
_SEARCH_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*", re.IGNORECASE)


class _ReadOnlyConnection(sqlite3.Connection):
    """Close read-only handles when used with ``with`` on Windows."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def normalize_tag_name(value: str) -> str:
    """Normalize a Danbooru name or an English prompt query."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def normalize_search_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def category_name(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized.isdigit():
            return _CATEGORY_NAMES.get(int(normalized), normalized)
        return normalized
    try:
        return _CATEGORY_NAMES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value or "").strip().casefold()


@dataclass(frozen=True)
class DanbooruTagCandidate:
    tag_id: int
    name: str
    prompt_tag: str
    category: str
    post_count: int
    is_deprecated: bool
    match_type: str
    query: str
    matched_alias: str = ""
    retrieval_score: float = 0.0

    def to_trace_dict(self) -> dict[str, object]:
        return {
            "tag_id": self.tag_id,
            "name": self.name,
            "post_count": self.post_count,
            "match_type": self.match_type,
            "query": self.query,
            "matched_alias": self.matched_alias,
            "retrieval_score": round(self.retrieval_score, 4),
        }


@dataclass(frozen=True)
class DanbooruTagRow:
    tag_id: int
    name: str
    category: str
    post_count: int
    is_deprecated: bool
    prompt_tag: str


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS tags (
            tag_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            normalized_name TEXT NOT NULL UNIQUE,
            prompt_tag TEXT NOT NULL,
            category TEXT NOT NULL,
            post_count INTEGER NOT NULL DEFAULT 0,
            is_deprecated INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_tags_searchable
            ON tags(category, is_deprecated, normalized_name);
        CREATE INDEX IF NOT EXISTS idx_tags_popularity
            ON tags(category, is_deprecated, post_count DESC);

        CREATE TABLE IF NOT EXISTS tag_aliases (
            antecedent_name TEXT PRIMARY KEY,
            normalized_antecedent TEXT NOT NULL UNIQUE,
            consequent_name TEXT NOT NULL,
            normalized_consequent TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE INDEX IF NOT EXISTS idx_alias_target
            ON tag_aliases(normalized_consequent);

        CREATE TABLE IF NOT EXISTS tag_implications (
            antecedent_name TEXT NOT NULL,
            consequent_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            PRIMARY KEY(antecedent_name, consequent_name)
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS tags_fts USING fts5(
            tag_id UNINDEXED,
            name,
            prompt_tag,
            aliases,
            tokenize = 'unicode61 remove_diacritics 1',
            prefix = '2 3 4'
        );
        """
    )


def configure_connection(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 2000")


class DanbooruTagStore:
    """Search an imported tag snapshot without network access."""

    def __init__(self, path: Path | str = DEFAULT_DANBOORU_DB_PATH) -> None:
        self.path = Path(path)

    @property
    def available(self) -> bool:
        return self.path.is_file() and self.path.stat().st_size > 0

    def _connect(self) -> sqlite3.Connection:
        if not self.available:
            raise FileNotFoundError(f"Danbooru tag DB is not available: {self.path}")
        connection = sqlite3.connect(
            f"file:{self.path.resolve()}?mode=ro",
            uri=True,
            timeout=2,
            factory=_ReadOnlyConnection,
        )
        configure_connection(connection)
        return connection

    def metadata(self, key: str, default: str = "") -> str:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = ?", (key,)
                ).fetchone()
        except (OSError, sqlite3.Error):
            return default
        return str(row[0]) if row else default

    @property
    def snapshot_version(self) -> str:
        return self.metadata("snapshot_version", "unknown")

    @property
    def tag_count(self) -> int:
        try:
            with self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0])
        except (OSError, sqlite3.Error):
            return 0

    def _candidate_from_row(
        self,
        row: sqlite3.Row,
        *,
        query: str,
        match_type: str,
        matched_alias: str = "",
        retrieval_score: float = 0.0,
    ) -> DanbooruTagCandidate:
        return DanbooruTagCandidate(
            tag_id=int(row["tag_id"]),
            name=str(row["name"]),
            prompt_tag=str(row["prompt_tag"]),
            category=str(row["category"]),
            post_count=int(row["post_count"]),
            is_deprecated=bool(row["is_deprecated"]),
            match_type=match_type,
            query=query,
            matched_alias=matched_alias,
            retrieval_score=float(retrieval_score),
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        allowed_categories: Sequence[str] | None = (GENERAL_CATEGORY,),
        include_deprecated: bool = False,
    ) -> tuple[DanbooruTagCandidate, ...]:
        """Return active candidates using exact, alias, prefix, then FTS.

        ``allowed_categories=None`` means all categories from the imported
        snapshot. The default remains ``general`` for existing resolver
        callers.
        """

        clean_query = normalize_search_text(query)
        normalized_query = normalize_tag_name(query)
        if not clean_query or limit <= 0:
            return ()
        categories = tuple(category_name(value) for value in allowed_categories) if allowed_categories is not None else ()
        category_clause = "1=1" if allowed_categories is None else f"category IN ({','.join('?' for _ in categories)})"
        category_params = [] if allowed_categories is None else [*categories]
        active_clause = "1=1" if include_deprecated else "is_deprecated = 0"
        results: dict[int, DanbooruTagCandidate] = {}

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT tag_id, name, prompt_tag, category, post_count, is_deprecated
                FROM tags
                WHERE {category_clause} AND {active_clause}
                  AND (normalized_name = ? OR normalized_name = ?)
                ORDER BY post_count DESC, tag_id
                LIMIT ?
                """,
                [*category_params, normalized_query, clean_query.replace(" ", "_"), limit],
            ).fetchall()
            for row in rows:
                results[int(row["tag_id"])] = self._candidate_from_row(
                    row, query=query, match_type="exact", retrieval_score=100.0
                )

            alias_rows = connection.execute(
                f"""
                SELECT t.tag_id, t.name, t.prompt_tag, t.category, t.post_count,
                       t.is_deprecated, a.antecedent_name
                FROM tag_aliases AS a
                JOIN tags AS t ON t.normalized_name = a.normalized_consequent
                WHERE a.normalized_antecedent = ?
                  AND a.status = 'active'
                  AND {category_clause.replace('category', 't.category')} AND {active_clause}
                ORDER BY t.post_count DESC, t.tag_id
                LIMIT ?
                """,
                [normalized_query, *category_params, limit],
            ).fetchall()
            for row in alias_rows:
                tag_id = int(row["tag_id"])
                results.setdefault(
                    tag_id,
                    self._candidate_from_row(
                        row,
                        query=query,
                        match_type="alias_exact",
                        matched_alias=str(row["antecedent_name"]),
                        retrieval_score=95.0,
                    ),
                )

            prefix_rows = connection.execute(
                f"""
                SELECT t.tag_id, t.name, t.prompt_tag, t.category, t.post_count, t.is_deprecated,
                       COALESCE(a.antecedent_name, '') AS matched_alias
                FROM tags AS t
                LEFT JOIN tag_aliases AS a ON a.normalized_consequent = t.normalized_name
                    AND a.status = 'active'
                WHERE {category_clause.replace('category', 't.category')} AND {active_clause}
                  AND (t.normalized_name LIKE ? OR t.prompt_tag LIKE ?
                       OR a.normalized_antecedent LIKE ?)
                ORDER BY t.post_count DESC, t.tag_id
                LIMIT ?
                """,
                [
                    *category_params,
                    f"{normalized_query}%",
                    f"{clean_query}%",
                    f"{normalized_query}%",
                    limit,
                ],
            ).fetchall()
            for row in prefix_rows:
                tag_id = int(row["tag_id"])
                results.setdefault(
                    tag_id,
                    self._candidate_from_row(
                        row,
                        query=query,
                        match_type="prefix",
                        matched_alias=str(row["matched_alias"] or ""),
                        retrieval_score=80.0,
                    ),
                )

            fts_query = self._fts_query(clean_query)
            if fts_query:
                fts_rows = connection.execute(
                    f"""
                    SELECT t.tag_id, t.name, t.prompt_tag, t.category, t.post_count, t.is_deprecated,
                           bm25(tags_fts) AS rank,
                           tags_fts.aliases AS aliases
                    FROM tags_fts
                    JOIN tags AS t ON t.tag_id = CAST(tags_fts.tag_id AS INTEGER)
                    WHERE tags_fts MATCH ?
                      AND {category_clause.replace('category', 't.category')} AND {active_clause}
                    ORDER BY rank ASC, t.post_count DESC, t.tag_id
                    LIMIT ?
                    """,
                    [fts_query, *category_params, limit],
                ).fetchall()
                for row in fts_rows:
                    tag_id = int(row["tag_id"])
                    rank = float(row["rank"])
                    score = 60.0 + min(20.0, max(0.0, -rank))
                    results.setdefault(
                        tag_id,
                        self._candidate_from_row(
                            row,
                            query=query,
                            match_type="fts",
                            retrieval_score=score,
                        ),
                    )

        return tuple(
            sorted(
                results.values(),
                key=lambda item: (-item.retrieval_score, -math.log1p(max(item.post_count, 0)), item.name),
            )[:limit]
        )

    def find_canonical(
        self,
        name: str,
        *,
        allowed_categories: Sequence[str] = (),
    ) -> DanbooruTagCandidate | None:
        """Return one active canonical row for a validated external dictionary."""

        normalized = normalize_tag_name(name)
        if not normalized:
            return None
        categories = tuple(category_name(value) for value in allowed_categories)
        category_clause = "1=1" if not categories else f"category IN ({','.join('?' for _ in categories)})"
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT tag_id, name, prompt_tag, category, post_count, is_deprecated
                FROM tags
                WHERE normalized_name = ? AND is_deprecated = 0 AND {category_clause}
                LIMIT 1
                """,
                [normalized, *categories],
            ).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(
            row,
            query=name,
            match_type="canonical",
            retrieval_score=100.0,
        )

    @staticmethod
    def _fts_query(clean_query: str) -> str:
        tokens = _SEARCH_TOKEN_RE.findall(clean_query)
        if not tokens:
            return ""
        # Quoting each token prevents punctuation in an LLM query from turning
        # into FTS operators. Prefixing the final token preserves recall for
        # singular/plural and common tag suffixes.
        escaped = [token.replace('"', '""') for token in tokens]
        if len(escaped) == 1:
            return f'"{escaped[0]}"*'
        return " AND ".join(f'"{token}"' for token in escaped)

    def get_rows(self, tag_ids: Iterable[int]) -> dict[int, DanbooruTagRow]:
        ids = tuple(dict.fromkeys(int(tag_id) for tag_id in tag_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT tag_id, name, category, post_count, is_deprecated, prompt_tag
                FROM tags WHERE tag_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
        return {
            int(row["tag_id"]): DanbooruTagRow(
                tag_id=int(row["tag_id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                post_count=int(row["post_count"]),
                is_deprecated=bool(row["is_deprecated"]),
                prompt_tag=str(row["prompt_tag"]),
            )
            for row in rows
        }

    def validate_candidate_ids(
        self,
        tag_ids: Iterable[int],
        *,
        candidate_ids: Iterable[int],
        allowed_categories: Sequence[str] = (GENERAL_CATEGORY,),
        locked_names: Iterable[str] = (),
    ) -> tuple[DanbooruTagRow, ...]:
        allowed = {int(value) for value in candidate_ids}
        categories = {category_name(value) for value in allowed_categories}
        locked = {normalize_tag_name(value) for value in locked_names}
        rows = self.get_rows(int(value) for value in tag_ids if int(value) in allowed)
        selected: list[DanbooruTagRow] = []
        for tag_id in tag_ids:
            row = rows.get(int(tag_id))
            if row is None or row.category not in categories or row.is_deprecated:
                continue
            if normalize_tag_name(row.name) in locked:
                continue
            selected.append(row)
        return tuple(dict((row.tag_id, row) for row in selected).values())


def rows_from_documents(
    tag_documents: Iterable[Mapping[str, object]],
    alias_documents: Iterable[Mapping[str, object]] = (),
    implication_documents: Iterable[Mapping[str, object]] = (),
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[dict[str, str]]]:
    """Normalize API/snapshot documents for the importer.

    Public snapshots sometimes omit Danbooru's numeric id.  In that case the
    importer assigns a deterministic positive surrogate; official API rows
    retain their real id.  The metadata table records this provenance.
    """

    tags: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(tag_documents, start=1):
        name = str(raw.get("name") or raw.get("tag") or "").strip()
        if not name:
            continue
        normalized = normalize_tag_name(name)
        if normalized in seen_names:
            continue
        seen_names.add(normalized)
        raw_id = raw.get("id") or raw.get("tag_id")
        try:
            tag_id = int(raw_id) if raw_id is not None else index
        except (TypeError, ValueError):
            tag_id = index
        try:
            post_count = int(raw.get("post_count") or raw.get("count") or 0)
        except (TypeError, ValueError):
            post_count = 0
        raw_deprecated = raw.get("is_deprecated", raw.get("deprecated", False))
        is_deprecated = bool(raw_deprecated) and str(raw_deprecated).casefold() not in {"0", "false", "no"}
        tags.append(
            {
                "tag_id": tag_id,
                "name": name,
                "normalized_name": normalized,
                "prompt_tag": name.replace("_", " "),
                "category": category_name(raw.get("category", "general")),
                "post_count": post_count,
                "is_deprecated": is_deprecated,
            }
        )

    aliases: list[dict[str, str]] = []
    for raw in alias_documents:
        antecedent = str(
            raw.get("antecedent_name") or raw.get("antecedent") or raw.get("from") or ""
        ).strip()
        consequent = str(
            raw.get("consequent_name") or raw.get("consequent") or raw.get("to") or ""
        ).strip()
        if antecedent and consequent:
            aliases.append(
                {
                    "antecedent_name": antecedent,
                    "normalized_antecedent": normalize_tag_name(antecedent),
                    "consequent_name": consequent,
                    "normalized_consequent": normalize_tag_name(consequent),
                    "status": str(raw.get("status") or "active"),
                }
            )

    implications: list[dict[str, str]] = []
    for raw in implication_documents:
        antecedent = str(raw.get("antecedent_name") or raw.get("antecedent") or "").strip()
        consequent = str(raw.get("consequent_name") or raw.get("consequent") or "").strip()
        if antecedent and consequent:
            implications.append(
                {
                    "antecedent_name": antecedent,
                    "consequent_name": consequent,
                    "status": str(raw.get("status") or "active"),
                }
            )
    return tags, aliases, implications
