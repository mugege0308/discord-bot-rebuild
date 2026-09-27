"""Persistent, user-confirmed character aliases.

The hand-maintained character dictionary is intentionally not modified by
this module.  This registry only stores aliases after an explicit user
confirmation and resolves them before the exploratory Danbooru search.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from services.danbooru_tag_store import (
    DEFAULT_DANBOORU_DB_PATH,
    DanbooruTagCandidate,
    DanbooruTagStore,
    normalize_tag_name,
)


REGISTRY_DB_PATH = DEFAULT_DANBOORU_DB_PATH.parent / "character_registry.sqlite3"
REGISTRY_SCHEMA_VERSION = "1"


class _RegistryConnection(sqlite3.Connection):
    """Close SQLite handles used by ``with`` blocks on Windows."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def normalize_character_alias(value: str) -> str:
    """Normalize safe orthographic variation, not semantic nicknames."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalized.replace("・", " ").replace("･", " ").replace("·", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _is_short_alias(value: str) -> bool:
    compact = re.sub(r"\s+", "", normalize_character_alias(value))
    return 0 < len(compact) <= 3


@dataclass(frozen=True)
class VerifiedCharacter:
    character_id: int
    canonical_danbooru_tag: str
    prompt_tag: str
    copyright_tag: str
    display_name: str
    verified: bool
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifiedCharacterMatch:
    character: VerifiedCharacter
    matched_alias: str
    normalized_alias: str
    alias_type: str
    requires_context: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VerifiedCharacterRegistry:
    def __init__(self, path: Path | str = REGISTRY_DB_PATH) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5, factory=_RegistryConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS characters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_danbooru_tag TEXT NOT NULL UNIQUE,
                prompt_tag TEXT NOT NULL,
                copyright_tag TEXT NOT NULL DEFAULT '',
                display_name TEXT NOT NULL DEFAULT '',
                verified INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS character_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                character_id INTEGER NOT NULL,
                alias TEXT NOT NULL,
                normalized_alias TEXT NOT NULL,
                alias_type TEXT NOT NULL DEFAULT 'input',
                requires_context INTEGER NOT NULL DEFAULT 0,
                verified INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                UNIQUE(character_id, normalized_alias),
                FOREIGN KEY(character_id) REFERENCES characters(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_character_aliases_normalized
                ON character_aliases(normalized_alias, verified);
            CREATE INDEX IF NOT EXISTS idx_character_aliases_character
                ON character_aliases(character_id);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO metadata(key, value)
                VALUES ('schema_version', '1');
            """
        )

    @staticmethod
    def _row_to_character(connection: sqlite3.Connection, row: sqlite3.Row) -> VerifiedCharacter:
        aliases = tuple(
            str(alias_row[0])
            for alias_row in connection.execute(
                """
                SELECT alias FROM character_aliases
                WHERE character_id = ? AND verified = 1
                ORDER BY id
                """,
                (int(row["id"]),),
            ).fetchall()
        )
        return VerifiedCharacter(
            character_id=int(row["id"]),
            canonical_danbooru_tag=str(row["canonical_danbooru_tag"]),
            prompt_tag=str(row["prompt_tag"]),
            copyright_tag=str(row["copyright_tag"] or ""),
            display_name=str(row["display_name"] or ""),
            verified=bool(row["verified"]),
            aliases=aliases,
        )

    def lookup(
        self,
        text: str,
        *,
        context_text: str = "",
    ) -> tuple[VerifiedCharacterMatch, ...]:
        """Return verified characters whose registered alias occurs in text.

        Short aliases are only eligible when a registered context alias or the
        canonical copyright tag is also present.  The method returns all
        distinct matches so the caller can abstain on ambiguity.
        """

        normalized_text = normalize_character_alias(text)
        normalized_context = normalize_character_alias(context_text or text)
        if not normalized_text:
            return ()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, a.alias, a.normalized_alias, a.alias_type,
                       a.requires_context
                FROM character_aliases AS a
                JOIN characters AS c ON c.id = a.character_id
                WHERE a.verified = 1 AND c.verified = 1
                ORDER BY a.requires_context ASC, length(a.normalized_alias) DESC, a.id
                """
            ).fetchall()
            matches: dict[int, VerifiedCharacterMatch] = {}
            for row in rows:
                if str(row["alias_type"]) in {"context", "copyright"}:
                    continue
                alias_key = str(row["normalized_alias"] or "")
                if not alias_key or alias_key not in normalized_text:
                    continue
                requires_context = bool(row["requires_context"])
                if requires_context and not self._has_context(
                    connection,
                    row,
                    normalized_context,
                ):
                    continue
                character = self._row_to_character(connection, row)
                matches.setdefault(
                    character.character_id,
                    VerifiedCharacterMatch(
                        character=character,
                        matched_alias=str(row["alias"]),
                        normalized_alias=alias_key,
                        alias_type=str(row["alias_type"]),
                        requires_context=requires_context,
                    ),
                )
            return tuple(matches.values())

    @staticmethod
    def _has_context(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        normalized_context: str,
    ) -> bool:
        character_id = int(row["id"])
        copyright_tag = normalize_character_alias(row["copyright_tag"] or "")
        if copyright_tag and copyright_tag in normalized_context:
            return True
        context_aliases = connection.execute(
            """
            SELECT normalized_alias FROM character_aliases
            WHERE character_id = ? AND verified = 1
              AND alias_type IN ('context', 'copyright')
            """,
            (character_id,),
        ).fetchall()
        return any(
            str(context_row[0]) and str(context_row[0]) in normalized_context
            for context_row in context_aliases
        )

    def register_verified_candidate(
        self,
        candidate: DanbooruTagCandidate,
        *,
        input_alias: str,
        copyright_tag: str = "",
        context_aliases: Sequence[str] = (),
        display_name: str | None = None,
        store: DanbooruTagStore | None = None,
    ) -> VerifiedCharacter:
        """Persist one DB-validated candidate after explicit confirmation."""

        search_store = store or DanbooruTagStore()
        rows = search_store.validate_candidate_ids(
            (candidate.tag_id,),
            candidate_ids=(candidate.tag_id,),
            allowed_categories=("character",),
        )
        if len(rows) != 1 or normalize_tag_name(rows[0].name) != normalize_tag_name(candidate.name):
            raise ValueError("確認候補がactive character tagとして検証できません。")
        canonical_tag = rows[0].name
        prompt_tag = rows[0].prompt_tag
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO characters(
                    canonical_danbooru_tag, prompt_tag, copyright_tag,
                    display_name, verified, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(canonical_danbooru_tag) DO UPDATE SET
                    prompt_tag = excluded.prompt_tag,
                    copyright_tag = CASE
                        WHEN excluded.copyright_tag <> '' THEN excluded.copyright_tag
                        ELSE characters.copyright_tag END,
                    display_name = CASE
                        WHEN excluded.display_name <> '' THEN excluded.display_name
                        ELSE characters.display_name END,
                    verified = 1,
                    updated_at = excluded.updated_at
                """,
                (
                    canonical_tag,
                    prompt_tag,
                    str(copyright_tag or "").strip(),
                    str(display_name or input_alias or canonical_tag).strip(),
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM characters WHERE canonical_danbooru_tag = ?",
                (canonical_tag,),
            ).fetchone()
            if row is None:
                raise RuntimeError("character registry row was not created")
            character_id = int(row["id"])
            aliases: list[tuple[str, str, bool]] = [
                (canonical_tag, "canonical", False),
                (prompt_tag, "prompt", False),
                (str(display_name or input_alias or canonical_tag).strip(), "display", False),
                (str(input_alias or "").strip(), "input", _is_short_alias(input_alias)),
            ]
            for context_alias in context_aliases:
                aliases.append((str(context_alias or "").strip(), "context", False))
            for alias, alias_type, requires_context in aliases:
                normalized_alias = normalize_character_alias(alias)
                if not normalized_alias:
                    continue
                connection.execute(
                    """
                    INSERT INTO character_aliases(
                        character_id, alias, normalized_alias, alias_type,
                        requires_context, verified, created_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(character_id, normalized_alias) DO UPDATE SET
                        alias = excluded.alias,
                        alias_type = excluded.alias_type,
                        requires_context = excluded.requires_context,
                        verified = 1
                    """,
                    (
                        character_id,
                        alias,
                        normalized_alias,
                        alias_type,
                        int(requires_context),
                        now,
                    ),
                )
            connection.commit()
            saved = connection.execute(
                "SELECT * FROM characters WHERE id = ?",
                (character_id,),
            ).fetchone()
            if saved is None:
                raise RuntimeError("saved character registry row disappeared")
            return self._row_to_character(connection, saved)


def registry_match_to_dict(match: VerifiedCharacterMatch) -> dict[str, object]:
    return {
        "character_id": match.character.character_id,
        "canonical_danbooru_tag": match.character.canonical_danbooru_tag,
        "prompt_tag": match.character.prompt_tag,
        "copyright_tag": match.character.copyright_tag,
        "display_name": match.character.display_name,
        "matched_alias": match.matched_alias,
        "normalized_alias": match.normalized_alias,
        "alias_type": match.alias_type,
        "requires_context": match.requires_context,
    }
