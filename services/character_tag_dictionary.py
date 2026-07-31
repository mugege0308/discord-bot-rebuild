"""Character tag dictionary.

Resolves a Japanese/English character name (and, where present, a work/series
name) found in freeform user text to the canonical Danbooru-style tags an
image model recognizes, so Prompt Designer can be told "this is character X
from series Y" instead of guessing from a bare Japanese name.

## How to add a character

Add one dict to `_RAW_CHARACTERS` below:

- `character_tag` (str, required): canonical Danbooru-style tag, e.g.
  `"komeiji koishi"`.
- `display_name` (str, required): canonical Japanese display name, e.g.
  `"古明地こいし"`. Passed to Prompt Designer as identification context only
  - it is never itself added as a positive/negative tag.
- `series_tag` (str, required): must match an existing `series_tag` in
  `_RAW_SERIES` below (add the series first if it doesn't exist yet).
- `safe_aliases` (tuple[str, ...], required, may be empty): aliases
  long/specific enough to resolve the character on their own - the full
  Japanese name, the full English/romaji name, or a name that isn't also an
  ordinary word or common given name.
- `ambiguous_aliases` (tuple[str, ...], optional): short aliases that double
  as ordinary words or common Japanese given names (e.g. a bare given name,
  a common noun). These only resolve when this character's own series alias
  is ALSO present in the same text - see the "花火" (Sparkle)/"きなこ"
  (Kinako Nanobana)/"こいし" (Koishi) entries below for the working pattern.
  A character can have both: a full name in `safe_aliases` that resolves on
  its own, and a bare given name in `ambiguous_aliases` that only resolves
  alongside its series - this is why there is no single `requires_series_match`
  flag on the whole character.
- `notes` (str, optional): free text. Use this to flag a tag/reading that
  could not be confirmed from anything in this repository, so a future
  contributor knows to verify it before trusting it.

A malformed entry (missing key, wrong type, unknown series_tag) is skipped
with a logged warning at import time - it never prevents the rest of the
dictionary, or the bot, from loading.

Appearance (hair/eye color, outfit, build, ...) is intentionally NOT part of
this dictionary. Character *identity* resolution and appearance
reinforcement are two separate concerns; only character_tag/series_tag are
resolved here, and services/shogou_prompt_designer.py never fabricates
appearance details from this module's output.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, fields as dataclass_fields


logger = logging.getLogger("secretary-bot.character_tag_dictionary")


@dataclass(frozen=True)
class CharacterTagResolution:
    character_tag: str
    display_name: str
    series_tag: str
    series_name: str
    matched_character_alias: str
    matched_series_alias: str | None
    match_type: str  # "character_and_series" or "character_only"


@dataclass(frozen=True)
class _SeriesEntry:
    series_tag: str
    series_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class _CharacterEntry:
    character_tag: str
    display_name: str
    series_tag: str
    safe_aliases: tuple[str, ...]
    ambiguous_aliases: tuple[str, ...] = ()
    notes: str = ""


_RAW_SERIES: tuple[dict, ...] = (
    {
        "series_tag": "touhou",
        "series_name": "東方Project",
        "aliases": ("東方Project", "東方", "touhou project", "touhou"),
    },
    {
        "series_tag": "honkai: star rail",
        "series_name": "崩壊：スターレイル",
        "aliases": (
            "崩壊：スターレイル",
            "崩壊:スターレイル",
            "崩壊スターレイル",
            "スターレイル",
            "スタレ",
            "honkai: star rail",
            "honkai star rail",
        ),
    },
    {
        "series_tag": "inazuma eleven go",
        "series_name": "イナズマイレブンGO",
        "aliases": ("イナズマイレブンGO", "イナズマイレブン go", "イナイレGO", "inazuma eleven go"),
    },
    {
        "series_tag": "danganronpa",
        "series_name": "ダンガンロンパ",
        "aliases": ("ダンガンロンパ", "danganronpa"),
    },
    {
        "series_tag": "blue archive",
        "series_name": "ブルーアーカイブ",
        "aliases": ("ブルーアーカイブ", "ブルアカ", "blue archive"),
    },
    {
        "series_tag": "sword art online",
        "series_name": "ソードアート・オンライン",
        "aliases": ("ソードアート・オンライン", "ソードアートオンライン", "sword art online", "sao"),
    },
    {
        "series_tag": "genshin impact",
        "series_name": "原神",
        "aliases": ("原神", "genshin impact", "genshin"),
    },
    {
        "series_tag": "pokemon",
        "series_name": "ポケットモンスター",
        "aliases": ("ポケットモンスター", "ポケモン", "pokemon", "pokémon"),
    },
    {
        "series_tag": "idolmaster shiny colors",
        "series_name": "アイドルマスター シャイニーカラーズ",
        "aliases": (
            "アイドルマスターシャイニーカラーズ",
            "アイドルマスター シャイニーカラーズ",
            "シャイニーカラーズ",
            "シャニマス",
            "idolmaster shiny colors",
            "shiny colors",
        ),
    },
    {
        "series_tag": "hamidashi creative",
        "series_name": "ハミダシクリエイティブ",
        "aliases": ("ハミダシクリエイティブ", "hamidashi creative"),
    },
    {
        "series_tag": "mano saba",
        "series_name": "まのさば",
        "aliases": ("まのさば", "mano saba"),
    },
    {
        "series_tag": "ao no hako",
        "series_name": "アオのハコ",
        "aliases": ("アオのハコ", "青のハコ", "ao no hako", "blue box"),
    },
)


# Tag-accuracy note (see final report / notes field below): character_tag for
# every entry flagged `notes=...` below could not be confirmed against
# anything already in this repository (no prior few-shot, no prior test, no
# prior usage). Those use a conservative surname-given romanization instead
# of a guessed/confirmed Danbooru tag, and add no appearance tags.
_RAW_CHARACTERS: tuple[dict, ...] = (
    {
        "character_tag": "komeiji koishi",
        "display_name": "古明地こいし",
        "series_tag": "touhou",
        "safe_aliases": ("古明地こいし", "komeiji koishi", "koishi komeiji"),
        "ambiguous_aliases": ("こいし", "koishi"),
    },
    {
        "character_tag": "sparkle",
        "display_name": "花火",
        "series_tag": "honkai: star rail",
        # "sparkle" is specific/unique enough as an English name to resolve on
        # its own (unlike the generic Japanese words below, which double as
        # ordinary vocabulary and stay gated behind a series match).
        "safe_aliases": ("sparkle",),
        # "花火" ("Hanabi") is the character's confirmed official Japanese
        # localization name - Danbooru's alias for this character is
        # "hanabi_(honkai:_star_rail)", pointing at the canonical
        # "sparkle_(honkai:_star_rail)" tag. "火花" ("spark") was this
        # dictionary's original guess before that was confirmed; it reads
        # plausibly (matches "Sparkle" semantically) but is NOT the
        # character's real name. Kept only for backward compatibility with
        # anyone who already typed it - see `notes`. Both stay series-gated
        # since either could plausibly appear in unrelated text (e.g. an
        # actual fireworks scene request).
        "ambiguous_aliases": ("花火", "火花", "スパークル"),
        "notes": (
            "「花火」(Hanabi)が確認済みの正式な日本語名（Danbooruのalias "
            "hanabi_(honkai:_star_rail) で確認）。「火花」はこの辞書が確認前に採用していた"
            "誤表記で、後方互換のためだけに残している。新規に追加する場合は「花火」を使うこと。"
        ),
    },
    {
        "character_tag": "nanobana kinako",
        "display_name": "菜花黄名子",
        "series_tag": "inazuma eleven go",
        "safe_aliases": ("菜花きなこ", "菜花黄名子", "菜花 黄名子", "nanobana kinako", "kinako nanobana"),
        # "黄名子"/"きなこ" alone: 黄名子 alone reads ambiguously and きなこ is
        # also the ordinary word for roasted soybean flour, so both stay
        # gated behind the series alias.
        "ambiguous_aliases": ("黄名子", "きなこ"),
        "notes": (
            "character_tagはDanbooru上の実タグ nanobana_kinako (姓 nanobana + 名 kinako) に合わせて "
            "'nanobana kinako' へ訂正済み（旧 'kinako nanobana' は姓名順が逆だった）。"
            "'kinako nanobana' は後方互換のため safe_aliases に残している。"
        ),
    },
    {
        "character_tag": "nanami chiaki",
        "display_name": "七海千秋",
        "series_tag": "danganronpa",
        "safe_aliases": ("七海千秋", "nanami chiaki", "chiaki nanami"),
        "notes": "character_tagはリポジトリ内で未確認。保守的な姓名順ローマ字表記を仮採用。要確認。",
    },
    {
        "character_tag": "shimoe koharu",
        "display_name": "下江コハル",
        "series_tag": "blue archive",
        "safe_aliases": ("下江コハル", "shimoe koharu", "koharu shimoe"),
        # "コハル"/"koharu" alone is an ordinary Japanese given name.
        "ambiguous_aliases": ("コハル", "koharu"),
    },
    {
        "character_tag": "kisaki",
        "display_name": "竜華キサキ",
        "series_tag": "blue archive",
        "safe_aliases": ("竜華キサキ", "ryuuge kisaki"),
        # "キサキ"/"kisaki" alone is also the ordinary Japanese word for
        # "empress consort", so it stays gated behind the series alias
        # (e.g. "ブルアカのキサキ" / "blue archive kisaki" resolve; bare
        # "キサキ"/"kisaki" does not).
        "ambiguous_aliases": ("キサキ", "kisaki"),
        "notes": (
            "正式なローマ字名は Ryuuge Kisaki（旧実装の 'Ryuka' は誤り）。"
            "Danbooru上の実タグは kisaki_(blue_archive)、別名 ryuuge_kisaki。"
            "本システムはseries_tagを別出力するため、character_tagは括弧なしの 'kisaki' を使用。"
        ),
    },
    {
        "character_tag": "kei",
        "display_name": "ケイ",
        "series_tag": "blue archive",
        # No safe alias: this character is only ever known by a single short
        # given name, which is also an extremely common Japanese/English
        # name/word, so it must always require the series alias.
        "safe_aliases": (),
        "ambiguous_aliases": ("ケイ", "kei"),
        "notes": (
            "character_tagはリポジトリ内で未確認。'kei'は一般的すぎる短い名前のため、"
            "Danbooru上は曖昧回避の表記（例: 'kei (blue archive)'）が必要な可能性がある。要確認。"
        ),
    },
    {
        "character_tag": "leafa",
        "display_name": "リーファ",
        "series_tag": "sword art online",
        # Short handle-like name; kept series-gated even though "leafa" isn't
        # an ordinary dictionary word, per this batch's cautious default for
        # any single short name without an explicit "resolves standalone"
        # requirement.
        "safe_aliases": (),
        "ambiguous_aliases": ("リーファ", "leafa"),
    },
    {
        "character_tag": "hu tao",
        "display_name": "胡桃",
        "series_tag": "genshin impact",
        # The English/romaji name is specific enough to resolve on its own;
        # the bare kanji "胡桃" is also the ordinary Japanese word for
        # "walnut" and stays series-gated.
        "safe_aliases": ("hu tao", "hutao"),
        "ambiguous_aliases": ("胡桃",),
    },
    {
        "character_tag": "suzuna",
        "display_name": "スズナ",
        "series_tag": "pokemon",
        "safe_aliases": (),
        "ambiguous_aliases": ("スズナ", "suzuna"),
        "notes": (
            "どのポケットモンスター作品のどのキャラクターかリポジトリ内で確認できず、"
            "character_tagは保守的なローマ字表記のみ。'スズナ'は一般的な語・人名とも紛らわしいため常に作品名同時指定を要求。要確認。"
        ),
    },
    {
        "character_tag": "suzuki hana",
        "display_name": "鈴木羽那",
        "series_tag": "idolmaster shiny colors",
        "safe_aliases": ("鈴木羽那", "suzuki hana", "hana suzuki"),
        "notes": "character_tagはリポジトリ内で未確認。保守的な姓名順ローマ字表記を仮採用。要確認。",
    },
    {
        "character_tag": "izumi kie",
        "display_name": "和泉妃愛",
        "series_tag": "hamidashi creative",
        "safe_aliases": ("和泉妃愛", "izumi kie"),
        "notes": (
            "「妃愛」の読み方がリポジトリ内で確認できず、保守的な読み(kie)を仮採用。"
            "character_tagともに要確認。外見タグは追加していない。"
        ),
    },
    {
        "character_tag": "meruru",
        "display_name": "メルル",
        "series_tag": "mano saba",
        "safe_aliases": (),
        "ambiguous_aliases": ("メルル", "meruru"),
        "notes": "出典作品「まのさば」の情報がリポジトリ内に乏しく、series_tag/character_tagともに保守的な直接ローマ字表記。要確認。",
    },
    {
        "character_tag": "shikano chinatsu",
        "display_name": "鹿野千夏",
        "series_tag": "ao no hako",
        "safe_aliases": ("鹿野千夏", "shikano chinatsu", "chinatsu shikano"),
        # "千夏"/"chinatsu" alone is an ordinary Japanese given name.
        "ambiguous_aliases": ("千夏", "chinatsu"),
        "notes": "character_tagはリポジトリ内で未確認。保守的な姓名順ローマ字表記を仮採用。要確認。",
    },
)


def _build_series_entries(raw_entries: tuple[dict, ...]) -> tuple[_SeriesEntry, ...]:
    built: list[_SeriesEntry] = []
    for raw in raw_entries:
        try:
            entry = _SeriesEntry(
                series_tag=str(raw["series_tag"]),
                series_name=str(raw["series_name"]),
                aliases=tuple(str(alias) for alias in raw["aliases"]),
            )
        except Exception:
            logger.exception("Skipping invalid character-dictionary series entry: %r", raw)
            continue
        built.append(entry)
    return tuple(built)


def _build_character_entries(
    raw_entries: tuple[dict, ...], valid_series_tags: frozenset[str]
) -> tuple[_CharacterEntry, ...]:
    built: list[_CharacterEntry] = []
    for raw in raw_entries:
        try:
            entry = _CharacterEntry(
                character_tag=str(raw["character_tag"]),
                display_name=str(raw["display_name"]),
                series_tag=str(raw["series_tag"]),
                safe_aliases=tuple(str(alias) for alias in raw.get("safe_aliases", ())),
                ambiguous_aliases=tuple(str(alias) for alias in raw.get("ambiguous_aliases", ())),
                notes=str(raw.get("notes", "")),
            )
        except Exception:
            logger.exception("Skipping invalid character-dictionary character entry: %r", raw)
            continue
        if entry.series_tag not in valid_series_tags:
            logger.warning(
                "Skipping character-dictionary entry with unknown series_tag: character_tag=%r series_tag=%r",
                entry.character_tag,
                entry.series_tag,
            )
            continue
        built.append(entry)
    return tuple(built)


_SERIES_ENTRIES: tuple[_SeriesEntry, ...] = _build_series_entries(_RAW_SERIES)
_SERIES_NAME_BY_TAG: dict[str, str] = {series.series_tag: series.series_name for series in _SERIES_ENTRIES}
_SERIES_BY_TAG: dict[str, _SeriesEntry] = {series.series_tag: series for series in _SERIES_ENTRIES}
_CHARACTER_ENTRIES: tuple[_CharacterEntry, ...] = _build_character_entries(
    _RAW_CHARACTERS, frozenset(_SERIES_NAME_BY_TAG)
)


_WHITESPACE_RE = re.compile(r"\s+")
_ASCII_ALNUM_RE = re.compile(r"[a-z0-9]")


def _normalize(text: str) -> str:
    # NFKC folds full-width/half-width variants (e.g. full-width colon,
    # full-width Latin letters) to a single common form. Whitespace runs are
    # collapsed to one space (not removed entirely) rather than stripped, so
    # word-boundary matching below still has real separators to work with.
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.casefold().replace("：", ":")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _alias_present(normalized_text: str, normalized_alias: str) -> bool:
    """True if `normalized_alias` occurs in `normalized_text`.

    ASCII aliases (English/romaji) require a word boundary on both sides
    (not preceded/followed by another ASCII letter or digit) so a short
    alias like "kei" cannot match inside an unrelated longer ASCII word.
    Non-ASCII (Japanese) aliases use plain substring containment instead:
    Japanese text has no inter-word spaces to anchor a boundary on, and
    substring matching against a curated, mostly multi-character alias list
    is the standard practical approach for this kind of Japanese keyword
    matching.
    """
    if not normalized_alias:
        return False
    if normalized_alias.isascii():
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(normalized_alias) + r"(?![a-z0-9])")
        return pattern.search(normalized_text) is not None
    return normalized_alias in normalized_text


def _find_alias(normalized_text: str, aliases: tuple[str, ...]) -> str | None:
    for alias in sorted(aliases, key=len, reverse=True):
        if _alias_present(normalized_text, _normalize(alias)):
            return alias
    return None


def _match_series(normalized_text: str) -> tuple[str | None, str | None]:
    for series in _SERIES_ENTRIES:
        alias = _find_alias(normalized_text, series.aliases)
        if alias is not None:
            return series.series_tag, alias
    return None, None


def known_character_and_series_tags() -> frozenset[str]:
    """Every canonical character_tag/series_tag this dictionary can resolve
    to, e.g. "komeiji koishi" or "touhou". Read-only and derived purely from
    the tables above.

    Used by selection preference (services/selection_preference.py) to keep
    character/series names out of general tag-preference statistics: a
    character being rated highly must not turn into "add komeiji koishi to
    every generation", so those exact tags are excluded before any stats are
    built from stored positive prompts.
    """
    tags: set[str] = set()
    for character in _CHARACTER_ENTRIES:
        tags.add(character.character_tag)
        tags.add(character.series_tag)
    for series in _SERIES_ENTRIES:
        tags.add(series.series_tag)
    return frozenset(tags)


def _resolve_all(user_text: str) -> list[CharacterTagResolution]:
    normalized_text = _normalize(user_text)
    if not normalized_text:
        return []

    resolutions: list[CharacterTagResolution] = []
    for character in _CHARACTER_ENTRIES:
        matched_safe_alias = _find_alias(normalized_text, character.safe_aliases)
        matched_ambiguous_alias = _find_alias(normalized_text, character.ambiguous_aliases)
        if matched_safe_alias is None and matched_ambiguous_alias is None:
            continue

        matched_series_tag, matched_series_alias = _match_series(normalized_text)

        if matched_safe_alias is not None:
            resolved_series_tag = matched_series_tag or character.series_tag
            match_type = "character_and_series" if matched_series_tag else "character_only"
            resolutions.append(
                CharacterTagResolution(
                    character_tag=character.character_tag,
                    display_name=character.display_name,
                    series_tag=resolved_series_tag,
                    series_name=_SERIES_NAME_BY_TAG.get(resolved_series_tag, resolved_series_tag),
                    matched_character_alias=matched_safe_alias,
                    matched_series_alias=matched_series_alias,
                    match_type=match_type,
                )
            )
            continue

        # Ambiguous-only match: only resolves alongside this character's own
        # series alias (the false-positive safety valve documented above).
        if matched_series_tag == character.series_tag:
            resolutions.append(
                CharacterTagResolution(
                    character_tag=character.character_tag,
                    display_name=character.display_name,
                    series_tag=character.series_tag,
                    series_name=_SERIES_NAME_BY_TAG.get(character.series_tag, character.series_tag),
                    matched_character_alias=matched_ambiguous_alias,
                    matched_series_alias=matched_series_alias,
                    match_type="character_and_series",
                )
            )

    return resolutions


def resolve_character_tags(user_text: str) -> CharacterTagResolution | None:
    """Resolves a Japanese/English character (and series) name found in
    freeform user text to the canonical Danbooru-style tags an image model
    recognizes. Only matches a short/common alias (e.g. a bare character
    given name that doubles as an ordinary word) when this character's own
    series alias is also present in the same text, to avoid false positives.

    Returns at most one resolution (the first dictionary-order match), for
    callers that only care about "is this a known-character request at
    all" (e.g. services/prompt_preference.py). Prefer
    `resolve_all_character_tags` for anything that must handle more than one
    character in the same request.
    """
    resolutions = _resolve_all(user_text)
    return resolutions[0] if resolutions else None


def resolve_all_character_tags(user_text: str) -> tuple[CharacterTagResolution, ...]:
    """Same matching as `resolve_character_tags`, but returns every distinct
    character resolved from `user_text` instead of stopping at the first.
    Each dictionary character entry can contribute at most one resolution
    (it is checked exactly once), so a character mentioned via several of
    its own aliases in the same text still only appears once here.
    """
    return tuple(_resolve_all(user_text))


@dataclass(frozen=True)
class BlockedAliasMention:
    """One already-registered character whose `ambiguous_alias` is present in
    `user_text` but did NOT resolve, because this character's own series
    alias was not also present (the same series-gate `_resolve_all` already
    enforces above). Never produced for a `safe_alias` match, since a
    safe_alias always resolves unconditionally whenever present - there is no
    "blocked" state for it.

    This is the one signal `/character_missing` (services/
    character_missing_finder.py) uses that is not a guess: the character and
    its alias are both already curated dictionary entries, and "did
    resolve_character_tags return it or not" is the exact same ground truth
    resolve_character_tags itself uses - not a separate heuristic.
    """

    character_tag: str
    display_name: str
    series_tag: str
    series_name: str
    matched_alias: str


def find_blocked_alias_mentions(user_text: str) -> tuple[BlockedAliasMention, ...]:
    """Every registered character whose ambiguous_alias appears in
    `user_text` but that did not resolve because the character's own series
    alias was not also present. Read-only: calls the exact same private
    matching helpers `_resolve_all` uses (never modifies them), so this can
    never disagree with what `resolve_character_tags`/`resolve_all_character_tags`
    would actually return for the same text.
    """
    normalized_text = _normalize(user_text)
    if not normalized_text:
        return ()

    resolved_tags = {resolution.character_tag for resolution in _resolve_all(user_text)}
    mentions: list[BlockedAliasMention] = []
    for character in _CHARACTER_ENTRIES:
        if character.character_tag in resolved_tags:
            continue
        matched_ambiguous = _find_alias(normalized_text, character.ambiguous_aliases)
        if matched_ambiguous is None:
            continue
        mentions.append(
            BlockedAliasMention(
                character_tag=character.character_tag,
                display_name=character.display_name,
                series_tag=character.series_tag,
                series_name=_SERIES_NAME_BY_TAG.get(character.series_tag, character.series_tag),
                matched_alias=matched_ambiguous,
            )
        )
    return tuple(mentions)


# ---------------------------------------------------------------------------
# Read-only browsing API (list / search / detail).
#
# These are for surfacing what's registered in this dictionary (e.g. Discord
# `/character_list`, `/character_info`, `/character_search` commands) - they
# do not participate in `resolve_character_tags` / `resolve_all_character_tags`
# matching above, and this module remains the only place character data is
# defined; callers only ever get read-only snapshots derived from it.


@dataclass(frozen=True)
class SeriesListEntry:
    """One row of the browsable series list - identity + its own aliases.

    Read-only snapshot for series-level diagnostics (e.g.
    `/character_validate`'s series-alias duplicate/collision checks), which
    need the full alias tuple that `CharacterListEntry` (only series_tag/
    series_name, no aliases) does not carry.
    """

    series_tag: str
    series_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class CharacterListEntry:
    """One row of the browsable character list - identity + which series."""

    character_tag: str
    display_name: str
    series_tag: str
    series_name: str


@dataclass(frozen=True)
class CharacterDetail:
    """Full detail for one character.

    `extra_fields` holds every `_CharacterEntry` field beyond the core
    identity (character_tag/display_name/series_tag) - currently
    safe_aliases/ambiguous_aliases/notes - built via reflection over
    `_CharacterEntry`'s dataclass fields rather than a hardcoded list. If a
    future field such as appearance_tags or identity_hints is added to
    `_CharacterEntry`/`_RAW_CHARACTERS`, it appears here automatically with
    no change needed in this function or in any command that renders it.
    """

    character_tag: str
    display_name: str
    series_tag: str
    series_name: str
    extra_fields: dict[str, object]


_CHARACTER_DETAIL_CORE_FIELD_NAMES = frozenset({"character_tag", "display_name", "series_tag"})


def _character_list_entry(character: _CharacterEntry) -> CharacterListEntry:
    return CharacterListEntry(
        character_tag=character.character_tag,
        display_name=character.display_name,
        series_tag=character.series_tag,
        series_name=_SERIES_NAME_BY_TAG.get(character.series_tag, character.series_tag),
    )


def list_characters() -> tuple[CharacterListEntry, ...]:
    """Every registered character, in dictionary declaration order.

    Read-only snapshot for `/character_list`. Grouping by series (if wanted
    by the caller) can be done from `series_tag`/`series_name` on each entry.
    """
    return tuple(_character_list_entry(character) for character in _CHARACTER_ENTRIES)


def list_series() -> tuple[SeriesListEntry, ...]:
    """Every registered series, in dictionary declaration order.

    Read-only snapshot for series-level diagnostics (e.g.
    `/character_validate`).
    """
    return tuple(
        SeriesListEntry(series_tag=series.series_tag, series_name=series.series_name, aliases=series.aliases)
        for series in _SERIES_ENTRIES
    )


def _character_detail_from_entry(character: _CharacterEntry) -> CharacterDetail:
    extra_fields = {
        field.name: getattr(character, field.name)
        for field in dataclass_fields(character)
        if field.name not in _CHARACTER_DETAIL_CORE_FIELD_NAMES
    }
    return CharacterDetail(
        character_tag=character.character_tag,
        display_name=character.display_name,
        series_tag=character.series_tag,
        series_name=_SERIES_NAME_BY_TAG.get(character.series_tag, character.series_tag),
        extra_fields=extra_fields,
    )


def get_character_detail(character_tag: str) -> CharacterDetail | None:
    """Full detail for one character by its exact `character_tag`, or None
    if unknown. Read-only snapshot for `/character_info`.
    """
    for character in _CHARACTER_ENTRIES:
        if character.character_tag == character_tag:
            return _character_detail_from_entry(character)
    return None


def list_character_details() -> tuple[CharacterDetail, ...]:
    """Full detail for every registered character, in dictionary declaration
    order. Read-only snapshot for bulk diagnostics (e.g.
    `/character_validate`) - equivalent to calling `get_character_detail` for
    every known `character_tag`, but without the repeated O(n) scans.
    """
    return tuple(_character_detail_from_entry(character) for character in _CHARACTER_ENTRIES)


def find_characters_by_name(query: str) -> tuple[CharacterListEntry, ...]:
    """Characters whose own identity (display_name / character_tag / any of
    its aliases) contains `query` (partial match, same Unicode/casefold
    normalization as alias resolution above). Does not consider series name
    or series aliases - use `search_characters` for that. Read-only snapshot
    for `/character_info`.
    """
    normalized_query = _normalize(query)
    if not normalized_query:
        return ()

    matches: list[CharacterListEntry] = []
    for character in _CHARACTER_ENTRIES:
        haystacks = (
            character.display_name,
            character.character_tag,
            *character.safe_aliases,
            *character.ambiguous_aliases,
        )
        if any(normalized_query in _normalize(text) for text in haystacks if text):
            matches.append(_character_list_entry(character))
    return tuple(matches)


def search_characters(query: str) -> tuple[CharacterListEntry, ...]:
    """Characters found via a broad, series-inclusive partial match: series
    name, series_tag, any series alias, character display_name,
    character_tag, or any character alias containing `query`. Matching on a
    series alias (e.g. "ブルアカ") returns every character in that series.
    Read-only snapshot for `/character_search`.
    """
    normalized_query = _normalize(query)
    if not normalized_query:
        return ()

    matches: list[CharacterListEntry] = []
    for character in _CHARACTER_ENTRIES:
        series = _SERIES_BY_TAG.get(character.series_tag)
        haystacks = [
            character.display_name,
            character.character_tag,
            *character.safe_aliases,
            *character.ambiguous_aliases,
        ]
        if series is not None:
            haystacks.append(series.series_name)
            haystacks.append(series.series_tag)
            haystacks.extend(series.aliases)
        else:
            haystacks.append(character.series_tag)
        if any(normalized_query in _normalize(text) for text in haystacks if text):
            matches.append(_character_list_entry(character))
    return tuple(matches)
