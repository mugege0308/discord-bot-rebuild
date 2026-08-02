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
- `signature_tags` (tuple[str, ...], optional, default empty): confirmed,
  already-Danbooru-style appearance tags (e.g. `"white hair"`) that should
  always be added to `positive` alongside this character's identity tags.
  This is an intentionally narrow escape hatch from the "no appearance data"
  rule below - use it only when the appearance is actually confirmed for
  this specific character (e.g. distinguishing two dictionary entries that
  share a display name/history, such as blue archive's two "kei" entries
  below), never as a place to guess at a character's usual look.

A malformed entry (missing key, wrong type, unknown series_tag) is skipped
with a logged warning at import time - it never prevents the rest of the
dictionary, or the bot, from loading.

Appearance (hair/eye color, outfit, build, ...) is intentionally NOT part of
this dictionary, with the one narrow exception of the opt-in
`signature_tags` field above. Character *identity* resolution and appearance
reinforcement are otherwise two separate concerns; services/
shogou_prompt_designer.py never fabricates appearance details from this
module's output beyond what `signature_tags` explicitly provides.
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
    # Confirmed appearance tags to always add to positive alongside this
    # character (see _CharacterEntry.signature_tags / module docstring).
    # Empty for every character except where explicitly curated.
    signature_tags: tuple[str, ...] = ()


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
    signature_tags: tuple[str, ...] = ()
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
    {
        "series_tag": "cafe stella to shinigami no chou",
        "series_name": "喫茶ステラと死神の蝶",
        "aliases": (
            "喫茶ステラと死神の蝶",
            "喫茶ステラ",
            "cafe stella",
            "cafe stella to shinigami no chou",
        ),
    },
    {
        # 既存の "inazuma eleven go" とは別作品・別ロースターのため独立した
        # series_tag として登録する（統合しないこと）。
        "series_tag": "inazuma eleven victory road",
        "series_name": "イナズマイレブン 英雄たちのヴィクトリーロード",
        "aliases": (
            "イナズマイレブン 英雄たちのヴィクトリーロード",
            "英雄たちのヴィクトリーロード",
            "ヴィクトリーロード",
            "イナイレV",
            "inazuma eleven victory road",
        ),
    },
    {
        "series_tag": "zenless zone zero",
        "series_name": "ゼンレスゾーンゼロ",
        "aliases": ("ゼンレスゾーンゼロ", "ゼンゼロ", "ZZZ", "zenless zone zero"),
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
        # 2026年実装の「生徒版ケイ」(プレイアブルユニットとしてのケイ)。
        # "ケイ"/"kei" は依然として一般的すぎる短い名前のため series 同時指定を
        # 要求する ambiguous_aliases のままだが、"生徒ケイ"/"ケイ 生徒" は
        # それ自体で十分に特定的なため safe_aliases として単独解決を許可する。
        "character_tag": "kei (blue archive)",
        "display_name": "ケイ",
        "series_tag": "blue archive",
        "safe_aliases": ("生徒ケイ", "ケイ 生徒"),
        "ambiguous_aliases": ("ケイ", "kei"),
        # 画像生成が旧来のアリス身体側のケイへ寄る問題への対処として追加。
        # 外見情報を辞書へ持たせない原則の例外として signature_tags を使う
        # (このケイに限り確認済みの特徴として明示的に指定されたもの)。
        "signature_tags": (
            "white hair",
            "very long hair",
            "pink eyes",
            "sharp eyes",
            "asymmetrical bangs",
        ),
        "notes": (
            "正式な外部タグは未確認のため、短く曖昧な 'kei' を避け、"
            "暫定タグとして 'kei (blue archive)' を採用（正式Danbooruタグ未確認、要確認）。"
            "本エントリは2026年実装の『生徒版ケイ』を指す想定。旧来のアリス身体側のケイは"
            "別エントリ 'kei (former) (blue archive)' として分離した。signature_tagsの5件は"
            "本タスクで明示指定された確認済み特徴として採用（他エントリのような外見の当て推量ではない）。"
        ),
    },
    {
        # 旧来の「アリス身体側」のケイ。生徒版ケイと表示名(display_name)は同じ
        # "ケイ" を共有する同一名の別形態のため、character_tag/display_name/
        # aliases のいずれも生徒版ケイと重複させず、明示的なエイリアスでのみ
        # 解決させる (series同時指定なしで単独解決できる safe_aliases)。
        "character_tag": "kei (former) (blue archive)",
        "display_name": "ケイ（旧アリス身体）",
        "series_tag": "blue archive",
        "safe_aliases": ("アリス身体のケイ", "旧ケイ"),
        "notes": (
            "正式な外部タグ・正式名称ともにリポジトリ内で未確認の暫定タグ。"
            "生徒版ケイ('kei (blue archive)')と同じ 'ケイ' という名前を共有する別形態のため、"
            "混同防止のため character_tag を 'kei (former) (blue archive)' として分離した。"
            "'former' は本辞書がこの区別のために付けた仮の英語表記であり、Danbooru上の実タグとは"
            "無関係（要確認）。外見タグは追加していない(旧形態の外見は本タスクで確認されていないため)。"
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
    {
        "character_tag": "chiaki (blue archive)",
        "display_name": "元宮チアキ",
        "series_tag": "blue archive",
        "safe_aliases": ("元宮チアキ", "motomiya chiaki", "chiaki motomiya"),
        # "チアキ"/"chiaki" alone is also this dictionary's existing
        # "nanami chiaki" (Danganronpa) given name - nanami chiaki has no
        # bare "チアキ"/"chiaki" alias registered (safe_aliases only), so
        # there is no alias collision, but this character's bare given name
        # still stays series-gated on principle.
        "ambiguous_aliases": ("チアキ", "chiaki"),
        # 髪色は資料間で不一致（黒髪 vs 青髪の両説あり）のため signature_tags
        # には含めない。角(horns)と長髪は複数資料で一致。
        "signature_tags": ("long hair", "horns"),
        "notes": (
            "character_tagはリポジトリ内・Danbooru上で未確認。既存の 'kei (blue archive)' で確認された"
            "曖昧タグ問題（bare 'kei' が別キャラ寄りに混ざる）と同型の問題を避けるため、'chiaki' 単独ではなく"
            "'chiaki (blue archive)' を暫定character_tagとして採用。要確認。"
            "固有武器名「カメラブリッツ」は実際のカメラ機能を持たない小銃であり、別途目撃される"
            "デジタルカメラ小道具とは別物のため、'camera'系タグはsignature_tagsに含めていない。"
            "目の色（赤目、単独ソースのみ）も確証不足のため未採用。"
        ),
    },
    {
        "character_tag": "ichikawa hinana",
        "display_name": "市川雛菜",
        "series_tag": "idolmaster shiny colors",
        "safe_aliases": ("市川雛菜", "ichikawa hinana", "hinana ichikawa"),
        "ambiguous_aliases": ("雛菜", "ひなな", "hinana"),
        "signature_tags": (),
        "notes": (
            "character_tagはリポジトリ内で未確認。保守的な姓名順ローマ字表記を仮採用。要確認。"
            "髪色・髪型・目の色ともに信頼できる資料で確認できなかったため signature_tags は空のまま登録。"
            "所属ユニット「ノクチル」の他メンバー（芹沢あさひ・樋口円香・福丸小糸など）との外見上の"
            "区別に使える確定情報も今回は得られなかった。将来的な視覚資料での確認を推奨。"
        ),
    },
    {
        "character_tag": "shiki natsume",
        "display_name": "四季ナツメ",
        "series_tag": "cafe stella to shinigami no chou",
        "safe_aliases": ("四季ナツメ", "shiki natsume", "natsume shiki"),
        # "ナツメ"/"natsume" alone is also an ordinary Japanese given name
        # (and the word for jujube), so it stays series-gated.
        "ambiguous_aliases": ("ナツメ", "natsume"),
        "signature_tags": ("black hair", "long hair"),
        "notes": (
            "character_tag・series_tagともにリポジトリ内・Danbooru上で未確認。保守的な姓名順ローマ字/"
            "ローマ字化タイトルを仮採用。要確認。signature_tagsの黒髪・ロングヘアは複数の検索結果で"
            "共通して言及されたが、キャラクター専門wikiでの明示記載ではなく暫定採用（要確認）。"
            "目の色は確認できず。作中設定上、本人は死神ではなく人間で、死亡時に魂の一部が蝶となった"
            "という設定のため、尻尾・翼等の死神的な身体的特徴は本人には適用しないこと。"
        ),
    },
    {
        "character_tag": "shinohara raika",
        "display_name": "忍原来夏",
        "series_tag": "inazuma eleven victory road",
        "safe_aliases": ("忍原来夏", "shinohara raika", "raika shinohara"),
        # "来夏"/"らいか"/"raika" alone reads as an ordinary Japanese given
        # name, so it stays series-gated.
        "ambiguous_aliases": ("来夏", "らいか", "raika"),
        "signature_tags": ("pink hair", "ponytail", "pink eyes"),
        "notes": (
            "character_tagはリポジトリ内・Danbooru上で未確認。保守的な姓名順ローマ字表記を仮採用。要確認。"
            "出典は『イナズマイレブン 英雄たちのヴィクトリーロード』であり、既存の series_tag "
            "'inazuma eleven go' とは別作品・別ロースターのため series_tag を分離した（混同しないこと）。"
            "ピンク髪・ロングポニーテール・ピンク目は独立した2件の情報源で一致したため signature_tags に採用。"
        ),
    },
    {
        "character_tag": "castorice",
        "display_name": "キャストリス",
        "series_tag": "honkai: star rail",
        # "キャストリス"/"castorice" はいずれも一般語・一般的な人名ではない
        # 固有名のため、safe_aliases として単独解決可能とする
        # (ambiguous_aliasesは不要)。
        "safe_aliases": ("キャストリス", "castorice"),
        "signature_tags": ("purple hair", "very long hair", "purple eyes", "pointy ears"),
        "notes": (
            "character_tagは公式ローカライズ名 'Castorice' をそのまま採用（他の新規エントリより確度が高い）。"
            "ただしDanbooru上の実タグそのものはリポジトリ内で未確認。要確認。"
            "髪色・目の色・尖った耳は独立した2件の情報源で一致したため signature_tags に採用。"
            "蝶モチーフの髪飾りや武器（大鎌）については単独ソースかつ状況証拠のみで確証不足のため含めていない。"
        ),
    },
    {
        "character_tag": "remielle dan",
        "display_name": "レミエール・ダン",
        "series_tag": "zenless zone zero",
        # "remielle"/"レミエール" 単独でも一般語ではない固有名のため
        # safe_aliases として単独解決可能とする（ambiguous_aliasesは不要）。
        # character_tagは "remielle" 単独ではなくフルネーム "remielle dan" を
        # 優先的に採用しているが、単独形も同じ人物を指すエイリアスとして
        # safe_aliases に含める。
        "safe_aliases": ("レミエール", "レミエール・ダン", "remielle", "remielle dan"),
        "signature_tags": ("pink hair", "very long hair", "pink eyes", "wings"),
        "notes": (
            "character_tagは 'remielle' 単独ではなくフルネーム 'remielle dan' を暫定採用。"
            "Danbooru上の実タグそのものはリポジトリ内で未確認。旧コードネーム「Ramiel」の存在も"
            "示唆されており、表記揺れの可能性がある。要確認。"
            "髪色・目の色・翼は単独の情報源（タグ集計サイト）のみに基づく暫定採用のため、"
            "外見特徴全体の確証は他エントリより弱いことに注意。"
        ),
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
                signature_tags=tuple(str(tag) for tag in raw.get("signature_tags", ())),
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
            # A safe_alias always resolves to this character's own
            # series_tag - never to whatever series happens to be mentioned
            # elsewhere in the same text. Without this check, a request
            # naming both this character's safe_alias AND an unrelated
            # (wrong) series - e.g. two similarly-named series like
            # "inazuma eleven go" vs "inazuma eleven victory road" - would
            # silently overwrite the character's real series_tag with the
            # wrong one. match_type only reflects whether the text also
            # happened to confirm the character's own series.
            match_type = "character_and_series" if matched_series_tag == character.series_tag else "character_only"
            resolutions.append(
                CharacterTagResolution(
                    character_tag=character.character_tag,
                    display_name=character.display_name,
                    series_tag=character.series_tag,
                    series_name=_SERIES_NAME_BY_TAG.get(character.series_tag, character.series_tag),
                    matched_character_alias=matched_safe_alias,
                    matched_series_alias=matched_series_alias if matched_series_tag == character.series_tag else None,
                    match_type=match_type,
                    signature_tags=character.signature_tags,
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
                    signature_tags=character.signature_tags,
                )
            )

    return _drop_alias_subsumed_matches(resolutions)


def _drop_alias_subsumed_matches(
    resolutions: list[CharacterTagResolution],
) -> list[CharacterTagResolution]:
    """Same-series safety valve for two dictionary entries that share an
    underlying name (e.g. blue archive's "kei (blue archive)" 生徒版 vs "kei
    (former) (blue archive)" 旧形態, both matched via a Japanese alias
    containing "ケイ"). Japanese alias matching is plain substring
    containment (see `_alias_present`), so a more specific character's own
    alias (e.g. "旧ケイ") always also contains a less specific sibling
    character's alias (e.g. "ケイ") as a literal substring. Without this
    step, mentioning the more specific character together with a series name
    (e.g. "ブルアカの旧ケイ") would resolve BOTH characters at once instead of
    only the one actually named.

    Drops a resolution only when another resolution in the same series has a
    matched_character_alias that literally contains this one's
    matched_character_alias as a substring (the more specific match wins).
    Every one of this dictionary's other alias pairs is already
    substring-disjoint (verified by the full test suite), so this is a no-op
    for every character except the one pair it exists for.
    """
    if len(resolutions) < 2:
        return resolutions

    subsumed_tags: set[str] = set()
    for shorter in resolutions:
        shorter_alias = _normalize(shorter.matched_character_alias)
        if not shorter_alias:
            continue
        for longer in resolutions:
            if longer.character_tag == shorter.character_tag or longer.series_tag != shorter.series_tag:
                continue
            longer_alias = _normalize(longer.matched_character_alias)
            if longer_alias != shorter_alias and shorter_alias in longer_alias:
                subsumed_tags.add(shorter.character_tag)
                break

    if not subsumed_tags:
        return resolutions
    return [resolution for resolution in resolutions if resolution.character_tag not in subsumed_tags]


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
    matching helpers `_resolve_all`/`_match_series` use (never modifies
    them), so this can never disagree with what
    `resolve_character_tags`/`resolve_all_character_tags` would actually
    return for the same text.

    Checks "was this character's own series alias present" directly via
    `_match_series` rather than merely "is this character_tag absent from
    `_resolve_all`'s output" - the latter would also be true for a character
    dropped by `_drop_alias_subsumed_matches` (a more specific sibling
    character, e.g. blue archive's two "kei" entries, matched instead), which
    is a different situation from "series alias missing" and must not be
    reported as one.
    """
    normalized_text = _normalize(user_text)
    if not normalized_text:
        return ()

    resolved_tags = {resolution.character_tag for resolution in _resolve_all(user_text)}
    matched_series_tag, _matched_series_alias = _match_series(normalized_text)
    mentions: list[BlockedAliasMention] = []
    for character in _CHARACTER_ENTRIES:
        if character.character_tag in resolved_tags:
            continue
        if matched_series_tag == character.series_tag:
            # Its own series alias WAS present - this is not the "series
            # missing" case find_missing_character_candidates surfaces, even
            # though this character_tag ended up absent from _resolve_all's
            # output (e.g. subsumed by a more specific sibling character).
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
