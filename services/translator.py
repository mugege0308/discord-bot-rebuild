import logging


logger = logging.getLogger("secretary-bot.translator")


def translate_text(text: str, source: str, target: str) -> str:
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=source, target=target).translate(text)


def translate_ja_to_en_to_ja(text: str) -> str:
    english = translate_text(text, source="ja", target="en")
    return translate_text(english, source="en", target="ja")
