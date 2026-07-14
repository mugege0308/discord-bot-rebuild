CHAT_RESPONSES = [
    (("おはよう",), "おはよう。\n今日も無理しすぎないでね。"),
    (("こんにちは", "こんにちわ"), "こんにちは。\n何か手伝えること、あるかな。"),
    (("ただいま",), "おかえり。\nひと息ついてからでだいじょうぶだよ。"),
    (("疲れた", "つかれた", "しんどい"), "おつかれさま。\n少し休んでもだいじょうぶだよ。ちゃんと見てるよ。"),
    (("ありがとう", "ありがと"), "どういたしまして。\n少しでも役に立てたなら、うれしいな。"),
    (("おやすみ",), "おやすみ。\nゆっくり休んでね。"),
    (("今日何しよう", "今日なにしよう", "何しよう", "なにしよう"), "まずは今日やることを1つだけ決めるのがよさそうだよ。"),
    (("何できる", "なにできる", "何ができる", "できる？"), "今は /help や操作パネルから、天気・メモ・ToDo・二重翻訳などが使えるよ。"),
]


def get_chat_response(message: str) -> str | None:
    text = message.strip()
    if not text:
        return None

    # 定型文だけで返すため、長い文章は未知コマンド扱いにします。
    if len(text) > 40:
        return None

    normalized = text.replace("　", " ").lower()
    for keywords, response in CHAT_RESPONSES:
        if any(keyword in normalized for keyword in keywords):
            return response

    return None
