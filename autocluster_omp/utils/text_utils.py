def leading_whitespace(text: str) -> str:
    return text[: len(text) - len(text.lstrip())]
