import re


def normalize_text(text):
    return re.sub( '\\s+', ' ', text).strip()
