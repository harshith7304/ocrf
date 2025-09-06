import re
import html
from bs4 import BeautifulSoup
import emoji

URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
MULTI_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html5lib")
    return soup.get_text(" ")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = strip_html(text)
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = text.lower()
    text = re.sub(r"[^a-z0-9<>()#@_\-\s\.,:;!?%$]", " ", text)
    text = MULTI_WS_RE.sub(" ", text).strip()
    return text


def concat_fields(row: dict, fields: list[str]) -> str:
    parts = []
    for f in fields:
        v = row.get(f, "")
        if v and isinstance(v, str):
            parts.append(v)
    return "\n".join(parts)
