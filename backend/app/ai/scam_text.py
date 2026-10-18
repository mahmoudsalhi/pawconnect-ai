"""Text preparation for the scam detector, shared by training (ml/m6) and the app.

Training data has phone numbers, links and emails replaced by [PHONE], [URL] and [EMAIL], so real
messages get the same treatment before scoring (`mask_contacts`). `normalize` then undoes common
obfuscation ("f l o u s", "fl0uci", "p@yer", "8O dt", stretched Arabic) so disguised words look like
the plain words the model learned. Arabizi digits that stand for letters (3, 5, 7, 9) are left alone.

Pure standard library on purpose: ml/m6 loads this file directly, without the rest of the app.
"""

import re

_EMAIL = re.compile(r"\b[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b", re.I)
_URL = re.compile(
    r"\bhttps?://\S+|\bwww\.\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|tn|net|org|ly|me|io|co|app|link|site|info|xyz|online|gl)\b(?:/\S*)?",
    re.I,
)
# Tunisian numbers (8 digits, optional +216 / 00216, spaces, dots or dashes) and other international numbers.
_PHONE = re.compile(r"(?:(?:\+|00)216[\s.-]?)?(?<![\d.])[2-9]\d(?:[\s.-]?\d){6}(?![\d.])|\+\d{1,3}(?:[\s.-]?\d){7,12}")

_TATWEEL = "ـ"
_LEET = [
    (re.compile(r"(?<=[a-z])@|@(?=[a-z])"), "a"),
    (re.compile(r"(?<=[a-z])€|€(?=[a-z])"), "e"),
    (re.compile(r"(?<=[a-z])!(?=[a-z])"), "i"),
    (re.compile(r"(?<![0-9])(?<=[a-z])0|0(?=[a-z])(?![0-9])"), "o"),
    (re.compile(r"(?<=[a-z])1(?=[a-z])"), "i"),
    (re.compile(r"(?<=[0-9])o\b"), "0"),  # "8O dt" -> "80 dt" (letter O used as zero)
]
# Three or more single characters split by a space, dot, underscore or hyphen: "f l o u s", "t.e.l.e.g.r.a.m".
_SPACED = re.compile(r"(?<!\w)(?:\w[ ._-]){2,}\w(?!\w)")


def mask_contacts(text: str) -> str:
    """Replace emails, links and phone numbers with the [EMAIL] / [URL] / [PHONE] tokens used in training."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub("[URL]", text)
    return _PHONE.sub("[PHONE]", text)


def _join(match: re.Match) -> str:
    return re.sub(r"[ ._-]", "", match.group(0))


def normalize(text: str) -> str:
    t = text.replace(_TATWEEL, "").lower()
    for pattern, repl in _LEET:
        t = pattern.sub(repl, t)
    t = _SPACED.sub(_join, t)
    t = re.sub(r"(?<=[^\W_])_(?=[^\W_])", "", t)  # "kh_a_m_s_i_n" -> "khamsin"
    return re.sub(r"\s+", " ", t).strip()


def prepare(text: str) -> str:
    """Everything a message goes through before the model sees it."""
    return normalize(mask_contacts(text or ""))
