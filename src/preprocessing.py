from __future__ import annotations
"""
Light-weight text cleaning helpers for Quora Question Pairs.
-----------------------------------------------------------
1. Unicode → ASCII, lower-case
2. Strip [math]...[/math] and HTML tags
3. Keep letters, numbers, apostrophes, '?'
4. Tokenise with regex
5. Drop NLTK stop-words
6. Porter-stem each token
"""

import math, numpy as np
import re, unicodedata
from functools import lru_cache
from typing import List
import nltk

# ------------------------------------------------------------------ #
# NLTK setup (downloads once) -------------------------------------- #
try:
    _stop = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    _stop = nltk.corpus.stopwords.words("english")

STOPWORDS = set(_stop)
_STEMMER  = nltk.PorterStemmer()

# ------------------------------------------------------------------ #
# Regex pre-compiles ------------------------------------------------ #
_MATH_RE  = re.compile(r"\[math].*?\[/math]", re.IGNORECASE | re.DOTALL)
_HTML_RE  = re.compile(r"<[^>]+>")
_BAD_CHR  = re.compile(r"[^a-z0-9'? ]+")
_TOKEN_RE = re.compile(r"[a-z0-9']+|\?")

@lru_cache(maxsize=200_000)
def _stem(tok: str) -> str:
    return _STEMMER.stem(tok)

def _ascii_lower(text: str) -> str:
    return (unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower())

# ------------------------------------------------------------------ #
def clean_text(text: str | None) -> str:
    """Return space-joined cleaned tokens."""
    # guard against NaN or None
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    text = _ascii_lower(text)
    text = _MATH_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _BAD_CHR.sub(" ", text)
    tokens = _TOKEN_RE.findall(text)
    return " ".join(_stem(t) for t in tokens if t not in STOPWORDS)

# ------------------------------------------------------------------ #
def clean_and_stats(text: str | None) -> tuple[str, int, int]:
    cleaned = clean_text(text)
    return cleaned, len(cleaned), len(cleaned.split())

# ------------------------------------------------------------------ #
def clean_series(series) -> List[str]:
    return [clean_text(x) for x in series.tolist()]

# Smoke-test
if __name__ == "__main__":
    s = "What is the step-by-step guide to invest in share market in India?"
    c, ln, wc = clean_and_stats(s)
    print("RAW :", s)
    print("CLEAN:", c, "| len", ln, "| words", wc)