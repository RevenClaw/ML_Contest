import re
import unicodedata
from functools import lru_cache

try:
    from anyascii import anyascii
except ImportError:
    anyascii = None  # Unit tests can exercise the dependency-free primitives.

LEGAL = {"private", "pvt", "limited", "ltd", "inc", "incorporated", "llc", "llp", "corporation", "corp", "company", "co"}
ADDRESS_ALIASES = {"road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd", "apartment": "apt", "suite": "ste", "floor": "fl", "building": "bldg"}


def normalize(text):
    # Keep combining marks: dropping them corrupts Indic scripts.
    text = unicodedata.normalize("NFKC", text).casefold().replace("&", " and ")
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " " for c in text).split())


def latin(text):
    if anyascii:
        return normalize(anyascii(text))
    return normalize("".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)))


def core_name(text):
    tokens = text.split()
    return " ".join(t for t in tokens if t not in LEGAL) or text


def address_form(text):
    return " ".join(ADDRESS_ALIASES.get(t, t) for t in text.split())


def grams(text):
    text = f" {text} "
    return {text[i:i+3] for i in range(len(text)-2)} if text.strip() else set()


def gram_document(text):
    # Hex tokens preserve spaces and scripts within a trigram under unicode61.
    return " ".join(sorted(g.encode("utf-8").hex() for g in grams(text)))


@lru_cache(maxsize=20000)
def forms(name, address):
    n, a = normalize(name), normalize(address)
    ln, la = latin(name), address_form(latin(address))
    return n, a, ln, la, core_name(ln)


def require_fast_text():
    import rapidfuzz  # noqa: F401
    if anyascii is None:
        raise RuntimeError("Install pinned requirements; anyascii is required for production normalization")

