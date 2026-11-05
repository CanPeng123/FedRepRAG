import re
import string
from typing import Optional

# up to 20 is usually enough for SLAKE counting; expand if needed
_NUM_WORDS = {
    "zero": 0,
    "one": 1, "a": 1, "an": 1, "single": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

# --------- Simple VQA-style normalization for exact-match ----------
_ARTICLES = {"a", "an", "the"}


def _to_int_maybe(s: str) -> Optional[int]:
    """
    Try to interpret s as an integer.
    Accepts:
      - "3"
      - "three"
      - "three (3)" / "3 organs"
    """
    if s is None:
        return None
    s = s.strip().lower()
    if not s:
        return None

    # 1) direct digits anywhere (take the first integer)
    m = re.search(r"\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass

    # 2) number word token
    # remove punctuation then split
    s2 = s.translate(str.maketrans("", "", string.punctuation))
    toks = s2.split()
    for t in toks:
        if t in _NUM_WORDS:
            return int(_NUM_WORDS[t])

    return None

def normalize_answer_with_numbers(s: str) -> str:
    """
    VQA-ish normalization + map number words to digits token-wise.
    """
    if s is None:
        return ""
    s = s.strip().lower()

    # unify yes/no variants (keep if you already do this elsewhere)
    if s in {"y", "yes.", "yeah", "yep", "true"}:
        s = "yes"
    if s in {"n", "no.", "nope", "false"}:
        s = "no"

    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # token-wise conversion
    toks = []
    for t in s.split():
        if t in _ARTICLES:
            continue
        if t in _NUM_WORDS:
            toks.append(str(_NUM_WORDS[t]))
        else:
            toks.append(t)

    return " ".join(toks)

def answers_match(pred: str, gold: str) -> bool:
    """
    Robust matching:
      - if both parse to ints -> compare ints
      - else compare normalized strings
    """
    pi = _to_int_maybe(pred)
    gi = _to_int_maybe(gold)
    if (pi is not None) and (gi is not None):
        return pi == gi

    return normalize_answer_with_numbers(pred) == normalize_answer_with_numbers(gold)


def normalize_answer(s: str) -> str:
    """VQA-ish normalization: lowercase, remove punctuation/articles, squeeze spaces."""
    if s is None:
        return ""
    s = s.strip().lower()

    # unify common yes/no variants
    if s in {"y", "yes.", "yeah", "yep", "true"}:
        s = "yes"
    if s in {"n", "no.", "nope", "false"}:
        s = "no"

    # remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))

    # remove articles
    words = [w for w in s.split() if w not in _ARTICLES]

    # squeeze spaces
    s = " ".join(words)
    return s
