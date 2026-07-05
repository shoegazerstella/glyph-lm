"""Deterministic, rule-based shorthand text compressor (see docs/prd.md)."""

import re

# GLYPH NOTE: rule order matters. Whole-word substitutions run first (they rely on
# \b word boundaries that must see the original word shapes), suffix rules run next,
# and double-vowel collapse runs last. Reordering these could cause a suffix or vowel
# rule to corrupt a glyph inserted by an earlier rule.
RULES: list[tuple[str, str]] = [
    (r"\bthe\b", "þ"),
    (r"\band\b", "&"),
    (r"\bthat\b", "ŧ"),
    (r"\bwith\b", "w/"),
    (r"\bfor\b", "4"),
    (r"\byou\b", "u"),
    (r"\bare\b", "r"),
    (r"\bnot\b", "ñ"),
    (r"\bhave\b", "hv"),
    (r"\bthis\b", "þs"),
    (r"tion\b", "ʃ"),
    (r"ing\b", "ŋ"),
    (r"ment\b", "mnt"),
    (r"ness\b", "ns"),
    (r"ould\b", "ld"),
    (r"ight\b", "ıt"),
    (r"([aeiou])\1+", r"\1"),
]

# GLYPH NOTE: re.IGNORECASE is applied so sentence-initial capitalized words
# ("The", "And", ...) — common throughout the Shakespeare corpus — still match the
# whole-word rules. This trades away capitalization fidelity for much better
# compression coverage; decode() does not attempt to restore original casing.
_COMPILED_RULES = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in RULES]

# GLYPH NOTE: decode() is intentionally approximate, not a true inverse. Several
# rules are not invertible (double-vowel collapse discards information), and 'u'/'r'
# are excluded below because reversing them would corrupt real occurrences of the
# words "u" and "r" that weren't produced by the "you"/"are" rules. This is fine per
# the PRD: decode exists only to make glyph-model completions human-readable.
_REVERSE_RULES: list[tuple[str, str]] = [
    ("þs", "this"),
    ("þ", "the"),
    ("&", "and"),
    ("ŧ", "that"),
    ("w/", "with"),
    ("4", "for"),
    ("ñ", "not"),
    ("hv", "have"),
    ("ʃ", "tion"),
    ("ŋ", "ing"),
    ("mnt", "ment"),
    ("ns", "ness"),
    ("ld", "ould"),
    ("ıt", "ight"),
]


def encode(text: str) -> str:
    for pattern, repl in _COMPILED_RULES:
        text = pattern.sub(repl, text)
    return text


def decode(text: str) -> str:
    for glyph, word in _REVERSE_RULES:
        text = text.replace(glyph, word)
    return text
