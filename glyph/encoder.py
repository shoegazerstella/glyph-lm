"""Shorthand text compressor grounded in real Plover steno chords (see docs/prd.md).

Unlike a hand-picked regex list, the word/phrase -> chord mapping is fitted from
the actual training corpus (glyph.steno_dict.build_shorthand_map): the most
frequent words and phrases in the corpus that have a real steno chord get
shortened, everything else is left as-is. Call fit() once (data.py does this while
building the corpora); encode()/decode() then use the fitted, cached mapping.
"""

import json
import os
import re

from glyph.steno_dict import build_shorthand_map

SHORTHAND_MAP_PATH = "data/glyph/shorthand_map.json"

# GLYPH NOTE: chords use characters outside [A-Za-z0-9*-] never appear in them, so
# these lookarounds are a safe substitute for \b (which misbehaves around chords
# starting/ending in '-' or '*', since those are non-word characters).
_CHORD_BOUNDARY = r"[A-Za-z0-9*-]"


class ShorthandCodec:
    def __init__(self, word_map: dict[str, str], phrase_map: dict[str, str]):
        self.word_map = word_map
        self.phrase_map = phrase_map

        combined = {**phrase_map, **word_map}
        self._lookup = combined
        self._reverse = {chord: text for text, chord in combined.items()}

        # GLYPH NOTE: phrases before words, longest-first, so a multi-word phrase
        # gets chorded as a whole instead of being consumed word-by-word first —
        # this is the only way encode() can reduce whitespace-level token count.
        ordered = sorted(phrase_map, key=len, reverse=True) + sorted(word_map, key=len, reverse=True)
        self._encode_re = (
            re.compile(r"\b(?:" + "|".join(re.escape(k) for k in ordered) + r")\b", re.IGNORECASE)
            if ordered
            else None
        )

        chords = sorted(combined.values(), key=len, reverse=True)
        self._decode_re = (
            re.compile(
                rf"(?<!{_CHORD_BOUNDARY})(?:" + "|".join(re.escape(c) for c in chords) + rf")(?!{_CHORD_BOUNDARY})"
            )
            if chords
            else None
        )

    def encode(self, text: str) -> str:
        if self._encode_re is None:
            return text
        return self._encode_re.sub(lambda m: self._lookup[m.group(0).lower()], text)

    def decode(self, text: str) -> str:
        if self._decode_re is None:
            return text
        return self._decode_re.sub(lambda m: self._reverse[m.group(0)], text)

    def special_tokens(self) -> list[str]:
        # GLYPH NOTE: tokenizer.py registers these as BPE special tokens so they're
        # guaranteed atomic vocab entries rather than having to earn a slot through
        # frequency-based merges (see README.md for why that mattered in practice).
        return list(self.phrase_map.values()) + list(self.word_map.values())


_codec: ShorthandCodec | None = None


def fit(corpus_text: str) -> ShorthandCodec:
    """Build the shorthand mapping from corpus_text and cache it to disk."""
    global _codec
    word_map, phrase_map = build_shorthand_map(corpus_text)
    os.makedirs(os.path.dirname(SHORTHAND_MAP_PATH), exist_ok=True)
    with open(SHORTHAND_MAP_PATH, "w") as f:
        json.dump({"words": word_map, "phrases": phrase_map}, f, indent=2)
    _codec = ShorthandCodec(word_map, phrase_map)
    return _codec


def _get_codec() -> ShorthandCodec:
    global _codec
    if _codec is None:
        with open(SHORTHAND_MAP_PATH) as f:
            data = json.load(f)
        _codec = ShorthandCodec(data["words"], data["phrases"])
    return _codec


def encode(text: str) -> str:
    return _get_codec().encode(text)


def decode(text: str) -> str:
    return _get_codec().decode(text)


def get_special_tokens() -> list[str]:
    return _get_codec().special_tokens()
