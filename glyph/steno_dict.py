"""Load and rank real steno chords from Plover's dictionary (see docs/prd.md).

GLYPH NOTE: rather than hand-picking arbitrary Unicode glyphs, shorthand codes come
from Plover (github.com/openstenoproject/plover, GPLv2+), an open-source real-time
stenography engine. Each chord is an actual, human-designed steno brief for that
word or phrase, grounding GlyphLM's compression rules in real stenographic theory
instead of ad hoc guesses. The dictionary is fetched and cached at build time, not
committed to the repo (GPLv2+ data kept separate from this project's own license).
"""

import json
import os
import re
import urllib.request
from collections import Counter

PLOVER_DICT_URL = "https://raw.githubusercontent.com/openstenoproject/plover/main/plover/assets/main.json"
CACHE_PATH = "data/steno_cache/main.json"

# GLYPH NOTE: keep only plain word/short-phrase translations — Plover's dictionary
# also contains formatting "briefs" (e.g. "{^s}", "{,}") and multi-stroke chords
# (chord contains "/"); those aren't plain shorthand-for-a-word/phrase mappings and
# would corrupt encode()/decode() if treated as one.
_CLEAN_VALUE_RE = re.compile(r"^[a-z']+( [a-z']+){0,2}$")


def _load_plover_dict() -> dict[str, str]:
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        urllib.request.urlretrieve(PLOVER_DICT_URL, CACHE_PATH)
    with open(CACHE_PATH) as f:
        return json.load(f)


def _value_to_chord_map() -> dict[str, str]:
    plover = _load_plover_dict()
    value_to_chord: dict[str, str] = {}
    for chord, value in plover.items():
        value = value.lower()
        if "/" in chord or not _CLEAN_VALUE_RE.match(value):
            continue
        # GLYPH NOTE: some words/phrases have multiple valid chords (steno allows
        # several ways to write the same brief); pick the shortest as canonical.
        if value not in value_to_chord or len(chord) < len(value_to_chord[value]):
            value_to_chord[value] = chord
    return value_to_chord


def build_shorthand_map(
    corpus_text: str, max_words: int = 150, max_phrases: int = 30
) -> tuple[dict[str, str], dict[str, str]]:
    """Rank real Plover word/phrase -> chord entries by frequency in corpus_text.

    Returns (word_map, phrase_map), kept separate so phrase substitutions can be
    applied before single-word substitutions in encoder.py.
    """
    value_to_chord = _value_to_chord_map()

    words = re.findall(r"[a-z']+", corpus_text.lower())
    word_freq = Counter(words)
    phrase_freq = Counter(f"{a} {b}" for a, b in zip(words, words[1:]))

    word_hits = [w for w, _ in word_freq.most_common() if w in value_to_chord]
    phrase_hits = [p for p, _ in phrase_freq.most_common() if p in value_to_chord]

    word_map = {w: value_to_chord[w] for w in word_hits[:max_words]}
    phrase_map = {p: value_to_chord[p] for p in phrase_hits[:max_phrases]}
    return word_map, phrase_map
