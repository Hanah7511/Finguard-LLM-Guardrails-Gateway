"""Matching names written in Arabic script against sanctions lists.

Two ways, because lists differ:
1. Arabic to Arabic. UN and UAE lists often give the name in Arabic too. Both sides are
   normalised (no vowel marks, one form of alef, ...) and compared as text.
2. Arabic to English letters. Most entries (all of OFAC) are only in English letters.
   Arabic usually leaves out short vowels, so both names are reduced to a "consonant key":
   vowels dropped, sounds that are spelled several ways merged (Mohammed, Muhammad and
   the Arabic letters all become "mhmd"). Two names match if their keys are close enough.

A key match is never treated as an exact match: the key loses information, so a person
must check it. Arabic text is written as \\u escapes so copy-paste cannot reverse it.
"""
from __future__ import annotations

import re

_ARABIC_LETTER = re.compile(r"[ء-ي]")
_MARKS = re.compile(r"[ً-ٰٟـ]")  # vowel marks and the stretching line

# Letters with several written forms get one form.
_ONE_FORM = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # alef forms
    "ى": "ي",  # alef maqsura -> ya
    "ة": "ه",  # ta marbuta -> ha
    "ؤ": "و",  # waw with hamza -> waw
    "ئ": "ي",  # ya with hamza -> ya
})

# Arabic letter -> consonant key. "" means the letter is a vowel, or a sound English
# spellings usually leave out (ain, hamza).
_ARABIC_KEY = {
    "ا": "", "و": "", "ي": "", "ء": "", "ع": "",
    "ب": "b", "ت": "t", "ث": "t", "ج": "j", "ح": "h",
    "خ": "k", "د": "d", "ذ": "d", "ر": "r", "ز": "z",
    "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "غ": "j", "ف": "f", "ق": "k", "ك": "k",
    "ل": "l", "م": "m", "ن": "n", "ه": "h",
}

# English spellings -> the same key. Two-letter sounds first. Arabic has no p or v,
# so Arabic spellings of foreign names use b and f.
_LATIN_PAIRS = [("kh", "k"), ("gh", "j"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("ph", "f"), ("ch", "s"), ("ck", "k")]
_LATIN_ONE = str.maketrans({"c": "k", "q": "k", "g": "j", "v": "f", "p": "b", "x": "ks"})
_LATIN_VOWELS = re.compile(r"[aeiouyw]")
_ARTICLES = {"al", "el"}


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER.search(text))


def normalize_arabic(name: str) -> str:
    """Arabic letters only, one form per letter, words separated by one space."""
    text = _MARKS.sub("", name).translate(_ONE_FORM)
    return " ".join(re.findall(r"[ء-ي]+", text))


def _squash(key: str) -> str:
    return re.sub(r"(.)\1+", r"\1", key)  # "nn" -> "n": doubled letters are often written once


def _arabic_word_key(word: str) -> str:
    if word.startswith("ال") and len(word) > 3:  # "al-" written onto the word
        word = word[2:]
    return _squash("".join(_ARABIC_KEY.get(ch, "") for ch in word))


def _latin_word_key(word: str) -> str:
    for pair, key in _LATIN_PAIRS:
        word = word.replace(pair, key)
    word = word.translate(_LATIN_ONE)
    return _squash(_LATIN_VOWELS.sub("", word))


def consonant_key(name: str) -> str:
    """The same key for a name in Arabic script or in English letters. Takes a name
    already cleaned by normalize_name (English) or normalize_arabic (Arabic)."""
    words = name.split()
    if has_arabic(name):
        keys = [_arabic_word_key(w) for w in words]
    else:
        keys = [_latin_word_key(w) for w in words if w not in _ARTICLES]
    return " ".join(k for k in keys if k)
