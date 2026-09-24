import json
import re
from typing import Any


_SENTENCE_BOUNDARY = re.compile(r"[.!?]+[\"')\]}\u2019\u201d]*(?=\s|$)")
_PERIOD_TOKEN = re.compile(r"(?:(?:[A-Za-z]\.)+|[A-Za-z]+\.)$")
_HONORIFICS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr."}
_ALWAYS_CONTINUING_ABBREVIATIONS = {"e.g.", "i.e."}
_CLOCK_ABBREVIATIONS = {"a.m.", "p.m."}
_CALENDAR_CONTINUATIONS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "tonight",
}
_STRONG_SENTENCE_STARTERS = {
    "a", "an", "the", "he", "she", "it", "they", "we", "i", "this", "that", "these", "those",
    "however", "nevertheless", "meanwhile", "then",
}


def _period_is_nonterminal(value: str, match: re.Match[str]) -> bool:
    marks = re.match(r"[.!?]+", match.group()).group()
    if marks != ".":
        return False
    period_index = match.start()
    if (period_index > 0 and period_index + 1 < len(value)
            and value[period_index - 1].isdigit() and value[period_index + 1].isdigit()):
        return True
    token_match = _PERIOD_TOKEN.search(value[:period_index + 1])
    if token_match is None:
        return False
    token = token_match.group()
    following = value[match.end():].lstrip()
    normalized = token.casefold()
    if normalized in _ALWAYS_CONTINUING_ABBREVIATIONS:
        following = following.lstrip(",;:").lstrip()
    if not following:
        return False
    next_word_match = re.match(r"[A-Za-z]+", following)
    if next_word_match is None:
        return False
    next_token = next_word_match.group()
    next_word = next_token.casefold()
    if normalized in _HONORIFICS or normalized in _ALWAYS_CONTINUING_ABBREVIATIONS:
        return True
    if len(token) == 2 and token[0].isupper():
        return True
    if normalized in _CLOCK_ABBREVIATIONS:
        return next_word in _CALENDAR_CONTINUATIONS or next_token[0].islower()
    if normalized == "u.s.":
        return next_word not in _STRONG_SENTENCE_STARTERS
    return False


def count_sentences(value: str) -> int:
    """Count summary sentences with deterministic, lightweight punctuation rules.

    Common non-terminal abbreviations and initials are ignored while ordinary
    terminal punctuation, including punctuation before a closing quote, creates a
    boundary. A final unpunctuated fragment counts as one sentence.
    """
    stripped = value.strip()
    if not stripped:
        return 0
    boundaries = []
    for match in _SENTENCE_BOUNDARY.finditer(stripped):
        if not _period_is_nonterminal(stripped, match):
            boundaries.append(match.end())
    count = len(boundaries)
    tail_start = boundaries[-1] if boundaries else 0
    if stripped[tail_start:].strip():
        count += 1
    return count


def normalize_answer(value: str, *, case_sensitive: bool = False) -> str:
    """Conservative normalization for short answers and labels.

    It collapses whitespace, optionally folds case, removes matching surrounding
    quotes, and removes one terminal sentence punctuation mark. Internal
    punctuation, units, signs, and decimal points remain meaningful.
    """
    value = " ".join(value.strip().split())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'`":
        value = value[1:-1].strip()
    if value.endswith((".", "!", "?")) and not re.search(r"\d\.\d[.!?]$", value):
        value = value[:-1].rstrip()
    return value if case_sensitive else value.casefold()


def final_answer(value: str) -> str:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    candidate = lines[-1] if lines else ""
    return re.sub(r"^(?:final(?: answer)?|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE)


def strip_code_fence(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(r"```(?:json|python)?\s*\n?(.*?)\n?```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def parse_json_response(value: str) -> Any:
    return json.loads(strip_code_fence(value))


def flatten_json(value: Any, path: str = "$") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            flattened.update(flatten_json(child, f"{path}.{key}"))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, child in enumerate(value):
            flattened.update(flatten_json(child, f"{path}[{index}]"))
        return flattened
    return {path: value}
