"""Checks that the numbers in an answer appear in the sources it cites.

For every quantity in the answer ($9.5B, +24%, 1,350, 2.5x, (420)bps):
- "ok": it appears in a source that its sentence cites (or, if the sentence
  cites nothing, in any source the model read);
- "other_source": it appears only in a source the sentence doesn't cite,
  which usually means a wrong citation or numbers mixed across sources;
- "not_found": it appears in none of the sources the model read.

This is a safety net, not a judge. It compares values, so "9.5 billion" in
an answer matches "$9.5B" in a source, but a number converted to another
unit ("9,500 million") is reported as not found even if it is right. And a
number that is found may still be used in the wrong place.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

# A number with what can come before and after it.
NUMBER = re.compile(
    r"""
    (?P<before>(?:US\$|AED|USD|EUR|GBP|[$£€¥₹])\s?)?
    (?P<open>\()?
    (?P<sign>[-+−])?
    (?P<core>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?P<close>\))?
    (?P<after>\s?(?:%|percent\b|pts?\b|pp\b|bps\b|x\b|bn\b|b\b|mn\b|m\b|k\b|p\b|¢|billion\b|million\b|thousand\b))?
    """,
    re.X | re.I,
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\[A-Z0-9$£€(+\-*])")


@dataclass
class NumberCheck:
    text: str  # the number as written in the answer
    status: str  # "ok", "other_source" or "not_found"
    cited: list[int]  # the sources its sentence cites
    found_in: list[int]  # the sources that contain it


@dataclass(frozen=True)
class Quantity:
    text: str
    value: float
    percent: bool


def _value(match: re.Match) -> float:
    return round(float(match.group("core").replace(",", "")), 6)


def _is_percent(match: re.Match) -> bool:
    after = (match.group("after") or "").strip().lower()
    return after in ("%", "percent")


def quantities(text: str) -> list[Quantity]:
    """The numbers in an answer worth checking: amounts, rates and ratios, not years or labels."""
    found = []
    for match in NUMBER.finditer(text):
        start, end = match.span()
        core = match.group("core")
        before_char = text[start - 1] if start > 0 else ""
        after_text = text[end : end + 2]

        # Period labels like 4Q'25, Q3, H1, FY25, 2025E, '24.
        if before_char.isalpha() or before_char in "'’" or after_text[:1].isalpha():
            continue
        if re.match(r"['’]\d", after_text):
            continue
        has_marker = bool(match.group("before") or match.group("after") or "." in core or "," in core)
        value = _value(match)
        if not has_marker:
            if value < 10:  # small counts like "3 segments"
                continue
            if 1900 <= value <= 2100 and len(core) == 4:  # years
                continue
        found.append(Quantity(match.group(0).strip(), value, _is_percent(match)))
    return found


def source_values(text: str) -> set[tuple[float, bool]]:
    """Every number in a source, as (value, is_percent)."""
    return {(_value(m), _is_percent(m)) for m in NUMBER.finditer(text)}


def _contains(values: set[tuple[float, bool]], quantity: Quantity) -> bool:
    if quantity.percent:
        return (quantity.value, True) in values
    return (quantity.value, False) in values or (quantity.value, True) in values


def _cited(text: str, source_count: int) -> list[int]:
    numbers = []
    for match in CITATION.finditer(text):
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            n = int(part)
            if 1 <= n <= source_count and n not in numbers:
                numbers.append(n)
    return numbers


def sentences_with_citations(answer: str, source_count: int) -> list[tuple[str, list[int]]]:
    """The answer cut into sentences, each with the sources it cites (citations removed from the text).

    A line of only citations ("[1]" at the end, a common style) applies to the
    sentences before it that cite nothing themselves.
    """
    result: list[tuple[str, list[int]]] = []
    waiting: list[int] = []  # positions of sentences still without a citation
    for line in answer.splitlines():
        for sentence in SENTENCE_END.split(line):
            cited = _cited(sentence, source_count)
            text = CITATION.sub(" ", sentence)
            if not re.search(r"\w", text):  # blank, or only citations (and punctuation)
                if not cited:
                    continue
                for position in waiting:
                    result[position] = (result[position][0], cited)
                waiting = []
                continue
            if not cited:
                waiting.append(len(result))
            result.append((text, cited))
    return result


def check_numbers(answer: str, sources: dict[int, str]) -> dict:
    """Check every number in `answer` against `sources` (source number -> the text the model read)."""
    values = {number: source_values(text) for number, text in sources.items()}
    checks: list[NumberCheck] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()

    for sentence, cited in sentences_with_citations(answer, len(sources)):
        for quantity in quantities(sentence):
            key = (quantity.text, tuple(cited))
            if key in seen:
                continue
            seen.add(key)
            found_in = [number for number in sorted(values) if _contains(values[number], quantity)]
            if not found_in:
                status = "not_found"
            elif not cited or any(number in found_in for number in cited):
                status = "ok"
            else:
                status = "other_source"
            checks.append(NumberCheck(quantity.text, status, cited, found_in))

    summary = {
        "checked": len(checks),
        "ok": sum(c.status == "ok" for c in checks),
        "other_source": sum(c.status == "other_source" for c in checks),
        "not_found": sum(c.status == "not_found" for c in checks),
    }
    return {"summary": summary, "numbers": [asdict(c) for c in checks]}
