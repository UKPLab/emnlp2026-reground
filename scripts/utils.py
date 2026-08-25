"""
Span-aware sentence splitter and reference-to-sentence mapper.

Key properties:
- Input text is never modified.
- Reference spans and abbreviations are protected from sentence splitting.
- Sentence boundaries ignore punctuation inside protected spans.
- Numbered list markers (e.g., "1.") are not treated as sentence ends.
- Blank lines, markdown list items, blockquotes, and headings act as hard boundaries.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Tuple


# ================================
# Regex primitives
# ================================

WS = r"\s+"
OPT_WS = r"\s*"
# Optional same-line whitespace between a reference keyword and its number.
# Deliberately not \s* so a match cannot cross a newline (e.g. "Table" \n "1").
SEP = r"[ \t]*"

NUM = r"\d+(?:\.\d+)*[A-Za-z]?"
RANGE = rf"{NUM}(?:{OPT_WS}[-–—]{OPT_WS}{NUM})?"
SERIES = rf"{RANGE}(?:{OPT_WS}(?:,|and|&|to){OPT_WS}{RANGE})*"

# Line references. Two forms:
#   (i)  the word "line(s)" or the dotted abbreviation "l." — allows 1+ digits.
#   (ii) the bare "L###" convention — requires 2+ digits so math symbols like
#        "L1"/"L2" (norms, losses) are not misread as line references.
LINE_RE = re.compile(
    r"""
    (?<![A-Za-z])
    (?:
        (?:lines?|l\.)\s*\(?\d+(?:\s*(?:[-–—]|to)\s*\d+)?\)?
      | [lL]\d{2,}(?:\s*[-–—]\s*[lL]?\d+)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ================================
# Data models
# ================================

@dataclass(frozen=True)
class ReferenceMention:
    kind: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


# ================================
# Abbreviations and hard boundaries
# ================================

ABBR_PATTERN = re.compile(
    r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|et al\.)",
    re.IGNORECASE,
)

LIST_ITEM_START = re.compile(r"(?m)^\s*(?:>\s*)?(?:[*\-+]|(?:\d+)[.)])\s+")
BLOCKQUOTE_START = re.compile(r"(?m)^\s*>\s+")
HEADING_START = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")
BLANKLINE_BOUNDARY = re.compile(r"\n")
_NUMBERED_MARKER_DOT = re.compile(r"^\s*(?:>\s*)?\d+\.$")


# ================================
# Reference patterns
# ================================

def build_reference_patterns() -> Dict[str, re.Pattern]:
    return {
        "section": re.compile(
            rf"""
            (?:\bsections?|\bsecs?\.?|§){SEP}{SERIES}
            |
            \b(?:introduction|related{WS}work|method(?:s|ology)?|approach|
               experiments?|results?|analysis|discussion|limitations?|
               conclusion|ablation|future{WS}work)
            {WS}section\b
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
        "table": re.compile(rf"\b(?:tables?|tabs?\.?){SEP}{SERIES}", re.IGNORECASE),
        "figure": re.compile(rf"\b(?:figures?|figs?\.?){SEP}{SERIES}", re.IGNORECASE),
        "appendix": re.compile(rf"\bappendix{SEP}[A-Za-z](?:\.\d+)*\b", re.IGNORECASE),
        "equation": re.compile(
            rf"\b(?:equations?|eqs?\.?){SEP}\(?{SERIES}\)?",
            re.IGNORECASE,
        ),
        "page": re.compile(
            rf"(?<![A-Za-z])(?:pages?|pp?\.?){SEP}\d+(?:{OPT_WS}[-–—]{OPT_WS}\d+)?",
            re.IGNORECASE,
        ),
        "footnote": re.compile(rf"\b(?:footnotes?|fn\.?){SEP}\d+", re.IGNORECASE),
        "line": LINE_RE,
    }


# ================================
# Interval helpers
# ================================

def _merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    intervals = sorted(intervals)
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _in_intervals(pos: int, intervals: List[Tuple[int, int]], starts: List[int]) -> bool:
    i = bisect_right(starts, pos) - 1
    return i >= 0 and intervals[i][0] <= pos < intervals[i][1]


def _is_numbered_list_marker_dot(text: str, dot_index: int) -> bool:
    if dot_index < 0 or dot_index >= len(text) or text[dot_index] != ".":
        return False
    line_start = text.rfind("\n", 0, dot_index) + 1
    return bool(_NUMBERED_MARKER_DOT.match(text[line_start : dot_index + 1]))


# ================================
# Reference detection
# ================================

def find_reference_mentions(
    text: str,
    ref_patterns: Optional[Dict[str, re.Pattern]] = None,
) -> List[ReferenceMention]:
    if ref_patterns is None:
        ref_patterns = get_reference_patterns()

    mentions: List[ReferenceMention] = []
    for kind, pat in ref_patterns.items():
        for m in pat.finditer(text):
            mentions.append(
                ReferenceMention(kind, m.group(0), m.start(), m.end())
            )

    mentions.sort(key=lambda x: (x.start, x.end, x.kind))
    deduped = []
    last = None
    for m in mentions:
        key = (m.start, m.end, m.kind)
        if key != last:
            deduped.append(m)
            last = key
    return deduped


def _protected_spans_for_splitting(
    text: str,
    ref_mentions: List[ReferenceMention],
    abbr_pattern: re.Pattern,
) -> List[Tuple[int, int]]:
    spans = [(m.start, m.end) for m in ref_mentions]
    spans.extend([m.span() for m in abbr_pattern.finditer(text)])
    return _merge_intervals(spans)


# ================================
# Sentence splitting
# ================================

_CLOSERS = set("\"')]}»”’")


def split_sentences_span_aware(
    text: str,
    ref_patterns: Optional[Dict[str, re.Pattern]] = None,
    abbr_pattern: re.Pattern = ABBR_PATTERN,
) -> List[SentenceSpan]:
    refs = find_reference_mentions(text, ref_patterns)
    protected = _protected_spans_for_splitting(text, refs, abbr_pattern)
    protected_starts = [s for s, _ in protected]

    forced_starts = set()

    for m in BLANKLINE_BOUNDARY.finditer(text):
        if m.end() < len(text):
            forced_starts.add(m.end())

    for m in HEADING_START.finditer(text):
        start = m.start()
        forced_starts.add(start)
        line_end = text.find("\n", start)
        if line_end == -1:
            line_end = len(text)
        line = text[start:line_end]
        colon = line.find(":")
        if colon != -1:
            forced_starts.add(start + colon + 1)
        elif line_end < len(text):
            forced_starts.add(line_end + 1)

    for pat in (LIST_ITEM_START, BLOCKQUOTE_START):
        for m in pat.finditer(text):
            if m.start() != 0:
                forced_starts.add(m.start())

    cut_points = [0]
    n = len(text)
    i = 0

    while i < n:
        ch = text[i]
        if ch in ".!?":
            if _in_intervals(i, protected, protected_starts):
                i += 1
                continue
            if ch == "." and _is_numbered_list_marker_dot(text, i):
                i += 1
                continue
            j = i + 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            if j >= n:
                cut_points.append(n)
                break
            if text[j].isspace():
                k = j
                while k < n and text[k].isspace():
                    k += 1
                cut_points.append(k)
                i = k
                continue
        i += 1

    cut_points = sorted(set(cut_points) | forced_starts | {0, n})

    sentences = []
    for a, b in zip(cut_points, cut_points[1:]):
        seg = text[a:b]
        if not seg.strip():
            continue
        l = len(seg) - len(seg.lstrip())
        r = len(seg) - len(seg.rstrip())
        s0, s1 = a + l, b - r
        if s0 < s1:
            sentences.append(
                SentenceSpan(text=text[s0:s1].strip(), start=s0, end=s1)
            )

    return sentences


# ================================
# Mapping references to sentences
# ================================

def map_references_to_sentences(
    text: str,
    ref_patterns: Optional[Dict[str, re.Pattern]] = None,
):
    if ref_patterns is None:
        ref_patterns = get_reference_patterns()

    refs = find_reference_mentions(text, ref_patterns)
    sents = split_sentences_span_aware(text, ref_patterns)

    out = []
    r = 0
    for sent in sents:
        sent_refs = []
        while r < len(refs) and refs[r].end <= sent.start:
            r += 1
        j = r
        while j < len(refs) and refs[j].start < sent.end:
            if sent.start <= refs[j].start and refs[j].end <= sent.end:
                sent_refs.append(
                    {
                        "kind": refs[j].kind,
                        "text": refs[j].text,
                        "span": (refs[j].start, refs[j].end),
                    }
                )
            j += 1
        out.append(
            {
                "sentence": sent.text,
                "span": (sent.start, sent.end),
                "references": sent_refs,
            }
        )

    return out


def safe_sentence_split(
    text: str,
    ref_patterns: Optional[Dict[str, re.Pattern]] = None,
) -> List[str]:
    return [s.text for s in split_sentences_span_aware(text, ref_patterns)]


@lru_cache(maxsize=1)
def get_reference_patterns() -> Dict[str, re.Pattern]:
    return build_reference_patterns()


def detect_references_in_text(
    text: str,
    ref_patterns: Optional[Dict[str, re.Pattern]] = None,
    *,
    return_mentions: bool = False,
    dedupe_text: bool = False,
) -> Dict[str, List]:
    if ref_patterns is None:
        ref_patterns = get_reference_patterns()

    refs = find_reference_mentions(text, ref_patterns)
    grouped = defaultdict(list)

    if return_mentions:
        for r in refs:
            grouped[r.kind].append(r)
        return dict(grouped)

    for r in refs:
        grouped[r.kind].append(r.text)

    if dedupe_text:
        grouped = {k: list(dict.fromkeys(v)) for k, v in grouped.items()}

    return dict(grouped)


# ================================
# Reference normalization (shared gold extraction for all retrieval runners)
# ================================

_NORM_PREFIX_RE = re.compile(
    r"""(?ix)^\s*
    (?:section|sec|§|table|tab|figure|fig|appendix|equation|eq|
       lines?|l|page|pp?|footnote|fn)
    s?\.?\s*"""
)


def _expand_numeric_range(label: str) -> List[str]:
    """Expand conservative integer / dotted ranges; leave ambiguous ones intact.
    "4-6" -> ["4","5","6"]; "4.1-4.3" -> ["4.1","4.2","4.3"]; "7A-7C" -> ["7A-7C"]."""
    s = label.replace("–", "-").replace("—", "-")
    if "-" not in s:
        return [label]
    a, b = [p.strip() for p in s.split("-", 1)]
    if re.fullmatch(r"\d+", a) and re.fullmatch(r"\d+", b):
        lo, hi = int(a), int(b)
        if lo <= hi and (hi - lo) <= 5000:
            return [str(x) for x in range(lo, hi + 1)]
        return [label]
    if re.fullmatch(r"\d+(?:\.\d+)+", a) and re.fullmatch(r"\d+(?:\.\d+)+", b):
        pa, pb = a.split("."), b.split(".")
        if len(pa) == len(pb) and pa[:-1] == pb[:-1] and pa[-1].isdigit() and pb[-1].isdigit():
            lo, hi = int(pa[-1]), int(pb[-1])
            if lo <= hi and (hi - lo) <= 5000:
                prefix = ".".join(pa[:-1])
                return [f"{prefix}.{i}" for i in range(lo, hi + 1)]
        return [label]
    return [label]


def normalize_reference(ref_type: str, match_text: str, expand_ranges_for=("Figure", "Table")):
    """Normalize a detected reference into [{"type": <Cap>, "label": <id>}].
    ref_type is a find_reference_mentions kind (line/section/figure/...); match_text is its surface."""
    ref_type = ref_type.capitalize()
    cleaned = _NORM_PREFIX_RE.sub("", match_text)
    cleaned = re.sub(r"[\(\)\[\]]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if ref_type == "Appendix":
        return [{"type": "Appendix", "label": cleaned.upper()}]

    if ref_type == "Section" and cleaned.lower().endswith(" section"):
        cleaned = cleaned[: -len(" section")].strip()

    if ref_type == "Line":
        cleaned = (cleaned.replace("–", "-").replace("—", "-")
                   .replace("to", "-").replace("and", ","))
        parts = re.findall(r"\d+(?:\s*-\s*\d+)?", cleaned)
        return [{"type": "Line", "label": p.replace(" ", "")} for p in parts]

    parts = re.findall(RANGE, cleaned.replace("–", "-").replace("—", "-"))
    if parts:
        out = []
        for p in parts:
            p = p.replace(" ", "")
            if ref_type in expand_ranges_for:
                out.extend({"type": ref_type, "label": e} for e in _expand_numeric_range(p))
            else:
                out.append({"type": ref_type, "label": p})
        return out

    if ref_type == "Section":
        return [{"type": "Section", "label": cleaned.lower()}]
    return [{"type": ref_type, "label": cleaned}]


def extract_and_normalize_references(sentence: str, expand_ranges_for=("Figure", "Table")):
    """Canonical gold extraction: detect references with the shared regex, normalize each."""
    out = []
    for m in find_reference_mentions(sentence):
        out.extend(normalize_reference(m.kind, m.text, expand_ranges_for))
    return out
