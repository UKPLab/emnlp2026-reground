import re
from typing import Dict, List,Iterable
from rapidfuzz import process, fuzz

# =========================
# Unhyphenation utilities
# =========================

# --- jam-robust normalization ---
_ALNUM_ONLY = re.compile(r"[^0-9a-z]+")
_HARD_HYPHEN = re.compile(r"-\s*$")  # ends with '-' (hyphenation at line break)
_SOFT_HYPHEN = "\u00ad"  # occasionally shows up from PDFs
_WS = re.compile(r"\s+")


def _name_key_raw(s: str) -> str:
    """Lowercased, collapsed whitespace (keeps spaces)."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _name_key_alnum(s: str) -> str:
    """Lowercased, letters+digits only (kills spaces/punct) — robust to jammed headers."""
    return _ALNUM_ONLY.sub("", (s or "").lower())


def _clean_soft_hyphens(s: str) -> str:
    # remove soft hyphens and collapse whitespace
    return _WS.sub(" ", s.replace(_SOFT_HYPHEN, "")).strip()


def _unhyphenate_concatenate(lines: Iterable[str]) -> str:
    """
    Merge lines into running text while undoing line-break hyphenation:
      "con-\ncept" -> "concept"
    Rule: if a line ends with '-', drop the hyphen and DO NOT insert a space.
          otherwise, insert a single space between lines.
    Also strips soft hyphens (U+00AD) and collapses whitespace.
    """
    pieces: List[str] = []
    lines = list(lines)
    for i, raw in enumerate(lines):
        line = _clean_soft_hyphens(raw.rstrip())
        if not line:
            continue
        if _HARD_HYPHEN.search(line) and i + 1 < len(lines):
            # remove trailing '-' and join next token without space
            pieces.append(line[:-1])
        else:
            pieces.append(line + " ")
    return "".join(pieces).strip()


# =========================
# Core line selection
# =========================


def _iter_lines_inclusive(sdr: dict, start_line: int, end_line: int) -> Dict[int, str]:
    """
    Return a Dict of {line_no: text} for lines in the inclusive range [start_line, end_line].
    """
    if start_line > end_line:
        start_line, end_line = end_line, start_line
    out = {
        str(ln): sdr["lines"][str(ln)]
        for ln in range(int(start_line), int(end_line) + 1)
        if str(ln) in sdr["lines"]
    }
    return out


def _collect_lines_from_numbers(sdr: dict, line_numbers: Iterable[int]):
    out = {ln: sdr["lines"][str(ln)] for ln in line_numbers if str(ln) in sdr["lines"]}
    return out


# =========================
# Public fetchers (concatenated text)
# =========================


def get_text_by_section_number(sdr: dict, section_number: str) -> str:
    """
    Concatenate text for the (one or many) sections whose 'number' equals section_number (string).
    Unhyphenates at joins. If multiple sections share the same number, they are concatenated in order.
    """
    section_number = str(section_number)
    spans = []
    for s in sdr.get("sections", []):
        if str(s.get("number")) == section_number:
            spans.append((s["start"], s["end"]))
    if not spans:
        return ""
    # collect lines across spans
    segs = []
    for a, b in sorted(spans):
        segs.extend(_iter_lines_inclusive(sdr, a, b).values())
    return _unhyphenate_concatenate(segs)


def get_text_by_section_name(sdr: dict, section_name: str, partial: bool = False, min_score: float = 0.82) -> str:
    """
    Retrieve text by section name, robust to jammed names (e.g. 'LITERATUREREVIEW').
    Matches by:
      - case-insensitive exact name
      - jam-robust alphanumeric form (no spaces/punct)
      - optional partial substring match
      - fallback fuzzy match (RapidFuzz) if nothing else matches
    """
    q_raw = _name_key_raw(section_name)
    q_aln = _name_key_alnum(section_name)

    spans = []
    fuzzy_pool = []

    for s in sdr.get("sections", []):
        name = (s.get("name") or "")
        start, end = int(s["start"]), int(s["end"])
        n_raw = _name_key_raw(name)
        n_aln = _name_key_alnum(name)

        # exact
        if n_raw == q_raw or n_aln == q_aln:
            spans.append((start, end))
            continue

        # partial substring
        if partial and (q_raw in n_raw or q_aln in n_aln):
            spans.append((start, end))
            continue

        fuzzy_pool.append((n_aln, (start, end)))

    # fuzzy fallback (if nothing matched)
    if not spans and fuzzy_pool and q_aln:
        choices = [k for (k, _) in fuzzy_pool]
        results = process.extract(
            q_aln,
            choices,
            scorer=fuzz.partial_ratio,
            score_cutoff=int(min_score * 100)
        )
        good_idxs = {idx for (_m, _score, idx) in results}
        spans = [fuzzy_pool[idx][1] for idx in good_idxs]

    # collect and unhyphenate
    if not spans:
        return ""
    segs = []
    for a, b in sorted(spans):
        segs.extend(_iter_lines_inclusive(sdr, a, b).values())
    return _unhyphenate_concatenate(segs)



def get_text_by_page(sdr: dict, page_index: int) -> str:
    """
    Concatenate text for a page (single page_index from sdr['pages']).
    """
    pg = sdr.get("pages", {}).get(int(page_index))
    if not pg:
        return ""
    lines = _iter_lines_inclusive(sdr, int(pg["start"]), int(pg["end"]))
    return _unhyphenate_concatenate(lines.values())

def get_text_by_line(sdr: dict, lines_query, concatenate: bool = True) -> str:
    """
    Concatenate text by line selection.
    Accepts:
      - int (single line)
      - (start, end) tuple for a range (inclusive)
      - list/iterable of ints for arbitrary sets
      - string (two ints separated by '-' or one int)
    """
    if isinstance(lines_query, int):
        lines = _collect_lines_from_numbers(sdr, [lines_query])
    elif isinstance(lines_query, str):
        if "-" in lines_query:
            parts = lines_query.split("-", 1)
            start, end = int(parts[0].strip()), int(parts[1].strip())
            lines = _iter_lines_inclusive(sdr, start, end)
        else:
            line_no = int(lines_query.strip())
            lines = _collect_lines_from_numbers(sdr, [line_no])
    elif isinstance(lines_query, tuple) and len(lines_query) == 2:
        lines = _iter_lines_inclusive(sdr, int(lines_query[0]), int(lines_query[1]))
    else:
        lines = _collect_lines_from_numbers(sdr, lines_query)

    if concatenate:
        return _unhyphenate_concatenate(lines.values())
    else:
        return lines
    
def get_caption(sdr: dict, ft_name: str) -> str:
    """
    Retrieve the caption text for a given figure or table number.
    """
    for caption_name, caption_text in sdr.get("captions", {}).items():
        if caption_name.strip().lower() == ft_name.strip().lower():
            return caption_text
    return ""

def get_entire_text(sdr: dict, all_all: bool = False) -> str:
    """
    Retrieve the entire text of the document, unhyphenated.
    """
    if all_all:
        lines = sdr.get("lines", {}).values()
        return _unhyphenate_concatenate(lines)
    else:
        get_last_page = max(int(pn) for pn in sdr.get("pages", {}).keys() if pn.isdigit())

        first_main_line = sdr["pages"]["1"]["start"]

        last_main_page = 8 if "8" in sdr.get("pages", {}) else get_last_page

        last_main_line = (
            sdr["pages"]["8"]["end"]
            if "8" in sdr.get("pages", {})
            else sdr["pages"][str(last_main_page)]["end"]
        )

        lines = [
            sdr["lines"][str(ln)]
            for ln in range(int(first_main_line), int(last_main_line) + 1)
            if str(ln) in sdr["lines"]
        ]

    return _unhyphenate_concatenate(lines)

def get_paragraph_by_id(sdr: dict, paragraph_id: str):
    """
    Retrieve paragraph by its ID.
    """
    for para in sdr.get("paragraphs", []):
        if str(para.get("paragraph_id")) == str(paragraph_id):
            return para
    return None


def get_section_text_by_number(sdr: dict, number: str):
    """
    Retrieve section text by its number.
    """

    sought_section = None

    for section in sdr.get("sections", []):
        if str(section.get("number")) == str(number):
            sought_section = section
            break

    if not sought_section:
        return None
    try:
        sec_para_start = sought_section["start_paragraph"]
        sec_para_end = sought_section["end_paragraph"]
    except KeyError:
        sec_para_start = None
        sec_para_end = None


    if sec_para_start is not None and sec_para_end is not None:
        segs = ""
        for para in sdr.get("paragraphs", []):
            pid = para.get("paragraph_id")
            if pid is not None and sec_para_start <= pid <= sec_para_end:
                segs += para["text"] + " "
        return segs.strip()
    
    section_text = get_text_by_line(
        sdr, (sought_section["start_line"], sought_section["end_line"])
    )
    return section_text

    
def get_paragraph_ids_by_line(
    sdr: dict,
    lines_query
) -> List[str]:
    """
    Return paragraph_ids that contain any of the specified line(s).

    Accepts:
      - int
      - (start, end) tuple
      - iterable of ints
      - string: "n" or "start-end"
    """

    # -------------------------
    # Normalize line numbers
    # -------------------------
    if isinstance(lines_query, int):
        target_lines = {lines_query}

    elif isinstance(lines_query, str):
        if "-" in lines_query:
            a, b = lines_query.split("-", 1)
            start, end = int(a.strip()), int(b.strip())
            target_lines = set(range(min(start, end), max(start, end) + 1))
        else:
            target_lines = {int(lines_query.strip())}

    elif isinstance(lines_query, tuple) and len(lines_query) == 2:
        start, end = int(lines_query[0]), int(lines_query[1])
        target_lines = set(range(min(start, end), max(start, end) + 1))

    else:
        target_lines = {int(ln) for ln in lines_query}

    # -------------------------
    # Match paragraphs
    # -------------------------
    matched_paragraph_ids = set()

    for para in sdr.get("paragraphs", []):
        pid = para.get("paragraph_id")
        if pid is None:
            continue

        # Expected paragraph line span
        try:
            p_start = int(para["start_line"])
            p_end = int(para["end_line"])
        except KeyError:
            continue  # no line info → cannot map reliably

        if p_start == -1 or p_end == -1:
            continue  # invalid line info

        # Intersection check
        if any(p_start <= ln <= p_end for ln in target_lines):
            matched_paragraph_ids.add(pid)

    return sorted(matched_paragraph_ids)
