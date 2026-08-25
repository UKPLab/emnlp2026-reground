import json
import re
from pathlib import Path

from pypdf import PdfReader
from rapidfuzz import fuzz
from tqdm import tqdm


TRAILING_NUM_RE = re.compile(r"^(.*?)(-)?\s*(\d{3,4})$")

START_THRESHOLD = 90
CONTINUE_THRESHOLD = 90
MISS_LIMIT = 5
MIN_COVERAGE = 0.35


def normalize(text):
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,;:()\-]", "", text)
    return text.strip()


def find_introduction_start_line(sections):
    for sec in sections:
        if sec["number"] == "0" and sec["name"].lower() == "abstract":
            return sec["end_line"] + 1
    raise RuntimeError("Abstract section not found")


# =========================================================
# Metadata
# =========================================================
def load_meta(pdf_path):
    meta_path = pdf_path.parent / "meta.json"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        venue = meta.get("accepted_at")
        title = meta.get("title")
        venue = venue if isinstance(venue, str) and venue.strip() else None
        title = title if isinstance(title, str) and title.strip() else None
        return venue, title
    except Exception:
        return None, None


def load_caption_meta(pdf_path):
    meta_path = pdf_path.parent / "output_json"
    if not meta_path.exists():
        return None

    files = list(meta_path.glob("*.json"))
    if not files:
        return None

    try:
        with open(files[0], encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_captions(meta):
    captions = {}
    for item in meta:
        t = item.get("figType")
        n = item.get("name")
        c = item.get("caption", "")
        if t and n and c:
            captions[f"{t} {n}"] = c.strip()
    return dict(sorted(captions.items()))


# =========================================================
# PDF Line Extraction
# =========================================================
def extract_lines(pdf_path):
    reader = PdfReader(pdf_path)
    lines = {}

    for page in reader.pages:
        raw = page.extract_text() or ""
        for row in raw.splitlines():
            m = TRAILING_NUM_RE.match(row.rstrip())
            if not m:
                continue
            content = (m.group(1) + (m.group(2) or "")).rstrip()
            lines[int(m.group(3))] = content

    if not lines:
        raise RuntimeError("No line numbers detected")

    return dict(sorted(lines.items()))


# =========================================================
# Section Detection
# =========================================================
SECTION_HEADER_RX = re.compile(
    r"""
    ^\s*
    (?:
        (?P<num>\d{1,2}(?:\.\d{1,2})*)
        \s+
        (?P<name>[A-Z][^\n]{0,120})
      |
        Appendix\s+(?P<app>[A-Z])(?:\s+(?P<app_name>[A-Z][^\n]{0,120}))?
    )
    \s*$
    """,
    re.VERBOSE,
)

UNNUMBERED_SECTION_RE = [
    re.compile(p) for p in [
        r"(?i)^limitations?\s*$",
        r"(?i)^ethics\s*$",
        r"(?i)^ethical\s+considerations?\s*$",
        r"(?i)^ethics\s+statement\s*$",
        r"(?i)^ethical\s+statement\s*$",
        r"(?i)^broader\s+impact\s*$",
    ]
]


def detect_sections(lines):
    items = sorted(lines.items())
    doc_last = items[-1][0]

    headers = []
    last_header_idx = -1

    for idx, (ln, txt) in enumerate(items):
        m = SECTION_HEADER_RX.match(txt.strip())
        if not m:
            continue

        gd = m.groupdict()
        if gd.get("num"):
            num = gd["num"]
            name = (gd["name"] or "").strip()
            level = num.count(".") + 1
        else:
            app = gd["app"]
            num = f"Appendix {app}"
            name = (gd["app_name"] or num).strip()
            level = 1

        headers.append(
            {
                "number": num,
                "name": name,
                "level": level,
                "start_line": ln,
                "end_line": None,
            }
        )
        last_header_idx = idx

    for idx, (ln, txt) in enumerate(items):
        if idx <= last_header_idx:
            continue
        for rex in UNNUMBERED_SECTION_RE:
            if rex.fullmatch(txt.strip()):
                headers.append(
                    {
                        "number": txt.strip().lower().replace(" ", "_"),
                        "name": txt.strip(),
                        "level": 1,
                        "start_line": ln,
                        "end_line": None,
                    }
                )

    headers.sort(key=lambda h: h["start_line"])

    for i, h in enumerate(headers):
        end = doc_last
        for j in range(i + 1, len(headers)):
            if headers[j]["level"] <= h["level"]:
                end = headers[j]["start_line"] - 1
                break
        h["end_line"] = max(h["start_line"], end)

    out = []

    abs_start = next(
        (ln for ln, txt in items if txt.strip().lower() == "abstract"), None
    )
    if abs_start is not None:
        abs_end = next(
            (h["start_line"] - 1 for h in headers if h["start_line"] > abs_start),
            doc_last,
        )
        out.append(
            {
                "number": "0",
                "name": "Abstract",
                "level": 0,
                "start_line": abs_start,
                "end_line": abs_end,
            }
        )

    out.extend(headers)
    return out


def get_section_headers(sections):
    return [f"{s['number']} {s['name']}" for s in sections]


# =========================================================
# GROBID Paragraphs
# =========================================================
def load_grobid_paragraphs(pdf_path):
    with open(pdf_path.parent / "paragraphs.json", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for sec, plist in data["paper_clean"].items():
        out[sec] = plist
        if "limitation" in sec.lower():
            break

    return out


# =========================================================
# Alignment
# =========================================================
def align_paragraphs_to_lines(paragraphs, lines, section_headers, intro_start_line):
    line_numbers = sorted(lines)
    norm_lines = [normalize(lines[n]) for n in line_numbers]
    section_headers = set(normalize(h) for h in section_headers)

    results = []
    cursor = intro_start_line
    para_id = 1

    for section, plist in paragraphs.items():
        for para in plist:
            p = normalize(para)
            p_tokens = set(p.split())

            start_hits_required = 1 if len(p_tokens) < 30 else 2 if len(p_tokens) < 60 else 3

            start_idx = None
            hits = misses = 0
            seen_tokens = set()

            i = cursor
            while i < len(norm_lines):
                score = fuzz.partial_ratio(p, norm_lines[i])

                if start_idx is None:
                    if score >= START_THRESHOLD:
                        hits += 1
                        if hits == start_hits_required:
                            start_idx = i - (start_hits_required - 1)
                            seen_tokens |= set(norm_lines[start_idx].split())
                    else:
                        hits = 0
                else:
                    if score >= CONTINUE_THRESHOLD:
                        misses = 0
                        seen_tokens |= set(norm_lines[i].split())
                    else:
                        misses += 1
                        if misses >= MISS_LIMIT:
                            break
                i += 1

            if start_idx is None:
                results.append(
                    {
                        "paragraph_id": para_id,
                        "section": section,
                        "text": para,
                        "start_line": -1,
                        "end_line": -1,
                        "status": "filled",
                    }
                )
                para_id += 1
                continue

            end_idx = max(start_idx, i - misses)
            coverage = len(seen_tokens & p_tokens) / max(len(p_tokens), 1)

            if coverage < MIN_COVERAGE:
                results.append(
                    {
                        "paragraph_id": para_id,
                        "section": section,
                        "text": para,
                        "start_line": -1,
                        "end_line": -1,
                        "status": "weak_match",
                    }
                )
                para_id += 1
                continue

            results.append(
                {
                    "paragraph_id": para_id,
                    "section": section,
                    "text": para,
                    "start_line": line_numbers[start_idx],
                    "end_line": line_numbers[end_idx],
                    "status": "matched",
                }
            )

            cursor = end_idx + 1
            para_id += 1

    return results


def fill_gaps(paragraphs, max_gap_lines=10):
    for i, p in enumerate(paragraphs):
        if p["start_line"] != -1:
            continue

        prev = next((x for x in paragraphs[:i][::-1] if x["start_line"] != -1), None)
        nxt = next((x for x in paragraphs[i + 1 :] if x["start_line"] != -1), None)

        if not prev or not nxt:
            continue

        gap = nxt["start_line"] - prev["end_line"] - 1
        if 0 < gap <= max_gap_lines:
            p["start_line"] = prev["end_line"] + 1
            p["end_line"] = nxt["start_line"] - 1

    return paragraphs


def attach_paragraphs_to_sections(sections, paragraphs):
    for sec in sections:
        ps = [p for p in paragraphs if p["section"] == sec["name"]]
        if ps:
            sec["start_paragraph"] = ps[0]["paragraph_id"]
            sec["end_paragraph"] = ps[-1]["paragraph_id"]
            sec["end_line"] = ps[-1]["end_line"]
        else:
            sec["start_paragraph"] = None
            sec["end_paragraph"] = None


# =========================================================
# SDR Builder
# =========================================================
def build_sdr(pdf_path):
    venue, title = load_meta(pdf_path)
    lines = extract_lines(pdf_path)
    sections = detect_sections(lines)
    paragraphs = load_grobid_paragraphs(pdf_path)

    section_headers = get_section_headers(sections)
    intro_start = find_introduction_start_line(sections)

    paragraphs = align_paragraphs_to_lines(
        paragraphs, lines, section_headers, intro_start
    )
    paragraphs = fill_gaps(paragraphs)
    attach_paragraphs_to_sections(sections, paragraphs)

    caption_meta = load_caption_meta(pdf_path)
    captions = extract_captions(caption_meta) if caption_meta else {}

    return {
        "metadata": {
            "venue": venue,
            "title": title,
            "paper_id": pdf_path.name,
        },
        "lines": lines,
        "paragraphs": paragraphs,
        "sections": sections,
        "captions": captions,
    }


def main(dataset):
    failures = []

    for pdf_path in tqdm(dataset.glob("*/v1/paper.pdf"), desc=str(dataset)):
        try:
            sdr = build_sdr(pdf_path)
        except Exception as e:
            tqdm.write(f"[ERROR] {pdf_path.parent.parent.name[:20]} → {e}")
            failures.append(pdf_path.parent.parent.name)
            continue

        try:
            out = pdf_path.parent / "paper.sdr.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(sdr, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"[ERROR] {pdf_path.parent.parent.name[:20]} → {e}")
            failures.append(pdf_path.parent.parent.name)

    with open(f"{dataset}_parsing_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main(Path("../data/papers"))
