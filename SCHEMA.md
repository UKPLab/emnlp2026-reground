# Data format

ReGround consists of two parts:

- `reground.json`: the grounding annotations (reviewer comment ↔ rebuttal sentence pairs).
- `nlpeer/`: the papers, reviews and rebuttals they come from (NLPeer v2), plus the parsed paper
  structure and extracted figure/table images that the retrieval tasks rank.

Both are joined by `venue` and `paper_id`.

## `reground.json`

A nested JSON object: venue → paper → rebuttal → aligned pairs.

```json
{
  "<venue>": {
    "<paper_id>": {
      "<rebuttal_id>": {
        "review": "...",
        "rebuttal": "...",
        "aligned": [
          {
            "rebuttal_sentence": "...",
            "review_spans": "..."
          }
        ]
      }
    }
  }
}
```

| Key / field | Description |
|---|---|
| `<venue>` | One of `EMNLP24`, `NAACL25`, `ACL25`, `COLING25`, `EMNLP25`. |
| `<paper_id>` | The NLPeer paper id; the paper's folder is `nlpeer/<venue>/<paper_id>/`. |
| `<rebuttal_id>` | The `note_id` of the author comment in `comments.json` that this entry is built from. |
| `review` | Full text of the reviewer turn the rebuttal replies to (usually the review report, sometimes a later reviewer comment). |
| `rebuttal` | Full text of the author rebuttal. |
| `aligned` | Pairs of a reviewer comment and a rebuttal sentence that addresses it. |
| `aligned[].review_spans` | The reviewer comment: a span of `review`. This is the **query** in the retrieval tasks. |
| `aligned[].rebuttal_sentence` | A sentence of `rebuttal` that answers the comment and refers explicitly to the submission (e.g. "see Table 3", "Section 4.2", "lines 128–131"). |

One review span can appear in several pairs when the authors answer it with several sentences.

Release contents: 3,656 papers, 6,300 rebuttals and 12,234 aligned pairs. This is a cleaned
version of the data used in the paper, with some incorrect pairs removed, so statistics computed
from it (`scripts/analyze_evidence_statistics.py`) differ slightly from the paper's Table 2.

### Gold evidence

The gold evidence is not stored as a separate field. It is derived from `rebuttal_sentence` with
`utils.extract_and_normalize_references`, which all retrieval runners and evaluators share:

```python
from utils import extract_and_normalize_references
extract_and_normalize_references("As shown in Table 2 and Figs. 4-5 (see also §3.1) ...")
# [{"type": "Table", "label": "2"}, {"type": "Figure", "label": "4"},
#  {"type": "Figure", "label": "5"}, {"type": "Section", "label": "3.1"}]
```

Reference types: `Section`, `Line`, `Table`, `Figure`, `Appendix`, `Equation`, `Page`,
`Footnote`. Figure and table ranges are expanded (`Figs. 4-5` → 4, 5). Line references keep
their ranges (`128-131`) and are resolved to the paragraphs that contain them.

## `nlpeer/`

One folder per paper: `nlpeer/<venue>/<paper_id>/v1/`.

| File | Origin | Contents |
|---|---|---|
| `meta.json` | NLPeer | Title, authors, abstract, license, decision (`accepted_at` / `not_accepted_at`). |
| `reviews.json` | NLPeer | List of reviews: `note_id`, `report` (structured review text), `scores`, `meta`. |
| `comments.json` | NLPeer | Discussion thread: `note_id`, `rid` (author of the note, `Authors` for rebuttals), `replyto`, `comment`. |
| `metareview.json` | NLPeer | Meta-review, if available. |
| `paper.pdf` | NLPeer | The anonymous submission, with line numbers. |
| `paper_clean.pdf` | added | `paper.pdf` with the line-number ruler removed (`cleanpdf.py`), used as GROBID input. |
| `paragraphs.json` | added | GROBID paragraphs grouped by section heading (`run_grob.py`). |
| `paper.sdr.json` | added | Structured paper: the retrieval candidate pool (see below; `parse_papers.py`). |
| `images/` | added | Figure and table crops from PDFFigures 2.0: `Figure_<n>_wc.png` / `Table_<n>_wc.png` include the caption, `_nc.png` exclude it. |
| `output_json/meta_paper.json` | added | Raw PDFFigures 2.0 output: caption, page and bounding boxes per figure/table. |

### `paper.sdr.json`

```json
{
  "metadata":   { "venue": "coling25", "title": "...", "paper_id": "..." },
  "lines":      { "1": "first line of the PDF text", "2": "...", "...": "..." },
  "paragraphs": [ { "paragraph_id": 1, "section": "...", "text": "...",
                    "start_line": 30, "end_line": 43, "status": "matched" } ],
  "sections":   [ { "number": "3.1", "name": "...", "level": 1,
                    "start_line": 120, "end_line": 160,
                    "start_paragraph": 12, "end_paragraph": 15 } ],
  "captions":   { "Figure 1": "Figure 1: ...", "Table 2": "Table 2: ..." }
}
```

- `lines` are keyed by the submission's line numbers, so line references in rebuttals can be
  resolved directly.
- `paragraphs[].status` records how a GROBID paragraph was located in the line-numbered text:
  `matched` (exact line span), `weak_match` (fuzzy), or `filled` (not located; `start_line` and
  `end_line` are `-1`).
- `sections` include the abstract as `number: "0"`. Unnumbered sections get a snake_case
  `number` derived from their heading (e.g. `limitations`, `ethics_statement`).
- Text retrieval candidates are lines, paragraphs, sections and captions, with ids
  `Line <n>`, `Paragraph <id>`, `Section <number> / <name>` and the caption key (`Table 2`).

## Retrieval outputs

Runners write one file per model to `<out-root>/<venue>/text_retrieval/` or
`<out-root>/<venue>/image_retrieval/`. Each file maps paper id → query index → the query, its gold
references and the model's ranking:

```json
{ "<paper_id>": { "0": { "query": "...", "gold": [{"type": "Table", "label": "2"}],
                         "scores": { "Table 2": 0.91, "Paragraph 14": 0.87, "...": 0.0 } } } }
```

`scores` maps each candidate to its score, ordered best first.

The `evaluate_*` scripts read these files.

## License

The content derives from NLPeer v2 and is released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), like NLPeer. The code in
this repository is under Apache 2.0.
