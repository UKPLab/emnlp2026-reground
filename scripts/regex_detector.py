import json
import os
from bisect import bisect_right
from collections import defaultdict

from tqdm import tqdm

from utils import (
    get_reference_patterns,
    split_sentences_span_aware,
    find_reference_mentions,
)


REFERENCE_PATTERNS = get_reference_patterns()


def _sentence_index_for_pos(sent_spans, pos):
    if not sent_spans:
        return None
    starts = [s.start for s in sent_spans]
    i = bisect_right(starts, pos) - 1
    if i >= 0 and pos < sent_spans[i].end:
        return i
    return None


def _context_from_sentence_index(sent_spans, i, window_sentences):
    if i is None:
        return None
    start_i = max(0, i - window_sentences)
    end_i = min(len(sent_spans), i + window_sentences + 1)
    return " ".join(
        s.text.strip()
        for s in sent_spans[start_i:end_i]
        if s.text.strip()
    )


def detect_references(comment_dict, context_sentences=0):
    text = comment_dict.get("comment", "") or ""
    if not text.strip():
        return []

    sent_spans = split_sentences_span_aware(
        text, ref_patterns=REFERENCE_PATTERNS
    )
    mentions = find_reference_mentions(
        text, ref_patterns=REFERENCE_PATTERNS
    )

    refs = []
    for m in mentions:
        i = _sentence_index_for_pos(sent_spans, m.start)
        context = _context_from_sentence_index(
            sent_spans, i, context_sentences
        )
        if context is None:
            continue
        if len(context.split()) <= 6:
            continue
        refs.append(
            {
                "reference_type": m.kind,
                "exact_match": m.text,
                "context": context,
            }
        )

    seen = set()
    unique_refs = []
    for r in refs:
        key = (r["reference_type"], r["exact_match"], r["context"])
        if key not in seen:
            unique_refs.append(r)
            seen.add(key)

    return unique_refs


def group_by_context(references):
    grouped = defaultdict(list)
    for ref in references:
        grouped[ref["context"]].append(
            {
                "reference_type": ref["reference_type"],
                "exact_match": ref["exact_match"],
            }
        )

    return [
        {"context": context, "references": refs}
        for context, refs in grouped.items()
    ]


def group_sents(ref_sents):
    output = {}
    for sent in ref_sents:
        ctx = sent["context"]
        output.setdefault(ctx, {})
        for ref in sent["references"]:
            output[ctx].setdefault(
                ref["reference_type"], []
            ).append(ref["exact_match"])
    return output


def detect_and_group_references(comment_dict, context_sentences=0):
    refs = detect_references(
        comment_dict, context_sentences=context_sentences
    )
    grouped = group_by_context(refs)
    return {
        "comment": comment_dict.get("comment"),
        "contains_references": bool(grouped),
        "referencing_sentences": group_sents(grouped),
    }


def main(results_dir="../results"):
    with open(os.path.join(results_dir, "threads.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    referencing_comments = {}
    total_referencing = 0
    total_comments = 0
    total_regexes = 0

    for paper_id, threads in tqdm(dataset.items()):
        for note_id, note in threads.items():
            if note.get("from") != "Authors":
                continue

            total_comments += 1
            enriched = detect_and_group_references(note)

            if enriched["contains_references"]:
                referencing_comments.setdefault(
                    paper_id, {}
                )[note_id] = enriched
                total_referencing += 1
                total_regexes += sum(
                    len(matches)
                    for ctx in enriched["referencing_sentences"].values()
                    for matches in ctx.values()
                )

    with open(
        os.path.join(results_dir, "1_regex_found.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(referencing_comments, f, indent=2, ensure_ascii=False)

    if total_comments == 0:
        print("No author comments found.")
        return

    print(
        f"Found {total_referencing} referencing author comments out of "
        f"{total_comments} total comments "
        f"({total_referencing / total_comments * 100:.2f}%)"
    )
    print(
        f"Total number of regex references found: {total_regexes}"
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    main(ap.parse_args().results_dir)
