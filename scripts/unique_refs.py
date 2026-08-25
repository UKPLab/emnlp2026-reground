import json
import os
import re
from utils import get_reference_patterns


REFERENCE_PATTERNS = get_reference_patterns()


def extract_normalized_refs(text, patterns):
    refs = set()

    for ref_type, pattern in patterns.items():
        for m in pattern.finditer(text):
            raw = m.group(0)

            if ref_type == "appendix":
                letter_match = re.search(
                    r"\bappendix\s+([A-Z])\b", raw, re.IGNORECASE
                )
                if letter_match:
                    refs.add(f"{ref_type}:{letter_match.group(1).upper()}")
                continue

            range_matches = re.findall(r"(\d+)\s*[-–]\s*(\d+)", raw)
            for a, b in range_matches:
                a, b = int(a), int(b)
                if 0 < (b - a) <= 20:
                    for x in range(a, b + 1):
                        refs.add(f"{ref_type}:{x}")

            tokens = re.findall(
                r"\d+(?:\.\d+)*[a-z]?(?:\([a-z]\))?",
                raw,
                flags=re.IGNORECASE,
            )

            for tok in tokens:
                tok = tok.lower().replace("(", "").replace(")", "")
                refs.add(f"{ref_type}:{tok}")

    return refs


def should_drop(rebuttal_sentence, review_span, patterns):
    if isinstance(review_span, list):
        review_span = " ".join(review_span)

    r_refs = extract_normalized_refs(rebuttal_sentence, patterns)
    v_refs = extract_normalized_refs(review_span, patterns)

    return bool(r_refs.intersection(v_refs))


def main(results_dir="../results"):
    with open(os.path.join(results_dir, "3_review_alignment.json"), encoding="utf-8") as f:
        aligned = json.load(f)

    kept = {}
    dropped_examples = []

    total = 0
    kept_count = 0
    dropped_count = 0

    for paper_id, paper_data in aligned.items():
        for comment_id, comment_data in paper_data.items():
            total += len(comment_data["aligned"])

            new_comment = {
                "review": comment_data["review"],
                "rebuttal": comment_data["rebuttal"],
                "aligned": [],
            }

            for pair in comment_data["aligned"]:
                if should_drop(
                    pair["rebuttal_sentence"],
                    pair["review_spans"],
                    REFERENCE_PATTERNS,
                ):
                    dropped_count += 1
                    dropped_examples.append(
                        {
                            "review_spans": pair["review_spans"],
                            "rebuttal_sentence": pair["rebuttal_sentence"],
                        }
                    )
                else:
                    new_pair = pair.copy()
                    new_pair.pop("labels", None)
                    new_comment["aligned"].append(new_pair)
                    kept_count += 1

            if new_comment["aligned"]:
                kept.setdefault(paper_id, {})[comment_id] = new_comment

    print(f"Total aligned references: {total}")
    print(f"Kept references: {kept_count}")
    print(
        f"Filtered out references: {dropped_count} "
        f"({dropped_count / total * 100:.2f}%)"
        if total > 0
        else "Filtered out references: 0"
    )
    print("=" * 40)

    with open(os.path.join(results_dir, "dropped_same_refs.json"), "w", encoding="utf-8") as f:
        json.dump(dropped_examples, f, indent=2, ensure_ascii=False)

    with open(os.path.join(results_dir, "4_unique_refs.json"), "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    main(ap.parse_args().results_dir)
