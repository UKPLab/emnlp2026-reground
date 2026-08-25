import json
import os
from tqdm import tqdm
from rapidfuzz import fuzz
from utils import safe_sentence_split


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def is_fuzzy_quote(ref_sentence: str, review_sents: list[str]) -> bool:
    if not ref_sentence:
        return False

    if ref_sentence.lstrip().startswith(">"):
        return True
    if ref_sentence.startswith("**") and ref_sentence.endswith("**"):
        return True
    if ref_sentence.lstrip().startswith("#"):
        return True

    ref_n = _norm(ref_sentence)
    if not ref_n:
        return False

    for rs in review_sents:
        if not rs:
            continue
        rs_n = _norm(rs)
        if not rs_n:
            continue

        if ref_n in rs_n or rs_n in ref_n:
            return True
        if fuzz.partial_ratio(ref_n, rs_n) >= 85:
            return True
        if fuzz.token_set_ratio(ref_n, rs_n) >= 80:
            return True

    return False


def find_parent_reviewer(note_id, comments):
    current = comments.get(note_id)
    if not current:
        return None

    visited = set()
    while current and current.get("replyto"):
        parent = comments.get(current["replyto"])
        if not parent:
            break

        parent_id = parent.get("note_id", current.get("replyto"))
        if parent_id in visited:
            break
        visited.add(parent_id)

        if parent.get("from") != current.get("from"):
            return parent

        current = parent

    return None


def get_review_text(review):
    if isinstance(review, str):
        return review

    if isinstance(review, dict):
        parts = []
        for k, v in review.items():
            if isinstance(v, str):
                parts.append(f"{k}:\n{v}")
        return "\n\n".join(parts)

    return ""


def main(results_dir="../results"):
    with open(os.path.join(results_dir, "1_regex_found.json"), encoding="utf-8") as f:
        dataset_regex_found = json.load(f)

    with open(os.path.join(results_dir, "threads.json"), encoding="utf-8") as f:
        dataset = json.load(f)

    no_quotes = {}
    total = 0
    kept = 0

    for paper_id, data in tqdm(dataset_regex_found.items(), desc="Processing..."):
        no_quotes[paper_id] = {}

        for comment_id, comment_data in data.items():
            no_quotes[paper_id][comment_id] = {"referencing_sentences": []}

            review = find_parent_reviewer(comment_id, dataset.get(paper_id, {}))
            if review is None:
                print(
                    f"[WARN] No parent review for paper {paper_id}, comment {comment_id}"
                )
                continue

            review_text = get_review_text(review.get("comment", ""))
            rebuttal_text = dataset[paper_id][comment_id].get("comment", "")
            review_sents = safe_sentence_split(review_text)

            for ref_sentence in comment_data.get("referencing_sentences", {}).keys():
                total += 1
                if not is_fuzzy_quote(ref_sentence, review_sents):
                    kept += 1
                    no_quotes[paper_id][comment_id]["referencing_sentences"].append(
                        ref_sentence
                    )

            no_quotes[paper_id][comment_id]["rebuttal"] = rebuttal_text
            no_quotes[paper_id][comment_id]["review"] = review_text

        no_quotes[paper_id] = {
            cid: cdata
            for cid, cdata in no_quotes[paper_id].items()
            if cdata["referencing_sentences"]
        }

    no_quotes = {pid: comments for pid, comments in no_quotes.items() if comments}

    removed = total - kept
    if total == 0:
        print("Total referencing sentences: 0 (nothing to filter)")
    else:
        print(f"Total referencing sentences: {total} after quote filtering: {kept}")
        print(f"Quote filtering removed {removed} sentences ({removed/total*100:.2f}%)")

    with open(os.path.join(results_dir, "2_regex_no_quotes.json"), "w", encoding="utf-8") as f:
        json.dump(no_quotes, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="../results")
    main(ap.parse_args().results_dir)
