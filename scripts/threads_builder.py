import json
import os
from collections import defaultdict


def process_reviews(reviews):
    return [
        {
            "note_id": r["note_id"],
            "rid": r.get("rid"),
            "comment": r["report"],
        }
        for r in reviews
    ]


def build_threads(top_comments, replies):
    children_map = defaultdict(list)
    for reply in replies.values():
        children_map[reply["replyto"]].append(reply)

    def build_tree(note_id):
        return [
            {
                "note_id": child["note_id"],
                "from": child["rid"],
                "comment": child["comment"],
                "replies": build_tree(child["note_id"]),
            }
            for child in children_map.get(note_id, [])
        ]

    return [
        {
            "top_note_id": top["note_id"],
            "from": top["rid"],
            "comment": top["comment"],
            "replies": build_tree(top["note_id"]),
        }
        for top in top_comments
    ]


def get_threads(path):
    with open(os.path.join(path, "v1", "reviews.json"), encoding="utf-8") as f:
        reviews = json.load(f)
    with open(os.path.join(path, "v1", "comments.json"), encoding="utf-8") as f:
        comments = json.load(f)
    return build_threads(process_reviews(reviews), comments)


def flatten_threads_grouped(threads):
    grouped = []

    def walk(node, depth, parent, collector):
        entry = {
            "note_id": node["note_id"],
            "from": node["from"],
            "comment": node["comment"],
            "depth": depth,
            "replyto": parent,
        }
        collector.append(entry)
        for child in node.get("replies", []):
            walk(child, depth + 1, node["note_id"], collector)

    for top in threads:
        collector = [
            {
                "note_id": top["top_note_id"],
                "from": top["from"],
                "comment": top["comment"],
                "depth": 0,
                "replyto": None,
            }
        ]
        for reply in top["replies"]:
            walk(reply, 1, top["top_note_id"], collector)
        grouped.append(collector)

    return grouped


def group_list_to_dict(grouped):
    return {
        entry["note_id"]: entry
        for group in grouped
        for entry in group
    }


def get_subfolders_with_comments(data_path):
    result = []
    for sub in os.listdir(data_path):
        path = os.path.join(data_path, sub, "v1", "comments.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                if json.load(f):
                    result.append(sub)
    return result


def build_threads_dataset(data_path):
    """Build the flat {paper_id: {note_id: entry}} threads dataset for one
    directory of paper folders (e.g. one nlpeer venue)."""
    dataset = {}
    for sf in get_subfolders_with_comments(data_path):
        threads = get_threads(os.path.join(data_path, sf))
        grouped = flatten_threads_grouped(threads)
        dataset[sf] = group_list_to_dict(grouped)
    return dataset


def main(data_path="../data/papers", results_dir="../results"):
    os.makedirs(results_dir, exist_ok=True)
    dataset = build_threads_dataset(data_path)
    with open(os.path.join(results_dir, "threads.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="../data/papers",
                    help="directory of paper folders (each with v1/reviews.json, v1/comments.json)")
    ap.add_argument("--results-dir", default="../results")
    a = ap.parse_args()
    main(a.data_path, a.results_dir)
