# run_image_rankers_acl.py
import argparse
import gc
import json
import os
import re
from typing import Dict, List, Any

import torch
from tqdm import tqdm

import image_methods as ir
from utils import extract_and_normalize_references  # canonical shared gold extraction


NLPEER_ROOT = os.environ.get("REGROUND_NLPEER", "nlpeer")
# figure/table images extracted by PDFFigures 2.0, shipped in the released nlpeer tree
IMAGES_PATH_DEFAULT = NLPEER_ROOT + "/{dataset_name}/{paper_id}/v1/images/"


def enrich(dataset: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    enriched: Dict[str, List[Dict[str, Any]]] = {}
    empty = 0
    for paper_id, paper_data in dataset.items():
        enriched[paper_id] = []
        for _, review_data in paper_data.items():
            if review_data.get("aligned") is None:
                continue
            for ref in review_data["aligned"]:
                ref["refs"] = extract_and_normalize_references(ref["rebuttal_sentence"])
                enriched[paper_id].append(ref)
        if len(enriched[paper_id]) == 0:
            empty += 1
    if empty > 0:
        tqdm.write(f"[INFO] Papers with no extracted references: {empty} / {len(dataset)}")
    return enriched


def select_figures_and_tables(
    enriched: Dict[str, List[Dict[str, Any]]], dataset_name: str, images_path_tmpl: str
):
    filtered = {}
    for paper_id, rebuttals in enriched.items():
        filtered_rebuttals = []
        for entry in rebuttals:
            gold = entry.get("refs", [])
            has_fig_or_tab = any(
                (ref.get("type") in ("Figure", "Table")) for ref in gold
            )
            if has_fig_or_tab:
                filtered_rebuttals.append(entry)

        filtered[paper_id] = {
            "rebuttals": filtered_rebuttals,
            "images_folder": images_path_tmpl.format(
                dataset_name=dataset_name, paper_id=paper_id
            ),
        }
    return filtered


def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_one_model(
    model_name: str,
    dataset_name: str,
    dataset: Dict[str, Any],
    out_dir: str,
    images_path_tmpl: str,
    cache_dir: str,
    image_batch_size: int,
    vlm_image_batch_size: int,
    save_every: int,
):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{model_name}.json")

    enriched = enrich(dataset)

    # limit enriched to only 3 papers for quick testing
    # enriched = dict(list(enriched.items())[:3])


    filtered = select_figures_and_tables(enriched, dataset_name, images_path_tmpl)

    # Build ranker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ir.RankerConfig(
        cache_dir=cache_dir,
        image_batch_size=image_batch_size,
        vlm_image_batch_size=vlm_image_batch_size,
    )
    rank_fn = ir.build_ranker(model_name, device, cfg)

    outputs: Dict[str, Dict[str, Any]] = {}
    processed = 0

    for paper_id, info in tqdm(filtered.items(), desc=f"Ranking [{dataset_name}] with [{model_name}]"):
        rebuttals = info["rebuttals"]
        folder = info["images_folder"]

        if not rebuttals:
            continue

        image_paths = ir.list_images(folder)
        if not image_paths:
            continue
        # IMPORTANT: stable order
        image_paths = sorted(image_paths)

        # Load images once per paper, reuse for all queries
        try:
            pil_images = ir.load_images(image_paths)
        except Exception as e:
            tqdm.write(
                f"[WARN] Failed to load images for paper={paper_id} folder={folder}: {e}"
            )
            continue

        paper_out: Dict[str, Any] = {}

        for q_idx, entry in enumerate(rebuttals):
            query = entry["review_spans"]
            ranking = rank_fn(query, image_paths, pil_images)
            scores = {os.path.basename(p).split(".")[0]: float(s) for p, s in ranking}

            paper_out[str(q_idx)] = {
                "query": query,
                "gold": entry.get("refs", []),
                "scores": scores,
            }

        outputs[paper_id] = paper_out
        processed += 1

        if save_every > 0 and processed % save_every == 0:
            atomic_write_json(out_path, outputs)
        
        # free per-paper images
        del pil_images
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    atomic_write_json(out_path, outputs)

    # free model memory
    del rank_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model keys, e.g. siglip2 openclip_l14 ... (see image_methods.RANKER_FACTORIES)",
    )
    parser.add_argument("--dataset", type=str, default="reground.json")
    parser.add_argument("--venues", nargs="+", default=None, help="default: all venues in the dataset")
    parser.add_argument("--images_path_tmpl", type=str, default=IMAGES_PATH_DEFAULT)
    parser.add_argument("--out-root", type=str, default="results")
    parser.add_argument("--cache_dir", type=str, default="./emb_cache")
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--vlm_image_batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    reground = json.load(open(args.dataset, "r", encoding="utf-8"))
    venues = args.venues or list(reground.keys())

    tqdm.write(f"[INFO] Models:   {args.models}")
    for dataset_name in venues:
        dataset = reground[dataset_name]
        out_dir = os.path.join(args.out_root, dataset_name, "image_retrieval")
        tqdm.write(f"[INFO] Processing {dataset_name}")
        for m in args.models:
            run_one_model(
                model_name=m,
                dataset_name=dataset_name,
                dataset=dataset,
                out_dir=out_dir,
                images_path_tmpl=args.images_path_tmpl,
                cache_dir=args.cache_dir,
                image_batch_size=args.image_batch_size,
                vlm_image_batch_size=args.vlm_image_batch_size,
                save_every=args.save_every,
            )

if __name__ == "__main__":
    main()


# Usage example:
# python image_compute_metrics.py --models siglip2 openclip_l14 eva02_clip_l14 jina_embeddings_v4 colpali_v1_3 colqwen2_5_v0_2 qwen3_vl_8b_yesno qwen3_vl_32b_yesno 
# python image_compute_metrics.py --models qwen3_vl_8b_yesno qwen3_vl_32b_yesno 

# python image_compute_metrics.py --models siglip2 openclip_l14 eva02_clip_l14 > ./logs/dual_encoders.log 2>&1
# python image_compute_metrics.py --models jina_embeddings_v4 > ./logs/jina.log 2>&1
# python image_compute_metrics.py --models colpali_v1_3 colqwen2_5_v0_2 > ./logs/colis.log 2>&1
# python image_compute_metrics.py --models qwen3_vl_8b_yesno > ./logs/qwen8.log 2>&1


# python image_compute_metrics.py --models qwen3_vl_32b_yesno > ./logs/qwen32.log 2>&1