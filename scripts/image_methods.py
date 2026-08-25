# image_ranking_acl.py
import glob
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import open_clip
from transformers import (
    AutoModel,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from transformers import ColPaliForRetrieval, ColPaliProcessor


# -------------------------
# Helpers
# -------------------------

SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")


def list_images(image_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in SUPPORTED_EXTS:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(set(paths))
    return paths


def load_images(image_paths: List[str]) -> List[Image.Image]:
    # Return a list aligned with image_paths
    imgs: List[Image.Image] = []
    for p in image_paths:
        with Image.open(p) as im:
            imgs.append(im.convert("RGB"))
    return imgs


def compute_cache_key(image_paths: List[str]) -> str:
    """
    IMPORTANT: cache key must be collision-resistant across papers.
    We hash the full sorted paths to avoid basename collisions (figure1.png repeats everywhere).
    """
    joined = "\n".join(image_paths)  # already sorted by caller
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [to_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    return obj


def load_cache(cache_dir: str, model_key: str, cache_key: str, image_paths: List[str]):
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{model_key}__{cache_key}.pt")
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location="cpu")
    if data.get("image_paths") != list(image_paths):
        return None
    return data.get("image_embeds")


def save_cache(
    cache_dir: str,
    model_key: str,
    cache_key: str,
    image_paths: List[str],
    image_embeds: Any,
) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{model_key}__{cache_key}.pt")
    tmp = path + ".tmp"
    data = {"image_paths": list(image_paths), "image_embeds": to_cpu(image_embeds)}
    torch.save(data, tmp)
    os.replace(tmp, path)


def ranking_from_scores(
    image_paths: List[str], scores: torch.Tensor
) -> List[Tuple[str, float]]:
    scores_list = scores.detach().cpu().tolist()
    return sorted(zip(image_paths, scores_list), key=lambda x: x[1], reverse=True)


def _batched(iterable, batch_size: int):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def _apply_chat_template(processor: Any, conversation: List[Dict[str, Any]]) -> str:
    # Some versions expose apply_chat_template on processor, others on processor.tokenizer.
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    if hasattr(processor, "tokenizer") and hasattr(
        processor.tokenizer, "apply_chat_template"
    ):
        return processor.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True
        )
    raise AttributeError(
        "Processor/tokenizer has no apply_chat_template(). Update transformers/Qwen processor."
    )


def _infer_input_device(model: torch.nn.Module) -> torch.device:
    """
    For sharded HF models, we want inputs on the first CUDA device in hf_device_map.
    If not sharded, use the first parameter's device.
    """
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict):
        cuda_devs: List[int] = []
        for v in hf_map.values():
            if isinstance(v, str) and v.startswith("cuda:"):
                try:
                    cuda_devs.append(int(v.split(":")[1]))
                except Exception:
                    pass
        if cuda_devs:
            return torch.device(f"cuda:{min(cuda_devs)}")
    # fallback
    return next(model.parameters()).device


def _find_single_token_id(tokenizer: Any, candidates: List[str]) -> Optional[int]:
    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    return None


# -------------------------
# Ranker configs
# -------------------------


@dataclass
class RankerConfig:
    cache_dir: str = "./emb_cache"
    image_batch_size: int = 32  # for dual-encoder image encoding
    vlm_image_batch_size: int = 4  # VLM forward can be heavy; chunk images per query
    text_max_length: int = 64  # for SigLIP2 text processing


# -------------------------
# Ranker factories
# Each ranker returns: rank(query, image_paths, pil_images)->[(path, score)]
# -------------------------


def make_openclip_l14_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "openclip_vitl14"
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Cache in-memory by exact image_paths tuple (collision-proof)
    in_memory_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            image_embeds = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if image_embeds is None:
                embs: List[torch.Tensor] = []
                for batch_imgs in _batched(pil_images, cfg.image_batch_size):
                    batch = torch.stack([preprocess(im) for im in batch_imgs]).to(
                        device
                    )
                    e = model.encode_image(batch)
                    embs.append(e)
                image_embeds = torch.cat(embs, dim=0)
                image_embeds = F.normalize(image_embeds, dim=-1)
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = image_embeds.to(device)
            in_memory_cache[key] = image_embeds

        text_tokens = tokenizer([query]).to(device)
        text_embed = model.encode_text(text_tokens)
        text_embed = F.normalize(text_embed, dim=-1)

        scores = (image_embeds @ text_embed.T).squeeze(-1)
        return ranking_from_scores(image_paths, scores)

    return rank


def make_eva02_clip_l14_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "eva02_l14"
    model_name = "EVA02-L-14"
    pretrained = "merged2b_s4b_b131k"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=pretrained,
    )
    model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(
        "ViT-L-14"
    )  # standard CLIP tokenizer works here

    in_memory_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            image_embeds = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if image_embeds is None:
                embs: List[torch.Tensor] = []
                for batch_imgs in _batched(pil_images, cfg.image_batch_size):
                    batch = torch.stack([preprocess(im) for im in batch_imgs]).to(
                        device
                    )
                    e = model.encode_image(batch)
                    embs.append(e)
                image_embeds = torch.cat(embs, dim=0)
                image_embeds = F.normalize(image_embeds, dim=-1)
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = image_embeds.to(device)
            in_memory_cache[key] = image_embeds

        text_tokens = tokenizer([query]).to(device)
        text_embed = model.encode_text(text_tokens)
        text_embed = F.normalize(text_embed, dim=-1)

        scores = (image_embeds @ text_embed.T).squeeze(-1)
        return ranking_from_scores(image_paths, scores)

    return rank


def make_siglip2_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "siglip2"
    model_id = "google/siglip2-so400m-patch16-512"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model.to(device).eval()

    in_memory_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            image_embeds = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if image_embeds is None:
                embs: List[torch.Tensor] = []
                for batch_imgs in _batched(pil_images, cfg.image_batch_size):
                    inputs = processor(images=batch_imgs, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    e = model.get_image_features(**inputs)
                    embs.append(e)
                image_embeds = torch.cat(embs, dim=0)
                image_embeds = F.normalize(image_embeds, dim=-1)
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = image_embeds.to(device)
            in_memory_cache[key] = image_embeds

        text_inputs = processor(
            text=[query],
            padding="max_length",
            max_length=cfg.text_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_embed = model.get_text_features(**text_inputs)
        text_embed = F.normalize(text_embed, dim=-1)

        scores = (image_embeds @ text_embed.T).squeeze(-1)
        return ranking_from_scores(image_paths, scores)

    return rank


def make_jina_embeddings_v4_ranker(device: torch.device, cfg: RankerConfig):
    import numpy as np

    model_key = "jina_embeddings_v4"
    model_id = "jinaai/jina-embeddings-v4"
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.to(device).eval()

    in_memory_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _to_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Empty embeddings from jina-embeddings-v4.")
            first = x[0]
            if isinstance(first, torch.Tensor):
                return torch.stack([t.to(device) for t in x], dim=0)
            if isinstance(first, np.ndarray):
                return torch.from_numpy(np.stack(x, axis=0)).to(device)
            return torch.tensor(x, device=device)
        return torch.tensor(x, device=device)

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            cached = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if cached is None:
                embs: List[torch.Tensor] = []
                for batch_imgs in _batched(pil_images, cfg.image_batch_size):
                    raw = model.encode_image(images=batch_imgs, task="retrieval")
                    e = _to_tensor(raw)
                    embs.append(e)
                image_embeds = torch.cat(embs, dim=0)
                image_embeds = F.normalize(image_embeds, dim=-1)
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = _to_tensor(cached)
            in_memory_cache[key] = image_embeds

        raw_text = model.encode_text(
            texts=[query], task="retrieval", prompt_name="query"
        )
        text_embeds = _to_tensor(raw_text)
        text_embeds = F.normalize(text_embeds, dim=-1)  # [1, D]

        scores = (image_embeds @ text_embeds.T).squeeze(-1)
        return ranking_from_scores(image_paths, scores)

    return rank


def make_colpali_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "colpali_v1_3"
    model_id = "vidore/colpali-v1.3-hf"
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    model = ColPaliForRetrieval.from_pretrained(model_id, dtype=dtype)
    model.to(device).eval()
    processor = ColPaliProcessor.from_pretrained(model_id, use_fast=True)

    in_memory_cache: Dict[Tuple[str, ...], Any] = {}

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            cached = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if cached is None:
                inputs_images = processor.process_images(
                    images=pil_images, return_tensors="pt"
                )
                inputs_images = inputs_images.to(device)
                image_embeds = model(**inputs_images).embeddings
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = (
                    cached.to(device)
                    if isinstance(cached, torch.Tensor)
                    else torch.as_tensor(cached, device=device)
                )
            in_memory_cache[key] = image_embeds

        inputs_text = processor.process_queries(text=[query], return_tensors="pt")
        inputs_text = inputs_text.to(device)
        query_embeds = model(**inputs_text).embeddings

        scores = processor.score_retrieval(
            query_embeddings=query_embeds, passage_embeddings=image_embeds
        )[0]
        return ranking_from_scores(image_paths, scores)

    return rank


def make_colqwen_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "colqwen2_5_v0_2"
    model_id = "vidore/colqwen2.5-v0.2"
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    # Some versions use dtype=..., some dtype=...
    try:
        model = ColQwen2_5.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = ColQwen2_5.from_pretrained(model_id, dtype=dtype)

    model.to(device).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_id, use_fast=True)

    in_memory_cache: Dict[Tuple[str, ...], Any] = {}

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        key = tuple(image_paths)
        cache_key = compute_cache_key(image_paths)

        image_embeds = in_memory_cache.get(key)
        if image_embeds is None:
            cached = load_cache(cfg.cache_dir, model_key, cache_key, image_paths)
            if cached is None:
                batch_images = processor.process_images(pil_images).to(device)
                image_embeds = model(**batch_images)
                save_cache(
                    cfg.cache_dir, model_key, cache_key, image_paths, image_embeds
                )
            else:
                image_embeds = (
                    cached.to(device)
                    if isinstance(cached, torch.Tensor)
                    else torch.as_tensor(cached, device=device)
                )
            in_memory_cache[key] = image_embeds

        batch_queries = processor.process_queries([query]).to(device)
        query_embeds = model(**batch_queries)

        scores = processor.score_multi_vector(query_embeds, image_embeds)[0]
        return ranking_from_scores(image_paths, scores)

    return rank


def _make_qwen_vlm_yesno_logit_ranker(
    model_key: str,
    model_cls: Any,
    model_id: str,
    device_map: str,
    cfg: RankerConfig,
):
    """
    VLM scorer: relevance = P(yes) from next-token distribution after the chat prompt.
    This avoids generate() and is batchable.
    """
    model = model_cls.from_pretrained(
        model_id,
        dtype="auto",
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    input_device = _infer_input_device(model)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    yes_id = _find_single_token_id(tokenizer, ["Yes", " yes", "yes"])
    no_id = _find_single_token_id(tokenizer, ["No", " no", "no"])
    if yes_id is None or no_id is None:
        raise RuntimeError(
            f"[{model_key}] Could not find single-token IDs for Yes/No. "
            "Update candidates in _find_single_token_id() for your tokenizer."
        )

    @torch.inference_mode()
    def score_batch(query: str, batch_imgs: List[Image.Image]) -> torch.Tensor:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are evaluating a figure or table from an ML/NLP paper.\n"
                            f'Query: "{query}"\n\n'
                            "Is this figure/table relevant to the query?\n"
                            "Answer only: Yes or No."
                        ),
                    },
                ],
            }
        ]
        prompt = _apply_chat_template(processor, conversation)

        inputs = processor(
            text=[prompt] * len(batch_imgs),
            images=batch_imgs,
            padding=True,
            return_tensors="pt",
        )
        # Put inputs on the first device (works for sharded models too)
        inputs = inputs.to(input_device)

        out = model(**inputs, use_cache=False)
        logits = out.logits  # [B, L, V]

        attn = inputs.get("attention_mask", None)
        if attn is None:
            # fallback: assume no padding
            idx = torch.full(
                (logits.size(0),),
                logits.size(1) - 1,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            idx = attn.sum(dim=1) - 1  # last non-pad token index

        next_logits = logits[torch.arange(logits.size(0), device=logits.device), idx, :]
        probs = F.softmax(next_logits[:, [no_id, yes_id]], dim=-1)
        return probs[:, 1].detach()  # P(Yes)

    @torch.inference_mode()
    def rank(query: str, image_paths: List[str], pil_images: List[Image.Image]):
        scores: List[torch.Tensor] = []
        for batch_imgs in _batched(pil_images, cfg.vlm_image_batch_size):
            scores.append(score_batch(query, batch_imgs))
        scores_t = torch.cat(scores, dim=0).to(torch.float32)
        return ranking_from_scores(image_paths, scores_t)

    return rank


def make_qwen3_vl_8b_ranker(device: torch.device, cfg: RankerConfig):
    # 'device' is unused because we rely on HF device_map for large models
    model_key = "qwen3_vl_8b_yesno"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    return _make_qwen_vlm_yesno_logit_ranker(
        model_key=model_key,
        model_cls=Qwen3VLForConditionalGeneration,
        model_id=model_id,
        device_map="auto",
        cfg=cfg,
    )


def make_qwen3_vl_32b_ranker(device: torch.device, cfg: RankerConfig):
    model_key = "qwen3_vl_32b_yesno"
    model_id = "Qwen/Qwen3-VL-32B-Instruct"
    return _make_qwen_vlm_yesno_logit_ranker(
        model_key=model_key,
        model_cls=Qwen3VLForConditionalGeneration,
        model_id=model_id,
        device_map="auto",
        cfg=cfg,
    )


# -------------------------
# Registry
# -------------------------

RANKER_FACTORIES: Dict[str, Callable[[torch.device, RankerConfig], Callable]] = {
    "siglip2": make_siglip2_ranker,
    "openclip_l14": make_openclip_l14_ranker,
    "eva02_clip_l14": make_eva02_clip_l14_ranker,
    "jina_embeddings_v4": make_jina_embeddings_v4_ranker,
    "colpali_v1_3": make_colpali_ranker,
    "colqwen2_5_v0_2": make_colqwen_ranker,
    "qwen3_vl_8b_yesno": make_qwen3_vl_8b_ranker,
    "qwen3_vl_32b_yesno": make_qwen3_vl_32b_ranker,
}


def build_ranker(model_name: str, device: torch.device, cfg: RankerConfig) -> Callable:
    if model_name not in RANKER_FACTORIES:
        raise ValueError(
            f"Unknown model_name: {model_name}. Known: {sorted(RANKER_FACTORIES)}"
        )
    return RANKER_FACTORIES[model_name](device, cfg)
