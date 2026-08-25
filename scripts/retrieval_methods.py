import json
import logging
import os

import get_from_paper
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from FlagEmbedding import BGEM3FlagModel, FlagReranker

logging.disable(logging.CRITICAL)

DEVICE = "cuda:0"
# sdr lives in the released nlpeer tree: nlpeer/<venue>/<paper_id>/v1/paper.sdr.json
NLPEER_ROOT = os.environ.get("REGROUND_NLPEER", "nlpeer")


def load_sdr(paper_id, venue, nlpeer=None):
    path = os.path.join(nlpeer or NLPEER_ROOT, venue, paper_id, "v1", "paper.sdr.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_by_type(sdr, ref_type):
    doc_ids = []
    doc_texts = []

    if ref_type == "Line":
        for ln, text in sdr["lines"].items():
            doc_ids.append(f"Line {ln}")
            doc_texts.append(text)
        return doc_ids, doc_texts

    if ref_type == "Paragraph":
        for para in sdr["paragraphs"]:
            doc_ids.append(f"Paragraph {para['paragraph_id']}")
            doc_texts.append(
                get_from_paper.get_text_by_line(
                    sdr, (para["start_line"], para["end_line"])
                )
            )
        return doc_ids, doc_texts

    if ref_type == "Section":
        for sec in sdr["sections"]:
            doc_ids.append(f"Section {sec['number']} / {sec['name']}")
            doc_texts.append(
                get_from_paper.get_section_text_by_number(sdr, sec["number"])
            )
        return doc_ids, doc_texts

    if ref_type == "Caption":
        return list(sdr["captions"].keys()), list(sdr["captions"].values())

    raise ValueError(ref_type)


def get_whole_paper_text(sdr):
    doc_ids = []
    doc_texts = []

    for ln, text in sdr["lines"].items():
        doc_ids.append(f"Line {ln}")
        doc_texts.append(text)

    for para in sdr["paragraphs"]:
        doc_ids.append(f"Paragraph {para['paragraph_id']}")
        doc_texts.append(para["text"])

    for sec in sdr["sections"]:
        doc_ids.append(f"Section {sec['number']} / {sec['name']}")
        doc_texts.append(
            get_from_paper.get_section_text_by_number(sdr, sec["number"])
        )

    for cid, text in sdr["captions"].items():
        doc_ids.append(cid)
        doc_texts.append(text)

    return doc_ids, doc_texts


class BM25Retriever:
    def __init__(self, sdr):
        self.sdr = sdr
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _index(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]

        ids, texts = get_whole_paper_text(self.sdr)
        bm25 = BM25Okapi([t.lower().split() for t in texts])
        self.cache[pid] = (ids, bm25)
        return ids, bm25

    def rank(self, query):
        ids, bm25 = self._index()
        scores = bm25.get_scores(query.lower().split())
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class SPLADEv3Retriever:
    def __init__(self, sdr):
        self.sdr = sdr
        self.model = SparseEncoder("naver/splade-v3", device=DEVICE)
        self.cache = {}
        if hasattr(self.model, "eval"):
            self.model.eval()

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _embeddings(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]

        ids, texts = get_whole_paper_text(self.sdr)
        emb = self.model.encode_document(texts)
        self.cache[pid] = (ids, emb)
        return ids, emb

    def rank(self, query):
        ids, d_emb = self._embeddings()
        q_emb = self.model.encode_query([query])
        scores = self.model.similarity(q_emb, d_emb)[0].cpu().numpy()
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class MPNetRetriever:
    def __init__(self, sdr):
        self.sdr = sdr
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _index(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]

        ids, texts = get_whole_paper_text(self.sdr)
        emb = self.model.encode_document(
            texts, convert_to_tensor=True, batch_size=128, device=DEVICE
        )
        self.cache[pid] = (ids, emb)
        return ids, emb

    def rank(self, query):
        ids, emb = self._index()
        q = self.model.encode_query([query], convert_to_tensor=True, device=DEVICE)
        scores = self.model.similarity(q, emb)[0].cpu().numpy()
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class BGEM3Retriever:
    def __init__(self, sdr):
        self.sdr = sdr
        self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=DEVICE)
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _embeddings(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]

        ids, texts = get_whole_paper_text(self.sdr)
        emb = self.model.encode(texts, batch_size=64)["dense_vecs"]
        self.cache[pid] = (ids, emb)
        return ids, emb

    def rank(self, query):
        ids, emb = self._embeddings()
        q = self.model.encode([query])["dense_vecs"][0]
        scores = q @ emb.T
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class MiniLMCrossEncoderRetriever:
    def __init__(self, sdr, batch_size=64):
        self.sdr = sdr
        self.batch_size = batch_size
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", device=DEVICE)
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _docs(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]
        ids, texts = get_whole_paper_text(self.sdr)
        self.cache[pid] = (ids, texts)
        return ids, texts

    def rank(self, query):
        ids, texts = self._docs()
        scores = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            pairs = [(query, t) for t in batch]
            scores[i : i + len(batch)] = self.model.predict(
                pairs, batch_size=len(pairs), convert_to_numpy=True
            )
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class BGEM3RerankerRetriever:
    def __init__(self, sdr, batch_size=32):
        self.sdr = sdr
        self.batch_size = batch_size
        self.model = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _docs(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]
        ids, texts = get_whole_paper_text(self.sdr)
        self.cache[pid] = (ids, texts)
        return ids, texts

    def rank(self, query):
        ids, texts = self._docs()
        scores = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            pairs = [[query, t] for t in batch]
            scores[i : i + len(batch)] = self.model.compute_score(pairs)
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class _SentenceEmbeddingRetriever:
    """Dense retriever over a SentenceTransformer with query/document prompts.
    Shared base for the Gemma and Qwen3 embedding retrievers."""

    MODEL_ID = None  # set by subclasses

    def __init__(self, sdr, batch_size=128):
        self.sdr = sdr
        self.batch_size = batch_size
        self.model = SentenceTransformer(self.MODEL_ID, device=DEVICE)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.cache = {}

    def set_sdr(self, sdr):
        self.sdr = sdr
        self.cache.clear()

    def _index(self):
        pid = self.sdr["metadata"]["paper_id"]
        if pid in self.cache:
            return self.cache[pid]
        ids, texts = get_whole_paper_text(self.sdr)
        emb = self.model.encode_document(
            texts, convert_to_tensor=True, batch_size=self.batch_size, device=DEVICE
        )
        self.cache[pid] = (ids, emb)
        return ids, emb

    def rank(self, query):
        ids, emb = self._index()
        q = self.model.encode_query([query], convert_to_tensor=True, device=DEVICE)
        scores = self.model.similarity(q, emb)[0].cpu().numpy()
        order = np.argsort(scores)[::-1]
        return {ids[i]: float(scores[i]) for i in order}


class GEMMA300Retriever(_SentenceEmbeddingRetriever):
    MODEL_ID = "google/embeddinggemma-300m"


class Qwen3EmbeddingRetriever(_SentenceEmbeddingRetriever):
    MODEL_ID = "Qwen/Qwen3-Embedding-4B"
