import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.services.chunker import create_document_chunks


# ==================================================
# CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

INDEX_DIR = BASE_DIR / "storage" / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_FILE = "chunks.json"
VECTOR_FILE = "vectors.faiss"

RRF_K = 60


# ==================================================
# EMBEDDING MODEL
# ==================================================

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model only once.
    """

    global _embedding_model

    if _embedding_model is None:

        print("Loading embedding model...")

        _embedding_model = SentenceTransformer(
            MODEL_NAME
        )

        print("Embedding model loaded.")

    return _embedding_model


# ==================================================
# TOKENIZATION
# ==================================================

def tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer for BM25.
    """

    return re.findall(
        r"\b[\w.%-]+\b",
        text.lower()
    )


# ==================================================
# DOCUMENT INDEXING
# ==================================================

def build_document_index(
    document_id: str,
    pages: List[Dict]
) -> Dict:
    """
    Create chunks and FAISS vector index.

    BM25 is built later in memory when the document
    is loaded for retrieval.
    """

    print(f"Creating chunks for: {document_id}")

    chunks = create_document_chunks(
        pages=pages,
        document_id=document_id
    )

    if not chunks:
        raise ValueError(
            "No searchable content found in document."
        )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    model = get_embedding_model()

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    vector_index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_index.add(
        embeddings
    )

    document_dir = INDEX_DIR / document_id

    document_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        vector_index,
        str(document_dir / VECTOR_FILE)
    )

    # Save chunks and metadata
    with open(
        document_dir / CHUNK_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    text_chunks = sum(
        chunk.get("source_type") == "text"
        for chunk in chunks
    )

    table_chunks = sum(
        str(chunk.get("source_type", "")).startswith(
            "table"
        )
        for chunk in chunks
    )

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "text_chunks": text_chunks,
        "table_chunks": table_chunks
    }


# ==================================================
# DOCUMENT RETRIEVER
# ==================================================

class DocumentRetriever:
    """
    Loads one indexed document and keeps all retrieval
    resources in memory.

    This prevents:
    - repeated FAISS disk loading
    - repeated chunks.json loading
    - repeated BM25 index construction
    """

    def __init__(
        self,
        document_id: str
    ):

        self.document_id = document_id

        document_dir = (
            INDEX_DIR / document_id
        )

        vector_path = (
            document_dir / VECTOR_FILE
        )

        chunk_path = (
            document_dir / CHUNK_FILE
        )

        if not vector_path.exists():

            raise FileNotFoundError(
                f"FAISS index not found for "
                f"document: {document_id}"
            )

        if not chunk_path.exists():

            raise FileNotFoundError(
                f"Chunk file not found for "
                f"document: {document_id}"
            )

        print(
            f"Loading document index: "
            f"{document_id}"
        )

        self.vector_index = faiss.read_index(
            str(vector_path)
        )

        with open(
            chunk_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.chunks = json.load(file)

        print("Building BM25 index...")

        self.tokenized_corpus = [
            tokenize(
                chunk["content"]
            )
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print(
            f"Retriever ready with "
            f"{len(self.chunks)} chunks."
        )

    # ----------------------------------------------
    # SEMANTIC SEARCH
    # ----------------------------------------------

    def semantic_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:

        if not self.chunks:
            return []

        model = get_embedding_model()

        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        query_vector = np.asarray(
            query_vector,
            dtype=np.float32
        )

        scores, indices = (
            self.vector_index.search(
                query_vector,
                min(top_k, len(self.chunks))
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append({
                "chunk": self.chunks[int(index)],
                "semantic_score": float(score)
            })

        return results

    # ----------------------------------------------
    # BM25 SEARCH
    # ----------------------------------------------

    def keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            results.append({
                "chunk": self.chunks[int(index)],
                "keyword_score": score
            })

        return results

    # ----------------------------------------------
    # HYBRID SEARCH
    # ----------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:

        if not query or not query.strip():

            raise ValueError(
                "Query cannot be empty."
            )

        if not self.chunks:

            return []

        search_depth = min(
            max(top_k * 3, 10),
            len(self.chunks)
        )

        semantic_results = (
            self.semantic_search(
                query=query,
                top_k=search_depth
            )
        )

        keyword_results = (
            self.keyword_search(
                query=query,
                top_k=search_depth
            )
        )

        fused_results = (
            reciprocal_rank_fusion(
                semantic_results,
                keyword_results
            )
        )

        return fused_results[:top_k]


# ==================================================
# RECIPROCAL RANK FUSION
# ==================================================

def reciprocal_rank_fusion(
    semantic_results: List[Dict],
    keyword_results: List[Dict]
) -> List[Dict]:

    fused_scores = {}
    chunk_lookup = {}

    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        chunk = result["chunk"]

        chunk_id = chunk["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0
            )
            + 1 / (RRF_K + rank)
        )

        chunk_lookup[chunk_id] = chunk

    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        chunk = result["chunk"]

        chunk_id = chunk["chunk_id"]

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0
            )
            + 1 / (RRF_K + rank)
        )

        chunk_lookup[chunk_id] = chunk

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id:
            fused_scores[chunk_id],
        reverse=True
    )

    results = []

    for chunk_id in ranked_ids:

        results.append({
            **chunk_lookup[chunk_id],
            "rrf_score": fused_scores[
                chunk_id
            ]
        })

    return results


# ==================================================
# BACKWARD-COMPATIBLE FUNCTION
# ==================================================

def load_document_index(
    document_id: str
) -> Tuple:

    """
    Kept for compatibility with existing code.
    """

    document_dir = INDEX_DIR / document_id

    vector_path = (
        document_dir / VECTOR_FILE
    )

    chunk_path = (
        document_dir / CHUNK_FILE
    )

    if (
        not vector_path.exists()
        or not chunk_path.exists()
    ):

        raise FileNotFoundError(
            "Document has not been indexed yet."
        )

    vector_index = faiss.read_index(
        str(vector_path)
    )

    with open(
        chunk_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return vector_index, chunks


def retrieve_evidence(
    document_id: str,
    query: str,
    top_k: int = 5
) -> List[Dict]:

    """
    Backward-compatible retrieval function.

    For multiple questions, prefer creating one
    DocumentRetriever and reusing it.
    """

    retriever = DocumentRetriever(
        document_id
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k
    )