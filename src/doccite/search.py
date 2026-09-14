"""SQLite snapshot + BM25 + cosine search + reciprocal rank fusion."""

import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from doccite.embeddings import Embedder, make_embedder, tokens
from doccite.models import Chunk, Hit

SCHEMA_VERSION = 1


def build_index(path: Path, chunks: list[Chunk], metadata: dict, embedder: Embedder) -> dict:
    if not chunks:
        raise ValueError("Cannot index an empty corpus")
    vectors = embedder.encode([f"{c.title}\n{c.text}" for c in chunks])
    if len(vectors) != len(chunks) or not vectors[0]:
        raise ValueError("Embedding response has an invalid shape")
    dimensions = len(vectors[0])
    if any(len(v) != dimensions or not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("Embedding response contains invalid vectors")
    info = {
        **metadata,
        "schema_version": SCHEMA_VERSION,
        "embedding_backend": embedder.name,
        "dimensions": dimensions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="index-", suffix=".sqlite", dir=path.parent)
    os.close(fd)
    try:
        with sqlite3.connect(temporary) as db:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE chunks (id TEXT PRIMARY KEY, payload TEXT NOT NULL, vector TEXT NOT NULL);
            """)
            db.executemany(
                "INSERT INTO metadata VALUES (?, ?)", [(k, json.dumps(v)) for k, v in info.items()]
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)",
                [
                    (c.id, json.dumps(asdict(c)), json.dumps(v))
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        # Readers see either complete old or complete new index, never half an ingestion.
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return info


class SearchIndex:
    def __init__(self, path: Path, embedder: Embedder | None = None):
        if not path.is_file():
            raise ValueError("Index not found. Run: doccite ingest")
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            self.metadata = {k: json.loads(v) for k, v in db.execute("SELECT * FROM metadata")}
            rows = db.execute("SELECT payload, vector FROM chunks ORDER BY rowid").fetchall()
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Index schema has changed; rebuild with doccite ingest")
        self.chunks = [Chunk(**json.loads(payload)) for payload, _ in rows]
        self.vectors = [json.loads(vector) for _, vector in rows]
        self.embedder = embedder or make_embedder(self.metadata["embedding_backend"])
        if self.embedder.name != self.metadata["embedding_backend"]:
            raise ValueError("Embedding backend mismatch; rebuild the index")
        self.documents = [Counter(tokens(f"{c.title} {c.text}")) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.df = Counter(t for d in self.documents for t in d)

    def idf(self, term: str) -> float:
        n = len(self.chunks)
        return math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))

    def coverage(self, question: str, passage: str) -> float:
        query, doc = set(tokens(question)), set(tokens(passage))
        total = sum(self.idf(t) for t in query)
        return sum(self.idf(t) for t in query & doc) / total if total else 0.0

    def search(self, question: str, *, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in {"hybrid", "bm25", "dense"}:
            raise ValueError("Search mode must be hybrid, bm25, or dense")
        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20")
        query = set(tokens(question))
        if not query:
            return []
        qvector = self.embedder.encode([question])[0] if mode != "bm25" else None
        if qvector is not None and len(qvector) != self.metadata["dimensions"]:
            raise ValueError("Query embedding dimensions do not match index")
        bm25, cosine = [], []
        for doc, length, vector in zip(self.documents, self.lengths, self.vectors, strict=True):
            score = 0.0
            for term in query:
                freq = doc[term]
                denom = freq + 1.5 * (0.25 + 0.75 * length / (self.average_length or 1))
                score += self.idf(term) * freq * 2.5 / denom
            bm25.append(score)
            cosine.append(
                sum(a * b for a, b in zip(qvector, vector, strict=True)) if qvector else 0.0
            )
        lexical = sorted(
            (i for i, s in enumerate(bm25) if s > 0), key=lambda i: (-bm25[i], self.chunks[i].id)
        )[:50]
        dense = sorted(
            (i for i, s in enumerate(cosine) if s > 0),
            key=lambda i: (-cosine[i], self.chunks[i].id),
        )[:50]
        ranks = [lexical, dense] if mode == "hybrid" else [lexical if mode == "bm25" else dense]
        fused: Counter = Counter()
        for ranking in ranks:
            for rank, i in enumerate(ranking, 1):
                fused[i] += 1 / (60 + rank)
        selected = sorted(fused, key=lambda i: (-fused[i], self.chunks[i].id))[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=fused[i],
                bm25=bm25[i],
                cosine=cosine[i],
                coverage=self.coverage(question, f"{self.chunks[i].title} {self.chunks[i].text}"),
            )
            for i in selected
        ]
