"""Interchangeable local embeddings. Hashing is a baseline, not a neural model."""

import hashlib
import math
import re
from collections import Counter
from typing import Protocol

STOPWORDS = set(
    """a an the is are was were be been being to of in on at for from by
with and or as that this these those it its i me my you your we our they their how
what when where why which who do does did can could would should will may might
use using used httpx python please tell explain about have has had get set make
way need want also than then into through there without all any some if so per
""".split()
)


def tokens(text: str) -> list[str]:
    # Split snake_case and dotted identifiers; keep digits (HTTP/2, version, defaults).
    result = re.findall(r"[a-z0-9]+", text.lower())
    return [
        t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t
        for t in result
        if t not in STOPWORDS
    ]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic signed unigram/bigram feature hashing; zero downloads."""

    name = "hash-v1-512"

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = tokens(text)
            features = words + [f"{a}:{b}" for a, b in zip(words, words[1:], strict=False)]
            vector = [0.0] * 512
            for feature, count in Counter(features).items():
                digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "big") % len(vector)
                vector[slot] += (1 if digest[4] & 1 else -1) * (1 + math.log(count))
            vectors.append(normalize(vector))
        return vectors


class SentenceTransformerEmbedder:
    MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # Pin model weights, tokenizer and configuration for reproducible embeddings.
    REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    name = f"sentence-transformer:{MODEL}@{REVISION}"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ValueError('Install neural embeddings with: pip install ".[semantic]"') from exc
        self.model = SentenceTransformer(self.MODEL, revision=self.REVISION, device="cpu")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        ).tolist()


def make_embedder(name: str) -> Embedder:
    if name in {"hash", HashEmbedder.name}:
        return HashEmbedder()
    if name in {"semantic", SentenceTransformerEmbedder.name}:
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {name}")
