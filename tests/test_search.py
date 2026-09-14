import math

import pytest

from doccite.embeddings import HashEmbedder
from doccite.search import SearchIndex, build_index


def test_hash_embeddings_are_deterministic_normalized_and_distinguish_text():
    encoder = HashEmbedder()
    a, b, c = encoder.encode(["read timeout", "read timeout", "banana orchard"])
    assert a == b and a != c
    assert math.isclose(sum(x * x for x in a), 1)
    assert encoder.encode([""])[0] == [0.0] * 512


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_known_passage(index, mode):
    hits = index.search("What are the four types of timeouts?", mode=mode)
    assert any(
        h.chunk.path == "docs/advanced/timeouts.md" and "four different" in h.chunk.text
        for h in hits
    )
    assert all(math.isfinite(h.score) for h in hits)


def test_empty_and_stopword_query(index):
    assert index.search("") == []
    assert index.search("how do I use the") == []


def test_mismatch_is_rejected(index_path):
    encoder = HashEmbedder()
    encoder.name = "different-backend"
    with pytest.raises(ValueError, match="mismatch"):
        SearchIndex(index_path, encoder)


def test_failed_rebuild_preserves_original_index(tmp_path, corpus):
    path = tmp_path / "index.sqlite"
    build_index(path, *corpus, HashEmbedder())
    original = path.read_bytes()

    class BrokenEmbedder:
        name = "broken"

        def encode(self, texts):
            raise RuntimeError("simulated embedding outage")

    with pytest.raises(RuntimeError):
        build_index(path, *corpus, BrokenEmbedder())
    assert path.read_bytes() == original


@pytest.mark.parametrize("k", [0, 21, -1])
def test_k_bounds(index, k):
    with pytest.raises(ValueError):
        index.search("timeout", k=k)
