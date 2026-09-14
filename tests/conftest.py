import pytest

from doccite.embeddings import HashEmbedder
from doccite.ingest import DEFAULT_CORPUS, load_corpus
from doccite.search import SearchIndex, build_index


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(DEFAULT_CORPUS)


@pytest.fixture(scope="session")
def index_path(tmp_path_factory, corpus):
    path = tmp_path_factory.mktemp("index") / "index.sqlite"
    build_index(path, *corpus, HashEmbedder())
    return path


@pytest.fixture(scope="session")
def index(index_path):
    return SearchIndex(index_path)
