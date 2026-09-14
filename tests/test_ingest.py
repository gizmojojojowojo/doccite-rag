import json
import shutil

import pytest

from doccite.ingest import DEFAULT_CORPUS, chunk_markdown, load_corpus


def test_all_passages_match_original_lines(corpus):
    chunks, metadata = corpus
    assert metadata["document_count"] == 12
    assert len({c.id for c in chunks}) == len(chunks)
    for chunk in chunks:
        lines = (DEFAULT_CORPUS / chunk.path).read_bytes().decode().splitlines(keepends=True)
        assert chunk.text == "".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert metadata["revision"] in chunk.url


def test_chunking_covers_source_once_and_ignores_headings_inside_fences():
    text = "# Title\n\nIntro.\n\n```python\n# Not a heading\nx = 3\n```\n\n## Next\n\nFinal."
    chunks = chunk_markdown(
        text, path="a.md", url="https://example.com", digest="digest", max_chars=100
    )
    assert "".join(c.text for c in chunks) == text
    assert chunks[-1].title == "Title > Next"
    assert all("Not a heading" not in c.title for c in chunks)
    assert any("```python\n# Not a heading\nx = 3\n```" in c.text for c in chunks)


@pytest.mark.parametrize("text", ["one line", "one\r\ntwo\r\n", "# Heading\n\n", "évidence\n", ""])
def test_chunker_edge_cases(text):
    chunks = chunk_markdown(text, path="a.md", url="https://example.com", digest="digest")
    assert "".join(c.text for c in chunks) == text


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("checksum", "checksum"),
        ("path", "Unsafe"),
        ("url", "pinned"),
        ("revision", "revision"),
        ("duplicate", "duplicate"),
    ],
)
def test_rejects_modified_or_unsafe_corpus(tmp_path, mutation, match):
    root = tmp_path / "corpus"
    shutil.copytree(DEFAULT_CORPUS, root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["files"][0]
    if mutation == "checksum":
        (root / entry["path"]).write_text("altered documentation")
    elif mutation == "path":
        entry["path"] = "../../outside.md"
    elif mutation == "url":
        entry["url"] = "https://evil.example/source"
    elif mutation == "revision":
        manifest["revision"] = "main"
    else:
        manifest["files"].append(entry.copy())
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        load_corpus(root)
