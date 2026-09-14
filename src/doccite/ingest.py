"""Ingest only manifest-listed, checksum-verified Markdown; preserve exact lines."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from doccite.models import Chunk

DEFAULT_CORPUS = Path(__file__).parent / "corpus" / "httpx"
CHUNKER_VERSION = "markdown-lines-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_markdown(
    text: str, *, path: str, url: str, digest: str, max_chars: int = 1200
) -> list[Chunk]:
    """Pack Markdown blocks without rewriting source text.

    Headings start new sections. Fenced blocks stay intact and may exceed the
    soft character budget. Line spans are inclusive and one-based.
    """
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []
    start = 0
    fence: str | None = None
    block_start = 0
    blocks: list[tuple[int, int]] = []

    def emit(end: int) -> None:
        nonlocal start
        if end <= start or not "".join(lines[start:end]).strip():
            start = end
            return
        passage = "".join(lines[start:end])
        identity = f"{digest}:{path}:{start + 1}:{end}:{CHUNKER_VERSION}"
        chunks.append(
            Chunk(
                id=sha256(identity.encode())[:20],
                path=path,
                title=" > ".join(h for _, h in headings) or Path(path).stem,
                text=passage,
                start_line=start + 1,
                end_line=end,
                source_url=url,
                document_sha256=digest,
            )
        )
        start = end

    def pack(end: int) -> None:
        nonlocal block_start, blocks
        if end > block_start:
            blocks.append((block_start, end))
        for a, b in blocks:
            if a > start and len("".join(lines[start:b])) > max_chars:
                emit(a)
        emit(end)
        blocks = []
        block_start = end

    for i, line in enumerate(lines):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            pack(i)
            level, title = len(heading.group(1)), heading.group(2)
            headings = [(n, t) for n, t in headings if n < level] + [(level, title)]
        elif not line.strip():
            blocks.append((block_start, i + 1))
            block_start = i + 1
    pack(len(lines))
    return chunks


def load_corpus(root: Path, *, max_chars: int = 1200) -> tuple[list[Chunk], dict]:
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest revision must be a full 40-character Git commit SHA")
    repo = manifest.get("repository", "").rstrip("/")
    parsed = urlparse(repo)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or len(parsed.path.strip("/").split("/")) != 2
    ):
        raise ValueError("Manifest repository must be an HTTPS GitHub repository URL")
    chunks = []
    seen = set()
    for entry in manifest["files"]:
        path = entry["path"]
        dest = (root / path).resolve()
        if not dest.is_relative_to(root) or not path.endswith(".md") or path in seen:
            raise ValueError(f"Unsafe, duplicate, or non-Markdown source path: {path}")
        seen.add(path)
        expected_url = f"{repo}/blob/{manifest['revision']}/{path}"
        if entry["url"] != expected_url:
            raise ValueError(f"Source URL does not match pinned revision: {path}")
        data = dest.read_bytes()
        digest = sha256(data)
        if digest != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        chunks.extend(
            chunk_markdown(
                data.decode("utf-8"),
                path=path,
                url=entry["url"],
                digest=digest,
                max_chars=max_chars,
            )
        )
    if not chunks:
        raise ValueError("Corpus has no indexable passages")
    metadata = {
        "library": manifest["library"],
        "version": manifest["version"],
        "revision": manifest["revision"],
        "manifest_sha256": sha256(manifest_bytes),
        "document_count": len(seen),
        "chunk_count": len(chunks),
        "chunker": CHUNKER_VERSION,
        "max_chars": max_chars,
    }
    return chunks, metadata
