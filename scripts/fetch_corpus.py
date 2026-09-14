"""Restore checksum-pinned public source files; never crawl arbitrary URLs.

Run from the project root: python scripts/fetch_corpus.py
The corpus is already bundled, so this is optional and uses the network.
"""

import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "src/doccite/corpus/httpx"
    manifest = json.loads((root / "manifest.json").read_text())
    revision = manifest["revision"]
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("Expected a pinned commit SHA")
    downloads = []
    for entry in manifest["files"]:
        path = entry["path"]
        destination = (root / path).resolve()
        if not destination.is_relative_to(root) or not path.startswith("docs/"):
            raise ValueError("Unexpected source path")
        url = f"https://raw.githubusercontent.com/encode/httpx/{revision}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(1_000_001)
        if len(data) > 1_000_000 or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source validation failed: {path}")
        downloads.append((destination, data))
    # All responses must pass verification before any existing source is replaced.
    for destination, data in downloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
            os.replace(temp, destination)
        finally:
            Path(temp).unlink(missing_ok=True)
    print(f"Verified and restored {len(downloads)} files at {revision}")


if __name__ == "__main__":
    main()
