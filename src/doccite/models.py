from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    id: str
    path: str
    title: str
    text: str
    start_line: int
    end_line: int
    source_url: str
    document_sha256: str

    @property
    def url(self) -> str:
        return f"{self.source_url}#L{self.start_line}-L{self.end_line}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    bm25: float
    cosine: float
    coverage: float


@dataclass(frozen=True)
class Citation:
    id: str
    chunk_id: str
    path: str
    quote: str
    start_line: int
    end_line: int
    url: str
    document_sha256: str


@dataclass(frozen=True)
class Claim:
    text: str
    citation_ids: list[str]


@dataclass
class Answer:
    question: str
    status: str  # answered | abstained
    reason: str
    claims: list[Claim] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def markdown(self) -> str:
        if self.status == "abstained":
            return (
                f"I couldn't establish an answer from the indexed documentation.\n\n{self.reason}"
            )
        lines = []
        for claim in self.claims:
            refs = " ".join(
                f"[{c.id}]({c.url})" for c in self.citations if c.id in claim.citation_ids
            )
            lines.append(f"{claim.text}\n\n{refs}")
        lines.append("### Source passages")
        for c in self.citations:
            quote = "\n".join("> " + line for line in c.quote.splitlines())
            lines.append(f"**[{c.id}]({c.url})** · `{c.path}:{c.start_line}`\n\n{quote}")
        return "\n\n".join(lines)
